import numpy as np
import tensorflow as tf
import random
from dataloader import Dataloader
from discriminator import Discriminator
from generator import Generator

# Generator Hyper-parameters
EMB_DIM = 32
HIDDEN_DIM = 32
X_LEN = 4
Y_LEN = 4
START_TOKEN = 0
GPRE_EPOCH_NUM = 120
DPRE_EPOCH_NUM = 120
SEED = 88
BATCH_SIZE = 64

filepath = "yoochoose-clicks.dat"

def g_pretrain_epoch(sess, model, dataloader):
    losses = []
    dataloader.reset_pointer()

    for i in range(dataloader.num_batch):
        x_batch, y_batch = dataloader.next_batch()
        _, loss = model.pretrain_step(sess, x_batch, y_batch)
        losses.append(loss)
    
    return np.mean(losses)

def d_pretrain_epoch(sess, generator, discriminator, dataloader):
    losses = []
    dataloader.reset_pointer()
    
    for i in range(dataloader.num_batch):
        x_batch, y_batch = dataloader.next_batch()
        gen_y = generator.generate(sess, x_batch) # batch_size * y_len
        _, loss = discriminator.d_step(sess, x_batch, y_batch, gen_y)
        losses.append(loss)
    
    return np.mean(losses)

def adv_train(sess, genenrator, discriminator, dataloader):
    dataloader.reset_pointer()
    for i in range(dataloader.num_batch):
        # train the generator for on step
        x_batch, y_batch = dataloader.next_batch()
        gen_y = genenrator.generate(sess, x_batch)
        d_score, delta_d_score = discriminator.discriminate(sess, x_batch, gen_y)
        adv_loss = generator.adv_step(sess, x_batch, delta_d_score)

        # train the discriminator
        for i in range(3):
            


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # load data
    print("loading data...")
    dataloader = Dataloader(filepath, X_LEN, Y_LEN, BATCH_SIZE)
    dataloader.load()
    item_num = dataloader.get_item_num()
    dataloader.create_batchs()
    print("data loaded.")
    print("item_num: " + str(dataloader.item_num))

    generator = Generator(item_num, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, X_LEN, Y_LEN, START_TOKEN)
    discriminator = Discriminator(item_num, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, X_LEN, Y_LEN, START_TOKEN)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())

    # start pretraining generator
    log = open("save/experiment-log.txt", "w+")
    print("start pretraining generator")
    log.write("pretraining generator...\n")
    for epoch in range(GPRE_EPOCH_NUM):
        loss = g_pretrain_epoch(sess, generator, dataloader)
        if epoch % 5 == 0:
            buffer = "epoch:\t" + str(epoch) + "\tloss:\t" + str(loss) + "\n"
            log.write(buffer)

    # start pretraining discriminator
    print("start pretraining discirminator")
    log.write("pretraining discriminator...\n")   
    for epoch in range(DPRE_EPOCH_NUM):
        loss = d_pretrain_epoch(sess, generator, discriminator, dataloader)
        if epoch % 5 == 0:
            buffer = "epoch:\t" + str(epoch) + "\tloss:\t" + str(loss) + "\n"
            log.write(buffer)

    # start adversarial training
    print("start adversarial training")
    log.write("adversarial training\n")

    for total_batch in range(200):
        



if __name__ == '__main__':
    main()