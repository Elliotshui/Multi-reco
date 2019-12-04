import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, x_len, y_len, start_token, learning_rate = 0.01, reward_gamma = 0.95):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.x_len = x_len
        self.y_len = y_len
        self.start_token = tf.constant([start_token] * self.batch_size, dtype = tf.int32)
        self.g_params = []
        self.learning_rate = learning_rate
        self.reward_gamma = reward_gamma
        self.grad_clip = 5.0

        # Variable definition
        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            self.encoder_unit = self.create_encoder_unit(self.g_params)
            self.decoder_unit = self.create_decoder_unit(self.g_params)
            self.output_unit = self.create_output_unit(self.g_params)

        # Placeholder definition
        self.x = tf.placeholder(tf.int32, shape = [self.batch_size, self.x_len])
        self.y = tf.placeholder(tf.int32, shape = [self.batch_size, self.y_len])
        self.delta_d_score = tf.placeholder(tf.float32, shape = [self.batch_size, self.y_len])

        # Embedding lookup
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm = [1,0,2])
            self.processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.y), perm = [1,0,2])
        
        emb_x = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.x_len)
        emb_x = emb_x.unstack(self.processed_x)
        emb_y = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.y_len)
        emb_y = emb_y.unstack(self.processed_y)

        # Initial states
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        # Encoder
        def encoder(i, h):
            x = emb_x.read(i)
            h_t = self.encoder_unit(x, h)
            return i + 1, h_t

        # Decoder for generation   
        gen_o = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.y_len,
                                             dynamic_size = False, infer_shape = True)
        gen_y = tensor_array_ops.TensorArray(dtype = tf.int32, size = self.y_len,
                                             dynamic_size = False, infer_shape = True)
        def g_decoder(i, x, h, gen_o, gen_y):
            h_t = self.decoder_unit(x, h)
            o_t = self.output_unit(h_t)
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_t = tf.nn.embedding_lookup(self.g_embeddings, next_token)
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                                                             tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_y = gen_y.write(i, next_token)
            return i + 1, x_t, h_t, gen_o, gen_y

        # Decoder for pretraining
        def pre_decoder(i, x, h, pred):
            h_t = self.decoder_unit(x, h)
            o_t = self.output_unit(h_t)
            pred = pred.write(i, tf.nn.softmax(o_t))
            x_t = emb_y.read(i)
            return i + 1, x_t, h_t, pred

        # Encoding
        _, self.h_e = control_flow_ops.while_loop(
            cond = lambda i, _1: i < self.x_len,
            body = encoder,
            loop_vars = (tf.constant(0, dtype = tf.int32), self.h0)
        )

        # Generation decoding
        _, _, _, self.gen_o, self.gen_y = control_flow_ops.while_loop(
            cond = lambda i, _1, _2, _3, _4: i < self.y_len,
            body = g_decoder,
            loop_vars = (tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h_e, gen_o, gen_y)
        )

        self.gen_y = self.gen_y.stack()
        self.gen_y = tf.transpose(self.gen_y, perm = [1, 0])

        # Pretraining decoding
        predictions = tensor_array_ops.TensorArray(
            dtype = tf.float32, size = self.y_len,
            dynamic_size = False, infer_shape = True
        )
        _, _, _, self.predictions = control_flow_ops.while_loop(
            cond = lambda i, _1, _2, _3: i < self.y_len,
            body = pre_decoder,
            loop_vars = (tf.constant(0, dtype = tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h_e, predictions)
        )

        self.predictions = tf.transpose(self.predictions.stack(), perm = [1, 0, 2])

        # Pretraining loss
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.y, [-1])), self.num_emb, 1.0, 0.0) *
            tf.log(tf.clip_by_value(tf.reshape(self.predictions, [-1, self.num_emb]), 1e-20, 1.0))
        ) / (self.y_len * self.batch_size)
        
        # Pretrain updates
        pretrain_opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

        # Adversarial training loss
        self.adv_loss = tf.reduce_sum(
            (tf.one_hot(tf.to_int32(tf.reshape(self.gen_y, [-1])), self.num_emb, 1.0, 0.0) *
            tf.log(tf.clip_by_value(tf.reshape(self.predictions, [-1, self.num_emb]), 1e-20, 1.0))) *
            tf.reshape(self.delta_d_score, [-1]))
        ) / (self.y_len * self.batch_size)

        # Adversarial updates
        adv_opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.adv_grad, _ = tf.clip_by_global_norm(tf.gradients(self.adv_loss, self.g_params)),
        self.adv_updates = adv_opt.apply_gradients(zip(self.adv_grad, self.g_params))


    def generate(self, sess, x):
        outputs = sess.run(self.gen_y, feed_dict = {self.x, x})
        return outputs

    def pretrain_step(self, sess, x, y):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict = {self.x: x, self.y: y})
        return outputs

    def adv_step(self, sess, x, delta_d_score):
        outputs = sess.run([self.adv_updates, self.adv_loss], feed_dict = {self.x: x, self.delta_d_score: delta_d_score})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev = 0.1)

    def create_encoder_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wie = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uie = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bie = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wfe = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ufe = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bfe = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Woge = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uoge = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.boge = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wce = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uce = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bce = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wie, self.Uie, self.bie,
            self.Wfe, self.Ufe, self.bfe,
            self.Woge, self.Uoge, self.boge,
            self.Wce, self.Uce, self.bce])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wie) +
                tf.matmul(previous_hidden_state, self.Uie) + self.bie
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wfe) +
                tf.matmul(previous_hidden_state, self.Ufe) + self.bfe
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Woge) +
                tf.matmul(previous_hidden_state, self.Uoge) + self.boge
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wce) +
                tf.matmul(previous_hidden_state, self.Uce) + self.bce
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit
    
    def create_decoder_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit
    
    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit