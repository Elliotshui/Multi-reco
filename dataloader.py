import numpy as np
class Dataloader(object):
    
    def __init__(self, filename, x_len, y_len, batch_size):
        self.filename = filename
        self.x_len = x_len
        self.y_len = y_len
        self.batch_size = batch_size
        self.item_lists = {}
        self.x_batches = []
        self.y_batches = []
        self.items = {}
        self.item_num = 1   # start_token included

    def load(self):
        f = open(self.filename, 'r')
        item_id = {}
        for line in f:
            line = line.split(',')
            session_id = int(line[0])
            if session_id not in item_id:
                item_id[session_id] = []
            item = int(line[2])
            if item not in self.items:
                self.items[item] = self.item_num
                self.item_num += 1
            item_id[session_id].append(self.items[item])
        del_id = []
        for session_id in item_id:
            if len(item_id[session_id]) < self.x_len + self.y_len:
                del_id.append(session_id)
        for session_id in del_id:
            del item_id[session_id]
        self.item_lists = item_id
        return
    
    def create_batchs(self):
        for session_id in self.item_lists:
            x = self.item_lists[session_id][0:self.x_len]
            y = self.item_lists[session_id][self.x_len: self.x_len + self.y_len]
            self.x_batches.append(x)
            self.y_batches.append(y)

        self.num_batch = int(len(self.x_batches) / self.batch_size)
        self.x_batches = self.x_batches[0: self.num_batch * self.batch_size]
        self.y_batches = self.y_batches[0: self.num_batch * self.batch_size]
        self.x_batches = np.split(np.array(self.x_batches), self.num_batch, 0)
        self.y_batches = np.split(np.array(self.y_batches), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        x_batch = self.x_batches[self.pointer]
        y_batch = self.y_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return x_batch, y_batch

    def reset_pointer(self):
        self.pointer = 0

    def get_item_num(self):
        return self.item_num
