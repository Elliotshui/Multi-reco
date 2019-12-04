import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Discriminator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, x_len, y_len, start_token, learning_rate = 0.01):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.x_len = x_len
        self.y_len = y_len
        # self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable = False)
        self.d_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0

        # Variable definition
        with tf.variable_scope('discriminator'):
            self.d_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.d_params.append(d_embeddings)
            self.encoder_unit = self.create_encoder_unit(self.d_params)
            self.decoder_unit = self.create_decoder_unit(self.d_params)
            self.output_unit = self.create_output_unit(self.d_params)

        # Placeholder definition
        self.x = tf.placeholder(tf.int32, shape = [self.batch_size, self.x_len])
        self.y = tf.placeholder(tf.int32, shape = [self.batch_size, self.y_len])
        self.gen_y = tf.placeholder(tf.int32, shape = [self.batch_size, self.y_len])

        # Embedding lookup
        with tf.device("cpu/0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.d_embeddings, self.x), perm = [1, 0 ,2])
            self.processed_y = tf.transpose(tf.nn.embedding_lookup(self.d_embeddings, self.y), perm = [1, 0 ,2])
            self.processed_gen_y = tf.transpose(tf.nn.embedding_lookup(self.d_embeddings, self.gen_y), perm = [1, 0 ,2])

        emb_x = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.x_len)
        emb_x = emb_x.unstack(self.processed_x)
        emb_y = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.y_len)
        emb_y = emb_y.unstack(self.processed_y)
        emb_gen_y = tensor_array_ops.TensorArray(dtype = tf.float32, size = self.y_len)
        emb_gen_y = emb_gen_y.unstack(self.processed_gen_y)

        # Initial states 
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        # Encoder
        def encoder(i, h):
            x = emb_x.read(i)
            h_t = self.encoder_unit(x, h)
            return i + 1, h_t
        _, self.h_e = control_flow_ops.while_loop(
            cond = lambda i, _1: i < self.x_len,
            body = encoder,
            loop_vars = (tf.constant(0, dtype = tf.int32), self.h0)
        )

        # Decoder for discrimination
        delta_pos_d_score = tensor_array_ops.TensorArray(
            dtype = tf.float32, size = self.y_len,
            dynamic_size = false, infer_shape = True
        )
        pos_d_score = tensor_array_ops.TensorArray(
            dtype = tf.float32, size = self.y_len,
            dynamic_size = false, infer_shape = True
        )
        delta_neg_d_score = tensor_array_ops.TensorArray(
            dtype = tf.float32, size = self.y_len,
            dynamic_size = false, infer_shape = True
        )     
        neg_d_score = tensor_array_ops.TensorArray(
            dtype = tf.float32, size = self.y_len,
            dynamic_size = false, infer_shape = True
        )       
        def decoder(i, h, o, d_score, delta_d_score, emb):
            x = emb.read(i)
            h_t = self.decoder_unit(x, h)
            o_t = self.output_unit(h_t)
            d_score = d_score.write(i, o_t)
            delta_d_score = delta_d_store.write(i, o_t - o)
            return i + 1, h_t, o_t, d_score, delta_d_score, emb

        _, _, self.pos_d_score, self.delta_pos_d_score = control_flow_ops.while_loop(
            cond = lambda i, _1, _2, _3, _4, _5: i < self.y_len,
            body = decoder,
            loop_vars = (tf.constant(0, dtype = tf.int32), self.h_e, tf.constant(0, dtype = tf.float32), pos_d_score, delta_pos_d_score, emb_y)
        )
        self.pos_d_score = self.pos_d_score.unstack()
        self.pos_d_score = tf.transpose(self.pos_d_score, perm = [1, 0])
        self.delta_pos_d_score = self.delta_pos_d_score.unstack()
        self.delta_pos_d_score = tf.transpose(self.delta_pos_d_score, perm = [1, 0])
        # self.pos_d_score = tf.reshape(tf.log(self.pos_d_score), [-1])

        _, _, self.neg_d_score = control_flow_ops.while_loop(
            cond = lambda i, _1, _2, _3, _4, _5: i < self.y_len,
            body = decoder,
            loop_vars = (tf.constant(0, dtype = tf.int32), self.h_e, tf.constant(0, dtype = tf.float32), neg_d_score, delta_pos_d_score, emb_gen_y)
        )
        self.neg_d_score = self.neg_d_score.unstack()
        self.neg_d_score = tf.transpose(self.neg_d_score, perm = [1, 0])
        self.delta_neg_d_score = self.delta_neg_d_score.unstack()
        self.delta_neg_d_score = tf.transpose(self.delta_neg_d_score, perm = [1, 0])
        # self.neg_d_score = tf.reshape(tf.log(self.neg_d_score), [-1])     

        self.d_score_loss = (-tf.reduce_sum(tf.reshape(tf.log(self.pos_d_score), [-1])) + 
                              tf.reduce_sum(tf.reshape(tf.log(self.neg_d_score), [-1]))) / (self.y_len * self.batch_size)
        self.delta_d_score_loss = (tf.reduce_sum(tf.reshape(tf.square(self.delta_pos_d_score), [-1])) +
                                   tf.reduce_sum(tf.reshape(tf.square(self.delta_neg_d_score), [-1]))) / (self.y_len * self.batch_size)

        d_opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)        
        self.d_grad, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_params), self.grad_clip)
        self.d_updates = d_opt.apply_gradients(zip(self.d_grad, self.d_params))


    def discriminate(self, sess, x, y):
        outputs = sess.run([self.neg_d_score, self.delta_neg_d_score], feed_dict = {self.x: x, self.gen_y: y})
        return outputs

    def d_step(self, sess, x, y, gen_y):
        outputs = sess.run([self.d_updates, self.d_loss], feed_dict = {self.x: x, self.y: y, self.gen_y: gen_y})
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
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, 1]))
        self.bo = tf.Variable(self.init_matrix([1]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            return logits

        return unit