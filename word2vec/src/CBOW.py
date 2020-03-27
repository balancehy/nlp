#!usr/bin/python3
"""
Implement Continuous Bag of Words model. 
"""

import tensorflow as tf
import numpy as np
# from sklearn.utils import shuffle


class CBOW(object):
    def __init__(self, graph_params):
        """
        Take dictionary of parameters and set graph.
        graph_params: Expected dictionary entries:
            'batch_size': integer batch size
            'vocab_size': integer number of words in dictionary. Also the NN input dimension.
            'embed_size': integer size of word embedding
            'seq_len': number of words used as context
            'hid_size': integer size of hidden layer
            'neg_samples': number of samples for noise contrastive estimation. 64 is a reasonable.
            'learning_rate': optimizer learning rate
        """
        self.graph = tf.Graph()
        self.params = graph_params.copy()
        self.tensors = dict()
        self.results = dict()
        self.summary = None
        self._set_default_params()
        self._build_graph()

    def _set_default_params(self):
        """Set default value for undefined parameters"""
        self.params['neg_samples'] = self.params.get('neg_samples', 64)
        self.params['embed_noise'] = self.params.get('embed_noise', 0.1)
        self.params['hid_noise'] = self.params.get('hid_noise', 0.1)
        self.params['model_name'] = self.params.get('model_name', '../model-save/model_save') # model saving path
        self.params['trunc_norm'] = self.params.get('trunc_norm', False)
        self.params["drop_rate"] = self.params.get("drop_rate", 0.0)

        return None

    def _build_graph(self):

        with self.graph.as_default():
            tensors = self.tensors  # shorten var names
            params = self.params 
            # Define model input and output
            tensors['x'] = tf.placeholder(
                tf.int32, shape=[None, params["seq_len"]], name="Input")
            tensors['target'] = tf.placeholder(
                tf.int32, shape=[None, 1], name="Target")
            tensors["drop_rate"] = tf.placeholder(tf.float32, shape=[])

            # embedding layer
            with tf.name_scope("embed_layer"):
                tensors['embed_weights'] = tf.Variable(
                    tf.random_uniform([params['vocab_size'],
                                    params['embed_size']],
                                    -params['embed_noise'],
                                    params['embed_noise']),
                                    name="embed_weights")
            # Flatten embeddings for all input words
                embed_stack = tf.nn.embedding_lookup(tensors['embed_weights'], tensors["x"])
                embed_stack = tf.reshape(embed_stack, shape=[-1, params['embed_size']*params["seq_len"]],
                name="embed_stack")

            # hidden layer
            with tf.name_scope("hidden_layer"):
                hid_weights = tf.Variable(tf.random_normal([params['embed_size'] * params["seq_len"],
                                    params['hid_size']],
                                    stddev=params['hid_noise']/(params['embed_size'] * params["seq_len"])**0.5),
                                    name="hid_weights")
                hid_bias = tf.Variable(tf.zeros([params['hid_size']]), name="hid_bias")
                hid_out = tf.nn.leaky_relu(tf.matmul(embed_stack, hid_weights) + hid_bias, name="hid_out", alpha=0.2)
                # hid_out = tf.nn.tanh(tf.matmul(embed_stack, hid_weights) + hid_bias, name="hid_out")
                hid_out = tf.nn.dropout(hid_out, rate=tensors["drop_rate"])

            # output layer
            with tf.name_scope("output_layer"):
                if params['trunc_norm']:
                    tensors['nce_weights'] = tf.Variable(
                        tf.truncated_normal([params['vocab_size'],
                                            params['hid_size']],
                                            stddev=1.0 / params['hid_size'] ** 0.5), 
                                            name="nce_weights")
                else:
                    tensors['nce_weights'] = tf.Variable(
                        tf.random_normal([params['vocab_size'],
                                        params['hid_size']],
                                        stddev=1.0 / params['hid_size'] ** 0.5),
                                        name="nce_weights")

                tensors['nce_bias'] = tf.Variable(tf.zeros([params['vocab_size']]), name="nce_bias")

            # loss function - noise contrastive estimation
            tensors['loss'] = tf.reduce_mean(
                tf.nn.nce_loss(tensors['nce_weights'], tensors['nce_bias'],
                               inputs=hid_out, labels=tensors['target'],
                               num_sampled=params['neg_samples'],
                               num_classes=params['vocab_size'],
                               num_true=1, remove_accidental_hits=True))

            # predictor. Keep logits.
            tensors['y_logits'] = tf.matmul(hid_out, tensors['nce_weights'], transpose_b=True) + tensors['nce_bias']
            tensors['y_pred'] = tf.argmax(tensors['y_logits'], axis=1)
            # optimizer
            if params['optimizer'] == 'RMSProp':
                tensors['optimizer'] = tf.train.RMSPropOptimizer(params['learning_rate'])
            elif params['optimizer'] == 'Momentum':
                tensors['optimizer'] = tf.train.MomentumOptimizer(params['learning_rate'], params['momentum'])
            elif params['optimizer'] == 'Adam':
                tensors['optimizer'] = tf.train.AdamOptimizer(params['learning_rate'])
            else:
                print('*** Optimizer not specified.')
            # Two steps minimize loss. Gradients are added to summary.
            grad_and_var = tensors['optimizer'].compute_gradients(tensors['loss'])
            tensors['optimizer'] = tensors['optimizer'].apply_gradients(grad_and_var)

            # variable init
            self.tensors['initializer'] = tf.global_variables_initializer()
            # saver
            self.tensors['saver'] = tf.train.Saver()
            # tensorboard log
            grad_norm = 0
            for grad, _ in grad_and_var:
                if isinstance(grad, tf.IndexedSlices): # get embedding_lookup gradient tensor
                    grad_value = grad.values
                else:   # get regular gradient tensor
                    grad_value = grad
                size = 1    # total element of tensor for normalization
                for s in grad_value.shape: 
                    if not isinstance(s, int):
                        size *= params["batch_size"]
                    else:
                        size *= s
                temp = tf.norm(grad_value, ord=2) / size**0.5
                tf.summary.scalar(name=grad_value.name, tensor=temp)
                grad_norm += temp

            tf.summary.scalar(name="grad_mean_norm", tensor=grad_norm)
            self.summary = tf.summary.merge_all()

        return None

    def eval_loss(self, x_val, y_val, session):
        """
        Calculate and return evaluation loss on a test set.
        :return: noise contrastive estimation loss
        """
        tensors = self.tensors  # shorten var names
        params = self.params            # shorten var names
        num_samples = x_val.shape[0]
        num_batches = num_samples // params['batch_size']
        avg_loss = 0
        avg_acc = 0
        for batch in range(num_batches):
            start_idx = batch * params['batch_size']
            end_idx = start_idx + params['batch_size']
            feed_dict = {tensors['x']: x_val[start_idx:end_idx, :],
                         tensors['target']: y_val[start_idx:end_idx, :],
                         tensors["drop_rate"]: 0.0}
            loss, y_pred = session.run([tensors['loss'], tensors['y_pred']], feed_dict=feed_dict)
            avg_loss += loss
            avg_acc += (len(np.where(y_pred.reshape(-1, 1) == y_val)[0]) / y_val.shape[0])

        return avg_loss/num_batches, avg_acc/num_batches

    def train(self, x, y, x_val=None, y_val=None, epochs=1, verbose=True, write_summary=False, pretrain_model=None):
        """
        Train model.
        verbose: if True, print update on training and validation error after every training epoch
        write_summary: if True, add summary object for tensorboard
        pretrain_model: a model suffix added to model name representing pretrained model
        return: training results
            'embed_weights' : np.array of shape (vocab_size, embed_size), input embedding
            'nce_weights' : np.array of shape (vocab_size, hid_size), embedding at output layer
            'nce_bias': np.array of shape (vocab_size, 1), bias at output layer
            'e_train' : training loss at end of each epoch
            'e_val' : validation loss at end of each epoch
        """
        tensors = self.tensors  # shorten var names
        params = self.params            # shorten var names
        num_batches = len(x) // params['batch_size']
        avg_loss, avg_loss_count, batch_count = (0, 0, 0)
        e_train, e_val = ([], [])
        min_val_loss = 1e10 # when val loss is less than this value, save model
        n_save = 0 # number of training model saved
        with tf.Session(graph=self.graph) as session:
            if pretrain_model is None:
                session.run(tensors['initializer'])   # initialize graph
            else:   # load pretrained model
                tensors['saver'].restore(session, params['model_name'] + '-' + pretrain_model)
            writer = tf.summary.FileWriter('../logdir', self.graph)
            for epoch in range(1, epochs+1):
                if verbose:
                    print('Epoch {}: '.format(epoch), end='')
                avg_loss, avg_loss_count = (0, 0)
                randidx = np.arange(len(x))
                np.random.shuffle(randidx) # shuffle index
                for batch in range(num_batches):
                    start_idx = batch * params['batch_size']
                    end_idx = start_idx + params['batch_size']
                    feed_dict = {tensors['x']: x[randidx[start_idx:end_idx], :],
                                 tensors['target']: y[randidx[start_idx:end_idx], :],
                                 tensors["drop_rate"]: params["drop_rate"]}
                    _, loss = session.run([tensors['optimizer'],
                                           tensors['loss']],
                                          feed_dict=feed_dict)
                    avg_loss += loss
                    avg_loss_count += 1
                    batch_count += 1
                    if write_summary:
                        writer.add_summary(session.run(self.summary, feed_dict=feed_dict), batch_count)

                val_loss, val_acc = self.eval_loss(x_val, y_val, session)
                e_val.append(val_loss)
                e_train.append(avg_loss / avg_loss_count)
                if verbose:
                    print('total batch = {}, loss_tr = {:.2f}, loss_va = {:.2f}, acc_va = {:.2f}'
                          .format(batch_count, avg_loss / avg_loss_count, val_loss, val_acc), end="")
                # Save training model, keep 5 most recent models
                tensors['saver'].save(session, params['model_name']+'-'+str(n_save%5 + 1))
                n_save += 1
                # Save testing model when testing loss decreases
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    print("\tSave better")
                    tensors['saver'].save(session, params['model_name']+'-bestVali')
                else:
                    print('\n')

            self.results['embed_weights'] = tensors['embed_weights'].eval()
            self.results['nce_weights'] = tensors['nce_weights'].eval()
            self.results['nce_bias'] = tensors['nce_bias'].eval()
            self.results['e_train'] = e_train
            self.results['e_val'] = e_val
            self.results['epochs'] = epochs
            val_loss, val_acc = self.eval_loss(x_val, y_val, session)
            print('End Training: total batch = {}, loss_tr = {:.2f}, loss_va = {:.2f}, acc_va = {:.2f}'
                  .format(batch_count, avg_loss / avg_loss_count, val_loss, val_acc))

        return self.results.copy()

    def predict(self, x, mode_suffix=None):
        """
        Predict target value in vocab dimension from all input words
        """
        if mode_suffix is None:
            mode_suffix = str(1)
        tensors = self.tensors  # shorten var names
        params = self.params            # shorten var names
        with tf.Session(graph=self.graph) as session:
            tensors['saver'].restore(session, params['model_name'] + '-' + mode_suffix)
            feed_dict = {tensors['x']: x,
                        tensors["drop_rate"]: 0.0}
            y_pred = session.run(tensors['y_pred'], feed_dict=feed_dict)
            
        return y_pred

    @staticmethod
    def build_training_set(word_array, seq_len):
        """
        Build data set for CBOW model. Take seq_len number of context words as inputs, the middle word as target.
        word_array: Array of integers representing words in a document.
        seq_len: total number of context words, should be distributed evenly on both sides.
        return: 2-tuple (features, target):
            1. np.array(dtype=np.int32, shape=(bs, seq_len): context words 
            2. np.array(dtype=np.int32, shape=(bs, 1): middle words
        """
        k = seq_len
        context_window = np.concatenate([np.arange(-k//2, 0), np.arange(1, k//2+1)])
        x = []
        y = []
        if isinstance(word_array[0], np.int32):
            num_words = len(word_array)
            # x = np.zeros((num_words-k, k), dtype=np.int32)
            # y = np.zeros((num_words-k, 1), dtype=np.int32)
            for idx in range(k//2, num_words-k//2):
                y.append(word_array[idx:idx+1])
                x.append(word_array[idx+context_window])
        elif isinstance(word_array[0], np.ndarray):
            for sent in word_array:
                num_words = len(sent)
                if num_words >= k + 1:
                    for idx in range(k//2, num_words-k//2):
                        y.append(sent[idx:idx+1])
                        x.append(sent[idx + context_window])
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)

        return x, y