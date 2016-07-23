#
#
#
# ==============================================================================

import theano
import theano.tensor as T
import warnings
import numpy as np
import time
import sys
import os
import collections as cl
import lasagne

from sklearn.cross_validation import StratifiedShuffleSplit

#os.environ['OMP_NUM_THREADS']='4'
#theano.config.openmp = True
#theano.config.cxx = 'g++-5'


class MyNet(object):

    def __init__(self,
                 net_setup={'input_size':10, 'hidden_size':10, 'output_size':10},
                 loss_setup={'objective': 'squared_error'},
                 **kwargs):

        '''

        :param net_setup: whatever is needed for setup_network function
        :param loss_setup: whatever is needed for setting up losses, but should have at least 'objective' property
        :param kwargs:
        :return:
        '''

        # some symbolic variables
        self._X_symb = T.matrix("X")
        self._Y_symb = T.matrix("Y_bin")
        #self.class_weights_symb = T.vector('class_weights')
        self._learning_rate_symb = T.scalar("learning rate")

        # set up net architecture
        self.net_setup = net_setup
        self.out_layer = self.setup_network()
        self.output_size = self.out_layer.output_shape[1]

        # symbolic outputs
        self._output_symb = lasagne.layers.get_output(self.out_layer)
        self._test_output_symb = lasagne.layers.get_output(self.out_layer, deterministic=True)


        # setup losses
        self.loss_setup = loss_setup
        self.add_to_train_loss()
        self._train_loss_setup()
        self._test_loss_setup()

        # compile
        self._method = kwargs.get('method', None)
        if kwargs.get('compile', False) and self._method is not None:
            self.compile_functions(method=self._method, **kwargs)

        # for recording training progress
        self.train_losses = []
        self.cv_losses = []

        self.train_losses_batch = []
        self.cv_losses_batch = []

        self.train_batches_per_epoch = 1
        self.cv_batches_per_epoch = 1

        # if using some additional sciring function
        self.train_scores = []
        self.cv_scores = []

        # score_func
        self.score = kwargs.get('score_func', None)
        self.score_every = 1

        # initialize training epochs counter
        self.epoch = 0
        self.max_epoch_num = 0


    def setup_network(self,):
        '''
        subclass and override this for other architectures
        :return:
        '''
        l_in = lasagne.layers.InputLayer((None, self.net_setup['input_size']), self._X_symb)
        l_h = lasagne.layers.DenseLayer(l_in, self.net_setup['hidden_size'], nonlinearity=lasagne.nonlinearities.sigmoid)
        l_o = lasagne.layers.DenseLayer(l_h, self.net_setup['output_size'], nonlinearity=lasagne.nonlinearities.sigmoid)

        return l_o


    def _common_loss(self, prediction_symb, target_symb):
        return getattr(lasagne.objectives, self.loss_setup['objective'])(prediction_symb, target_symb)


    def add_to_train_loss(self):
        '''
        subclass and override this for other architectures
        '''
        self._add_train_loss = 0.


    def _train_loss_setup(self, ):
        # basic
        self._loss_symb = self._common_loss(self._output_symb, self._Y_symb)

        # maybe add regularization
        reg_params = self.loss_setup.get('regularization')
        if reg_params:
            regularization = lasagne.regularization.regularize_network_params(
                self.out_layer,
                getattr(lasagne.regularization, reg_params.get('type', 'l2'))
            ) * reg_params.get('C', 1.)

            self._loss_symb = self._loss_symb + regularization

        # class weights
        #self._loss_symb = self._loss_symb * self._class_weights_symb[:, None]

        # add something else if subclassing
        self._loss_symb = self._loss_symb + self._add_train_loss

        # final
        self._loss_symb = self._loss_symb.mean()


    def _test_loss_setup(self):

        self._test_loss_symb = self._common_loss(self._test_output_symb, self._Y_symb)
        self._test_loss_symb = self._test_loss_symb.mean()


    def compile_functions(self, method, **kwargs):

        self._method = method

        # get all params vars
        params_symb = lasagne.layers.get_all_params(self.out_layer)

        updates = getattr(lasagne.updates, self._method, 'sgd')(self._loss_symb, params_symb, self._learning_rate_symb, **kwargs)
        print("Using {} method with prams:".format(self._method,))
        for k, v in kwargs:
            print('\t', k, '=', v)


        self._train_func = theano.function(
            inputs=[
                self._X_symb,
                self._Y_symb,
                self._learning_rate_symb,
                #self.class_weights_symb
            ],
            outputs=self._loss_symb,
            updates=updates,
            allow_input_downcast=True,
        )

        self._cv_func = theano.function(
            inputs=[self._X_symb, self._Y_symb],
            outputs=self._test_loss_symb,
            allow_input_downcast=True,
        )

        self._predict_proba_func = theano.function(
            inputs=[self._X_symb],
            outputs=self._test_output_symb,
            allow_input_downcast=True,
        )


    @staticmethod
    def _iterate_minibatches(arr_length, batchsize):
        for start_idx in range(0, arr_length, batchsize):
            yield start_idx, np.min([start_idx + batchsize, arr_length])


    def add_score_function(self, score_func, score_every=1):
        '''
        score_func should be a function that takes 2 arrays, truth and predictions, and outputs a number
        :param score_func:
        :return:
        '''
        self.score = score_func
        self.score_every = score_every


    def _get_learning_rate_schedule(self, learning_rate):

        # need a learning rate value for each epoch
        lr_schedule = np.zeros(self.max_epoch_num)

        # if constant
        if isinstance(learning_rate, (float, int)):
            lr_schedule = learning_rate * np.ones(self.max_epoch_num).astype(float)

        # if in form of [[starting_epoch_num, lr_value],]
        elif isinstance(learning_rate, (dict, list, np.ndarray)):
            # convert to list of lists
            if isinstance(learning_rate, dict):
                learning_rate = learning_rate.items()

            # sort ascending of starting_epoch_num
            learning_rate = sorted(list(learning_rate), key=lambda x: x[0])

            # fill out
            for row in learning_rate:
                lr_schedule[row[0]:] = row[1]

        return lr_schedule


    def fit(self, X, Y,
            CV_data=False,
            learning_rate=0.1,
            batch_size=500,
            max_epoch_num=10,
            #class_weights=None,
            shuffle=False,
            patience=5,
            record_batches=False,
            progress_every=10,
            train_loss_stop=False,
            stop_file='',
            restart=False):

        if restart:
            self.epoch = 0

        self.max_epoch_num = max_epoch_num
        use_score = False
        if self.score is not None:
            use_score = True

        ######################## prepare data #######################
        # deal with crossvalidation data
        use_CV = True
        X_tr = X
        Y_tr = Y
        if type(CV_data) == float and (1 > CV_data > 0):
            sss = StratifiedShuffleSplit(Y, n_iter=1, test_size=CV_data, random_state=137)
            tr_i, te_i = list(sss)[0]
            X_tr, X_te = X[tr_i], X[te_i]
            Y_tr, Y_te = Y[tr_i], Y[te_i]
        elif isinstance(CV_data, (list, np.ndarray)) and len(CV_data) == 2:
            X_te, Y_te = CV_data
        else:
            print("No cross validation dataset")
            use_CV = False

        # deal with class weights
        # if not isinstance(class_weights, (np.ndarray, list)):   # if no weights supplied
        #     class_weights = np.ones(X_tr.shape[0])
        # elif class_weights.shape[0] == Y_tr.shape[1]: # if supplied weights per class, need to make a weight for each example
        #     class_weights = class_weights[Y_tr.argmax(axis=1)]
        # elif class_weights.shape[0] != X_tr.shape[0]:
        #     warnings.warn("Not using class weights because provided class weights length "
        #                   "is not equal to number of classes nor to the number of samples... If passing "
        #                   "float as CS_data, use class_weights as a weight per class array.")
        #     class_weights = np.ones(X_tr.shape[0])


        # deal with learning rate schedule
        lr_schedule = self._get_learning_rate_schedule(learning_rate)


        ######################## training #######################
        # start training
        t0 = time.time()
        stop = False
        while self.epoch < max_epoch_num and not stop:

            # main updates
            train_batches = 0
            train_batch_loss = 0
            for start_ix, end_ix in self._iterate_minibatches(X_tr.shape[0], batch_size):

                if record_batches:
                    self.train_losses_batch.append(self._train_func(X_tr[start_ix: end_ix],
                                                                    Y_tr[start_ix: end_ix],
                                                                    lr_schedule[self.epoch],
                                                                    #class_weights[start_ix: end_ix]
                                                                    ))
                    train_batch_loss += self.train_losses_batch[-1]

                else:
                    train_batch_loss += self._train_func(X_tr[start_ix: end_ix],
                                                         Y_tr[start_ix: end_ix],
                                                         lr_schedule[self.epoch],
                                                         #class_weights[start_ix: end_ix]
                                                         )

                train_batches += 1

            self.train_losses.append(train_batch_loss / train_batches)
            self.train_batches_per_epoch = train_batches

            # calcualte cross validation loss
            if use_CV:
                cv_batches = 0
                cv_batch_loss = 0
                for start_ix, end_ix in self._iterate_minibatches(X_te.shape[0], batch_size):

                    if record_batches:
                        self.cv_losses_batch.append(self._cv_func(X_te[start_ix: end_ix],
                                                                  Y_te[start_ix: end_ix]))
                        cv_batch_loss += self.cv_losses_batch[-1]

                    else:
                        cv_batch_loss += self._cv_func(X_te[start_ix: end_ix],
                                                       Y_te[start_ix: end_ix])

                    cv_batches += 1

                self.cv_losses.append(cv_batch_loss / cv_batches)
                self.cv_batches_per_epoch = cv_batches

            if use_score and self.epoch % self.score_every == 0:
                self.train_scores.append(self.score(Y_tr, self.predict_proba(X_tr)))
                if use_CV:
                    self.cv_scores.append(self.score(Y_te, self.predict_proba(X_te)))

            # optionally shuffle the whole train dataset
            if shuffle:
                sh_ix = np.arange(X_tr.shape[0])
                np.random.shuffle(sh_ix)
                X_tr, Y_tr = X_tr[sh_ix], Y_tr[sh_ix]
                #class_weights = class_weights[sh_ix]


            # additional stopping conditions
            stop = self._get_stop_conditions(patience, use_CV, stop_file, train_loss_stop)

            # print out progress
            self._print_progress(progress_every, use_CV, use_score, t0)

            # finally increment epoch number
            self.epoch += 1


    def _get_stop_conditions(self, patience, use_CV, stop_file, train_loss_stop):

        stop = False
        # stop if CV loss gets worse
        if use_CV and patience > 0 and self.epoch > patience:
            # should be higher than min by increase_threshold, which is 6 sigmas larger than last patience_num losses
            increase_threshold = 6 * np.std(self.cv_losses[-patience:])
            last_min = np.argmin(self.cv_losses)

            # stop if value is larger than 6 sigmas
            if last_min < self.epoch - patience / 2 and self.cv_losses[last_min] + increase_threshold < self.cv_losses[-1]:
                print("exceeded threshold", increase_threshold, last_min)
                stop = True

            # stop if didnt improve minimum for  patience number of epochs
            elif last_min < self.epoch - patience:
                print("exceeded patience", increase_threshold, last_min)
                stop = True

        # stop if nan
        if np.isnan(self.train_losses[-1]):
            stop = True

        # stop if there is word "stop" in a stop-file, usefull for stopping without interupting the whole program
        if len(stop_file) > 0 and os.path.exists(stop_file):
            with open(stop_file) as f_stop:
                if f_stop.read().startswith('stop'):
                    stop = True

        # stop at a certain value of train loss
        if train_loss_stop and np.mean(self.train_losses[-5:]) < train_loss_stop:
            stop = True

        return stop


    def _print_progress(self, progress_every, use_CV, use_score, t0):

        if self.epoch % progress_every == 0:
            print("done epoch {}, time from start: {:.3f}".format(self.epoch, time.time() - t0))
            print("\tCurrent train loss: {}".format(self.train_losses[-1]))
            if use_CV: print("\tCurrent CV loss: {}".format(self.cv_losses[-1]))
            if use_score: print("\tCurrent train score: {}".format(self.train_scores[-1]))
            if use_score and use_CV: print("\tCurrent CV score: {}".format(self.cv_scores[-1]))

            sys.stdout.flush()


    def predict_proba(self, X, batch_size=512):
        res = np.zeros((X.shape[0], self.output_size))
        for start_ix, end_ix in self._iterate_minibatches(X.shape[0], batch_size):
            res[start_ix: end_ix, :] = self._predict_proba_func(X[start_ix: end_ix])

        return res


    def predict(self, X, batch_size=512):
        return self.predict_proba(X, batch_size)


    def save_net(self, fname='MyNet_params.npy'):

        np.save(fname, lasagne.layers.get_all_param_values(self.out_layer))
        print("Saved all layers params to {}".format(fname))


    def load_net(self, fname):

        lasagne.layers.set_all_param_values(self.out_layer, np.load(fname))


class MyAE(MyNet):

    def setup_network(self,):

        # input layer
        l_in = lasagne.layers.InputLayer((None, self.net_setup['input_size']), self._X_symb, name='input layer')
        l_d = lasagne.layers.DropoutLayer(l_in, self.net_setup['dropout_p'])
        l_h1 = lasagne.layers.DenseLayer(l_d, self.net_setup['hidden_size_2'], nonlinearity=lasagne.nonlinearities.rectify)
        l_h3 = lasagne.layers.DenseLayer(l_h1, self.net_setup['hidden_size_1'], nonlinearity=lasagne.nonlinearities.rectify)
        l_enc = lasagne.layers.DenseLayer(l_h3, self.net_setup['output_size'], nonlinearity=lasagne.nonlinearities.linear,
                                          W=lasagne.init.GlorotNormal())
        l_h2 = lasagne.layers.DenseLayer(l_enc, self.net_setup['hidden_size_1'], nonlinearity=lasagne.nonlinearities.rectify, W=l_enc.W.T)
        l_h4 = lasagne.layers.DenseLayer(l_h2, self.net_setup['hidden_size_2'], nonlinearity=lasagne.nonlinearities.rectify, W=l_h3.W.T)
        l_o = lasagne.layers.DenseLayer(l_h4, self.net_setup['input_size'], nonlinearity=lasagne.nonlinearities.linear, W=l_h1.W.T)

        self.enc_layer = l_enc

        return l_o

    def compile_mid_output(self,):

        enc_symb = lasagne.layers.get_output(self.enc_layer, deterministic=True)

        self.encode = theano.function(
            inputs=[self._X_symb,],
            outputs=enc_symb,
            allow_input_downcast=True,
        )