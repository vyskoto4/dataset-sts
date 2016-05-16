#!/usr/bin/python3
"""
KeraSTS interface for the SNLI dataset of the Textual Entailment task.

Training example:
	tools/train.py avg snli  data/snli/snli_1.0_train.pickle data/snli/snli_1.0_dev.pickle vocabf="data/snli/v1-vocab.pickle" testf="data/snli/snli_1.0_test.pickle" inp_w_dropout=0.5


Before training, you must however run:
   tools/snli_preprocess.py data/snli/snli_1.0_train.jsonl data/snli/snli_1.0_dev.jsonl data/snli/snli_1.0_test.jsonl data/snli/snli_1.0_train.pickle  data/snli/snli_1.0_dev.pickle data/snli/snli_1.0_test.pickle data/snli/v1-vocab.pickle

current experimental model implementation based on https://github.com/shyamupa/snli-entailment/blob/master/amodel.py

"""

from __future__ import print_function
from __future__ import division


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GRU
from keras.layers.core import Activation, Dropout, Lambda, Dense, RepeatVector, TimeDistributedDense, Reshape
from keras.models import Graph
from keras import backend as K

import pickle
import pysts.eval as ev
import numpy as np

from keras.regularizers import l2
from pysts.kerasts import graph_input_anssel
import pysts.kerasts.blocks as B

from .anssel import AbstractTask


# 60 like spad
xmaxlen=60;

def get_Y(X):
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim

def get_R(X):
    Y, alpha = X.values()  # Y should be (None,L,k) and alpha should be (None,L,1) and ans should be (None, k,1)
    tmp=K.permute_dimensions(Y,(0,)+(2,1))  # copied from permute layer, Now Y is (None,k,L) and alpha is always (None,L,1)
    ans=K.T.batched_dot(tmp,alpha)
    return ans

def get_H_n(X):
    ans=X[:, -1, :]  # get last element from time dim
    return ans




class SnliTask(AbstractTask):
    def __init__(self):
        self.name = 'snli'
        self.spad=60
        self.s0pad = self.spad
        self.s1pad= self.spad
        self.emb = None
        self.vocab = None

    def load_vocab(self, vocabf):
        # use plain pickle because unicode
        self.vocab = pickle.load(open(vocabf, "rb"))
        return self.vocab


    def load_set(self,fname):
        si0, si1, f0, f1, y = pickle.load(open(fname,"rb"))
        gr = graph_input_anssel(si0, si1, y, f0, f1)
        return ( gr,y,self.vocab)

    def config(self, c):
        c['loss'] = 'categorical_crossentropy'
        c['nb_epoch'] = 32
        c['batch_size'] = 200
        c['epoch_fract'] = 1/4

    def build_model(self, module_prep_model, do_compile=True):
        if self.c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, self.c, output='binary')

        model = self.prep_model(module_prep_model)

        for lname in self.c['fix_layers']:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss={'score': self.c['loss']}, optimizer=self.c['opt'])
        return model

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf),(self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = model.predict(gr)['score']
            res.append(ev.eval_snli(ypred, gr['score'], fname))
        return tuple(res)


    def prep_model(self, module_prep_model):
        # Input embedding and encoding
        model = Graph()
        sdim=2
        N = B.embedding(model, self.emb, self.vocab, self.s0pad, self.s1pad,
                        self.c['inp_e_dropout'], self.c['inp_w_dropout'], add_flags=self.c['e_add_flags'])

        emb_outs=['e0_', 'e1_']
        model.add_node(name='backward', inputs=[emb_outs], merge_mode='concat', layer=Activation('linear'))
        model.add_node(name='rnn0', inputs='emb_merge', outputs=['e0s'],
                              layer=GRU(input_dim=N, input_length=2*self.spad,
                                        init='glorot_uniform', activation='tanh',
                                        return_sequences=True,go_backwards=True))
        model.add_node(name='forward', inputs='emb_merge', outputs=['e0s'],
                              layer=GRU(input_dim=N, input_length=2*self.spad,
                                        init='glorot_uniform', activation='tanh',
                                        return_sequences=True))
        #model.add_node(GRU(opts.lstm_units, return_sequences=True), name='forward', input='emb_merge')
        #model.add_node(GRU(opts.lstm_units, return_sequences=True, go_backwards=True), name='backward',  input='emb_merge')


        # FIXME! the code is dirty...
        model.add_node(Dropout(0.1), name='dropout', inputs=['forward','backward'])


        k=N # glove dims
        L=self.spad # L = number of words of premise
        model.add_node(Lambda(get_H_n, output_shape=(k,)), name='h_n', input='dropout')


        model.add_node(Lambda(get_Y, output_shape=(L, k)), name='Y', input='dropout')
        # model.add_node(SliceAtLength((None,N,k),L), name='Y', input='dropout')
        model.add_node(Dense(k,W_regularizer=l2(0.01)),name='Wh_n', input='h_n')
        model.add_node(RepeatVector(L), name='Wh_n_cross_e', input='Wh_n')
        model.add_node(TimeDistributedDense(k,W_regularizer=l2(0.01)), name='WY', input='Y')
        model.add_node(Activation('tanh'), name='M', inputs=['Wh_n_cross_e', 'WY'], merge_mode='sum')
        model.add_node(TimeDistributedDense(1,activation='softmax'), name='alpha', input='M')
        model.add_node(Lambda(get_R, output_shape=(k,1)), name='_r', inputs=['Y','alpha'], merge_mode='join')
        model.add_node(Reshape((k,)),name='r', input='_r')
        model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wr', input='r')
        model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wh', input='h_n')
        model.add_node(Activation('tanh'), name='h_star', inputs=['Wr', 'Wh'], merge_mode='sum')

        model.add_node(Dense(3, activation='softmax'), name='out', input='h_star')
        model.add_output(name='score', input='out')

        return model


    def build_model(self, module_prep_model, do_compile=True):
        if self.c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, self.c, output='binary', spad=self.spad)
        model = self.prep_model(module_prep_model)

        if do_compile:
            model.compile(loss={'score': self.c['loss']}, optimizer=self.c['opt'])
        return model

    def fit_callbacks(self, weightsf):
        return [ModelCheckpoint(weightsf, save_best_only=True),
                EarlyStopping(patience=3)]


    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[self.trainf]['Accuracy'],
                  pfx, mres[self.valf]['Accuracy'],
                  pfx, mres[self.testf].get('Accuracy', np.nan)))

def task():
    return SnliTask()
