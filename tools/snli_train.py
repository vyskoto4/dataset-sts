#!/usr/bin/python3
"""
Train a KeraSTS model on the The Stanford Natural Language Inference (SNLI) Corpus task.

Usage: tools/snli_train.py MODEL VOCAB TRAINDATA VALDATA [PARAM=VALUE]...

Example: tools/ubuntu_train.py cnn data/anssel/ubuntu/v2-vocab.pickle data/anssel/ubuntu/v2-trainset.pickle data/anssel/ubuntu/v2-valset.pickle

See coments in anssel_train.py.

If this crashes due to out-of-memory error, you'll need to lower the batch
size - pass e.g. batch_size=128.  To speed up training, you may want to
conversely bump the batch_size if you have a smaller model (e.g. cnn).

First, you must however run:
    tools/ubuntu_preprocess.py --revocab data/anssel/ubuntu/v2-trainset.csv data/anssel/ubuntu/v2-trainset.pickle data/anssel/ubuntu/v2-vocab.pickle
    tools/ubuntu_preprocess.py data/anssel/ubuntu/v2-valset.csv data/anssel/ubuntu/v2-valset.pickle data/anssel/ubuntu/v2-vocab.pickle
    tools/ubuntu_preprocess.py data/anssel/ubuntu/v2-testset.csv data/anssel/ubuntu/v2-testset.pickle data/anssel/ubuntu/v2-vocab.pickle
(N.B. this will include only the first 1M samples of the train set).

(TODO: Make these downloadable.)

Notes:
    * differs from https://github.com/npow/ubottu/blob/master/src/merge_data.py
      in that all unseen words outside of train set share a single
      common random vector rather than each having a different one
      (or deferring to stock GloVe vector)
    * reduced vocabulary only to words that appear at least twice,
      because the embedding matrix is too big for our GPUs otherwise
    * in case of too long sequences, the beginning is discarded rather
      than the end; this is different from KeraSTS default as well as
      probably the prior art

Ideas (future work):
    * move the fit_generator-related functions to KeraSTS
    * rebuild the train set to train for a ranking task rather than
      just identification task?
"""

from __future__ import print_function
from __future__ import division

import importlib
import numpy as np
try:
    import cPickle
except ImportError:  # python3
    import pickle as cPickle
import pickle
import random
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.models import Graph
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2


import pysts.embedding as emb
import pysts.eval as ev
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.hyperparam import hash_params
from pysts.vocab import Vocabulary

from pysts.kerasts import graph_input_anssel, graph_input_slice
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import AnsSelCB
from pysts.kerasts.objectives import ranknet, ranksvm, cicerons_1504

import anssel_train
import models  # importlib python3 compatibility requirement


# XXX: We didn't verify this really covers all sentences.  It should cover
# a lot, though.
s0pad = 160
s1pad = 160






def load_set(fname, vocab):
    si0, si1, f0, f1, labels = cPickle.load(open(fname, "rb"))
    gr = graph_input_anssel(si0, si1, labels, f0, f1)
    return gr



def config(module_config, params):
    c = dict()
    c['embdim'] = 300
    c['inp_e_dropout'] = 3/4
    c['inp_w_dropout'] = 0
    c['e_add_flags'] = True

    c['ptscorer'] = B.dot_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2

    c['loss'] ='categorical_crossentropy'
    c['batch_size'] = 160
    c['nb_epoch'] = 32
    module_config(c)

    for p in params:
        k, v = p.split('=')
        c[k] = eval(v)

    ps, h = hash_params(c)
    return c, ps, h


def prep_model(glove, vocab, module_prep_model, c, oact, s0pad, s1pad):
    # Input embedding and encoding
    model = Graph()
    N = B.embedding(model, glove, vocab, s0pad, s1pad, c['inp_e_dropout'], c['inp_w_dropout'], add_flags=c['e_add_flags'])

    # Sentence-aggregate embeddings
    final_outputs = module_prep_model(model, N, s0pad, s1pad, c)

    # Measurement
    kwargs = dict()
    if c['ptscorer'] == B.mlp_ptscorer:
        kwargs['sum_mode'] = c['mlpsum']
    model.add_node(name='scoreS', input=c['ptscorer'](model, final_outputs, c['Ddim'], N, c['l2reg'], **kwargs),
                   layer=Activation('linear'))
    model.add_node(name='out', input='scoreS',
                   layer=Dense(3, W_regularizer=l2(c['l2reg'])))
    model.add_node(name='outS', input='out',
                   layer=Activation('softmax'))

    model.add_output(name='classes', input='outS')


def build_model(glove, vocab, module_prep_model, c, s0pad=s0pad, s1pad=s1pad, optimizer='adam'):
    if c['ptscorer'] is None:
        # non-neural model
        return module_prep_model(vocab, c)

    if c['loss'] == 'binary_crossentropy':
        oact = 'sigmoid'
    else:
        # ranking losses require wide output domain
        oact = 'linear'

    model = prep_model(glove, vocab, module_prep_model, c, oact, s0pad, s1pad)
    model.compile(loss={'score': c['loss']}, optimizer=optimizer)
    return model


def train_and_eval(runid, module_prep_model, c, glove, vocab, gr, s0, grt, s0t, s0pad=s0pad, s1pad=s1pad, do_eval=True):
    print('Model')
    model = build_model(glove, vocab, module_prep_model, c, s0pad=s0pad, s1pad=s1pad)

    print('Training')

    # XXX: samples_per_epoch is in brmson/keras fork, TODO fit_generator()?
    model.fit(gr, validation_data=grt,
              callbacks=[AnsSelCB(s0t, grt),
                         ModelCheckpoint('weights-'+runid+'-bestval.h5', save_best_only=True, monitor='mrr', mode='max'),
                         EarlyStopping(monitor='mrr', mode='max', patience=4)],
              batch_size=c['batch_size'], nb_epoch=c['nb_epoch'], samples_per_epoch=int(len(s0)*c['epoch_fract']))
    model.save_weights('weights-'+runid+'-final.h5', overwrite=True)
    if c['ptscorer'] is None:
        model.save_weights('weights-'+runid+'-bestval.h5', overwrite=True)
    model.load_weights('weights-'+runid+'-bestval.h5')

    if do_eval:
        print('Predict&Eval (best epoch)')
        ev.eval_anssel(model.predict(gr)['score'][:,0], s0, gr['score'], 'Train')
        ev.eval_anssel(model.predict(grt)['score'][:,0], s0t, grt['score'], 'Val')
    return model



if __name__ == "__main__":
    modelname, vocabf, trainf, valf = sys.argv[1:5]
    params = sys.argv[5:]

    module = importlib.import_module('.'+modelname, 'models')
    conf, ps, h = config(module.config, params)

    runid = '%s-%x' % (modelname, h)
    print('RunID: %s  (%s)' % (runid, ps))

    print('GloVe')
    glove = emb.GloVe(N=conf['embdim'])

    print('Dataset (vocab)')
    vocab = pickle.load(open(vocabf, "rb"))  # use plain pickle because unicode
    print('Dataset (train)')
    gr = load_set(trainf, vocab)
    print('Dataset (val)')
    grt = load_set(valf, vocab)
    print('Padding (val)')
    #pad_graph(grt)

    train_and_eval(runid, module.prep_model, conf, glove, vocab, gr, grt)


'''
module = importlib.import_module('.'+modelname, 'models')
conf={}
conf['embdim']=300
modelname='avg'

vocabf='snli_1.0_train.vocab'
dataft='../snli_1.0/snli_1.0_train.jsonl'
datafts='../snli_1.0/snli_1.0_test.jsonl'
vocab = Vocabulary(sentence_gen([dataft,datafts]), count_thres=2)
pickle.dump(vocab, open(vocabf, "wb"))

#s0i, s1i, labels = load_set(dataf, vocab)
glove = emb.GloVe(N=conf['embdim'])
'''