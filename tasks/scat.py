from __future__ import print_function
from __future__ import division

from keras.models import Graph
import numpy as np
import random
import traceback
import cPickle
import pysts.embedding as emb


from keras.layers.core import Dropout, Activation, Dense
from keras.layers.recurrent import  GRU
from keras.layers.embeddings import Embedding

import pysts.loader as loader
from pysts.kerasts import graph_input_slice, graph_input_prune
import pysts.kerasts.blocks as B
from keras.callbacks import Callback
from keras.regularizers import l2


def multiclass_accuracy(y, ypred):
    """
    Compute accuracy for multiclass classification tasks
    Returns (rawacc, class_acc) where rawacc is the accuracy on the whole set
    and class_acc contains accuracies on all classes respectively
    """
    result = np.zeros(ypred.shape)
    clss=ypred.shape[1]
    ok=0
    for row in range(ypred.shape[0]):
        if np.argmax(ypred[row])  == y[row]:
            ok+=1
    rawacc = (ok*1.0)/y.shape[0]
    return rawacc


class SCAT_CB(Callback):
    """ A callback that monitors SCAT accuracy after each epoch """
    def __init__(self, task):
        self.task= task


    def on_epoch_end(self, epoch, logs={}):
        ypred=[]
        for ogr in self.task.sample_pairs(self.task.grv, batch_size=len(self.task.grv['score']), shuffle=False, once=True):
            ypred += list(self.model.predict(ogr)['score'])
        ypred = np.array(ypred)
        acc =multiclass_accuracy(self.task.grv['score'], ypred)
        print('                                                       val acc %f' % (acc,))
        logs['acc'] = acc

def default_config(model_config, task_config):
    # TODO: Move this to AbstractTask()?
    c = dict()
    c['spad']=20
    c['embdim'] = 300
    c['embprune'] = 100
    c['embicase'] = False
    c['inp_e_dropout'] = 1/2
    c['inp_w_dropout'] = 0


    c['loss'] = 'categorical_crossentropy'  # you really want to override this in each task's config()
    c['balance_class'] = False

    c['opt'] = 'adam'
    c['batch_size'] = 160
    c['nb_epoch'] = 16
    c['nb_runs'] = 1
    c['epoch_fract'] = 1/4

    task_config(c)
    if c.get('task>model', False):  # task config has higher priority than model
        model_config(c)
        task_config(c)
    else:
        model_config(c)
    return c


class SCAT(object):
    def set_conf(self, c):
        c['spad']=30
        self.c = c


    def load_set(self,fname):
        si, sj, labels = cPickle.load(open(fname, "rb"))
        gr= \
            { 'si': si,
              'sj': sj,
              labels : labels
            }
        self.num_classes=len(set(labels))
        return (gr, labels)

    def load_vocab(self, fname):
        vocabf = cPickle.load(open(fname, "rb"))
        return vocabf

    def load_data(self, trainf, valf, vocabf, testf=None):
        self.trainf = trainf
        self.valf = valf
        self.testf = testf

        self.gr, self.y = self.load_set(trainf)
        self.grv, self.yv= self.load_set(valf)
        self.vocab = self.load_vocab(vocabf)
        if testf is not None:
            self.grt, self.yt = self.load_set(testf)
        else:
            self.grt, self.yt = (None, None)


    def sample_pairs(self, gr, batch_size, shuffle=True, once=False):
        """ A generator that produces random pairs from the dataset """
        try:
            id_N = int((len(gr['si0']) + batch_size-1) / batch_size)
            ids = list(range(id_N))
            while True:
                if shuffle:
                    # XXX: We never swap samples between batches, does it matter?
                    random.shuffle(ids)
                for i in ids:
                    sl = slice(i * batch_size, (i+1) * batch_size)
                    ogr = graph_input_slice(gr, sl)
                    ogr['se0'] = self.emb.map_jset(ogr['sj0'])
                    # print(sl)
                    # print('<<0>>', ogr['sj0'], ogr['se0'])
                    # print('<<1>>', ogr['sj1'], ogr['se1'])
                    yield ogr
                if once:
                    break
        except Exception:
            traceback.print_exc()


    def prep_model(self):
        def rnn(model,c, N):
            c['dropout'] = 1/5
            c['dropoutfix_inp'] = 0
            c['dropoutfix_rec'] = 0
            c['l2reg'] = 1e-4

            c['rnnact'] = 'tanh'
            c['rnninit'] = 'glorot_uniform'

            # model-external:
            c['inp_e_dropout'] = 1/5
            c['inp_w_dropout'] = 0
            # anssel-specific:
            c['mlpsum'] = 'sum'
            model.add_node(name='forward', input=input,
                                  layer=GRU(input_dim=N, output_dim=N, input_length=c,
                                            init=c['rnninit'], activation=c['rnnact'],
                                            return_sequences=True,
                                            dropout_W=c['dropoutfix_inp'], dropout_U=c['dropoutfix_rec']))


        model = Graph()
        w2v = emb.Word2Vec(N=self.c['embdim'], w2vpath=self.c['w2vPath'])
        embmatrix = self.vocab.embmatrix(w2v)

        model.add_input('si0', input_shape=(self.c['spad'],), dtype='int')
        model.add_input('se0', input_shape=(self.c['spad'], w2v.N))
        model.add_node(name='emb', input='si0',
                              layer=Embedding(input_dim=embmatrix.shape[0], input_length=self.c['spad'],
                                              output_dim=w2v.N, mask_zero=True,
                                              weights=[embmatrix], trainable=True,
                                              dropout=self.c['input_w_dropout']))

        model.add_node(name='e0', inputs=['emb', 'se0'], merge_mode='sum', layer=Activation('linear'))
        model.add_node(name='embdrop', inputs='e0',
                              layer=Dropout(self.c['input_e_dropout'], input_shape=(w2v.N,)))


        final_outputs = rnn(model, self.c, w2v.N)
        model.add_node(name="out", layer=Dense(output_dim=self.num_classes, W_regularizer=l2(self.c['l2reg'], activation='tanh' )))
        model.add_output(name ="score", input='out')

        return model

    def fit_callbacks(self):
        return [SCAT_CB(self)]

    def fit_model(self, model, **kwargs):
        if self.c['ptscorer'] is None:
            return model.fit(self.gr, **kwargs)
        batch_size = kwargs.pop('batch_size')
        kwargs['callbacks'] = self.fit_callbacks()
        return model.fit_generator(self.sample_pairs(self.gr, batch_size), **kwargs)

    def predict(self, model, gr):
        batch_size = self.c['batch_size']
        ypred = []
        for ogr in self.sample_pairs(gr, batch_size, shuffle=False, once=True):
            ypred += list(model.predict(ogr)['score'][:,0])
        return np.array(ypred)
