from __future__ import print_function
from __future__ import division

import importlib
from keras.layers.core import Activation
from keras.models import Graph
import numpy as np
import random
import traceback

import pysts.loader as loader
from pysts.kerasts import graph_input_slice, graph_input_prune
import pysts.kerasts.blocks as B

from tasks import AbstractTask

from collections import defaultdict
def default_config(model_config, task_config):
    # TODO: Move this to AbstractTask()?
    c = dict()
    c['embdim'] = 300
    c['embprune'] = 100
    c['embicase'] = False
    c['inp_e_dropout'] = 0
    c['inp_w_dropout'] = 0
    c['e_add_flags'] = True

    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 2
    c['Dinit'] = 'glorot_uniform'
    c['f_add_kw'] = False

    c['loss'] = 'mse'  # you really want to override this in each task's config()
    c['balance_class'] = False

    c['opt'] = 'adam'
    c['fix_layers'] = []  # mainly useful for transfer learning, or 'emb' to fix embeddings
    c['batch_size'] = 160
    c['nb_epoch'] = 16
    c['nb_runs'] = 1
    c['epoch_fract'] = 1

    c['prescoring'] = None
    c['prescoring_prune'] = None
    c['prescoring_input'] = None

    task_config(c)
    if c.get('task>model', False):  # task config has higher priority than model
        model_config(c)
        task_config(c)
    else:
        model_config(c)
    return c


class Mintent(AbstractTask):

    def config(self,c):
        return    
    def set_conf(self, c):
        self.c = c

        if 's0pad' in self.c:
            self.s0pad = self.c['s0pad']
            self.s1pad = self.c['s1pad']
        elif 'spad' in self.c:
            self.spad = self.c['spad']
            self.s0pad = self.c['spad']
            self.s1pad = self.c['spad']

    def load_vocab(self, vocabf):
        _, _, self.vocab = self.load_set(vocabf)
        return self.vocab
    
    def load_grt(self,fname):
        d = defaultdict(list)
        classes={}
        for line in open(fname):
            sentence,cls = line[:-1].split(": ")
            if cls not in classes:
               classes[cls]=len(classes.keys())
            s_split=sentence.split()
            d[cls].extend(s_split)
        
       
        self.grt = {
                     's0': [d[doc] for doc in d],
                     's1': [[] for doc in d ] ,
                     'classes': [classes[doc] for doc in d]
                      # labels : -- not needed for training
                   }
       
        self.d = d
    def load_grv(self,fname):
        s0=[]
        s1=[]
        labels=[]
        linenums=[]
        lnum=0    
        for line in open(fname):
            sentence,s_cls = line[:-1].split(": ")
            s_split=sentence.split()
            lnum+=1
            for cls,wlist in self.d.iteritems():
                s0.append(s_split)
                s1.append(wlist)
                linenums.append(lnum)
                if cls==s_cls:
                   labels.append(1)
                else:
                   labels.append(0)
                
        self.grv = {
              'linenums': linenums, 
              's0': s0,
              's1': s1,
              'labels': labels
              } 
        print(len(self.grv))
    
    def load_data(self, trainf, valf, testf=None):
        # create trainf valf...
        self.trainf = trainf
        self.valf = valf
        print(valf)
        self.testf = testf
        self.gr={'score':[]}
        self.load_grt(trainf)
        self.load_grv(valf)
        #if testf is not None:
        #    self.grt, self.yt, _ = self.load_set(testf)
        #else:
        #    self.grt, self.yt = (None, None)



    def build_model(self, module_prep_model, oact='sigmoid'):
        # Input embedding and encoding
        return module_prep_model([],self.c)  

    def fit_model(self, model, **kwargs):
        #kwargs['callbacks'] = self.fit_callbacks(kwargs.pop('weightsf'))
        return model.fit(self.grt)

    def eval(self, model):
        score = model.predict(self.grv)['score']
        res=defaultdict(list)
        for i in range(len(self.grv['s0'])):
            res[self.grv['linenums'][i]].append([self.grv['s0'][i],score[i],self.grv['labels'][i]])
          
        print(res)
        for s in res:
            data = res[s]
            data.sort(key= lambda x: x[1], reverse=True)
            if data[0][2]==1:
               print('ok')
            else: 
               print('bad')

def task():
    return Mintent()
 
