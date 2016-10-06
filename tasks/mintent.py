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
            d[cls].append(sentence.split())
        
        sentences = sum(d.values(),[])
        if len(sentences)%2==1:
           sentences.append("")
        
        
       
        self.grt = {
                     's0': sentences[::2],
                     's1': sentences[1::2] ,
                      # labels : -- not needed for training
                   }
       
        self.d = d
    def load_grv(self,fname):
        s0=[]
        s1=[]
        labels=[]
        linenums=[] # linenum is a sentence identifier in eval
        lnum=0    
        for line in open(fname):
            sentence,s_cls = line[:-1].split(": ")
            s_split=sentence.split()
            lnum+=1
            for cls,slist in self.d.iteritems():
                for s in slist:
                    s0.append(s_split)
                    s1.append(s)
                    linenums.append(lnum)
                    labels.append(1 if cls==s_cls else 0)                
        self.grv = {
              'linenums': linenums, 
              's0': s0,
              's1': s1,
              'labels': labels
              } 
    
    def load_data(self, trainf, valf, testf=None):
        # create trainf valf...
        self.trainf = trainf
        self.valf = valf
        self.testf = testf
        self.gr={'score':[]}
        self.load_grt(trainf)
        self.load_grv(valf)



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
        train_sentences=len(res)
        cls_correct=0  
        for s in res:
            data = res[s]
            data.sort(key= lambda x: x[1], reverse=True)
            if data[0][2]==1:
               cls_correct+=1
        acc =cls_correct*1.0/train_sentences
        print("Accuracy %.3f (%d out of %d correct)"%(acc, cls_correct,train_sentences))
        return acc
def task():
    return Mintent()
 
