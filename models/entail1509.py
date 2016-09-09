"""
A model that is similar to the one from
Rocktaschel et al. "Reasoning about entailment with neural attention."
approaches 2015-state-of-art results on the anssel-wang task (with
token flags).

The implementation is inspired by https://github.com/shyamupa/snli-entailment/blob/master/amodel.py


"""

from keras.layers.core import Layer
from keras.layers import GRU, Dropout, Lambda, Dense, RepeatVector, TimeDistributedDense, Activation, Reshape, Permute, Flatten
from keras.regularizers import l2
from keras import backend as K
import pysts.kerasts.blocks as B

spad=60

def config(c):
    c['dropout'] = 4/5
    c['dropoutfix_inp'] = 0
    c['dropoutfix_rec'] = 0
    c['l2reg'] = 1e-4
    c['rnnact'] = 'tanh'
    c['rnninit'] = 'glorot_uniform'
    c['sdim'] = 2
    c['ptscorer']=B.to_n_simple_ptscorer




def get_H_n(X):
    ans=X[:, -1, :]
    return ans


class SplitSequence(Layer):
    def __init__(self, split_ind , **kwargs):
        self.split_ind=split_ind
        super(SplitSequence, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, :self.split_ind, :]

    def get_output_shape_for(self, input_shape):
        shape = input_shape
        input_shape[1]=self.split_ind
        return input_shape


def get_R(X):
    Y, alpha = X.values()  
    ans=K.T.batched_dot(Y,alpha)
    return ans

def rnn_input(model, N, spad, input,c):
        model.add_node(name='forward', input=input,
                          layer=GRU(input_dim=N, output_dim=N, input_length=spad,
                                    init=c['rnninit'], activation=c['rnnact'],
                                    return_sequences=True,
                                    dropout_W=c['dropoutfix_inp'], dropout_U=c['dropoutfix_rec']))

        model.add_node(name='backward', input=input,
                          layer=GRU(input_dim=N, output_dim=N, input_length=spad,
                                    init=c['rnninit'], activation=c['rnnact'],
                                    return_sequences=True, go_backwards=True,
                                    dropout_W=c['dropoutfix_inp'], dropout_U=c['dropoutfix_rec']))
        outputs=['e0s_', 'e1s_']
        model.add_node(name='rnndrop', inputs=['forward', 'backward'], merge_mode='concat', concat_axis=1,
                              layer=Dropout(c['dropout'], input_shape=( spad, N) ))
        return ['rnndrop']*2

def get_Y(X):
    spad = K.spad
    return X[:, :spad, :]  

def entailment_embedding(model, inputs,N=608, spad=60, l2reg=1e-4):
    setattr(K, 'spad',spad)
    model.add_node(Lambda(get_H_n, output_shape=(N,)), name='h_n', input=inputs[1])
    model.add_node(Lambda(get_Y, output_shape=(N,spad)), name='Y', input=inputs[0])
    #model.add_node(SplitSequence(spad), name='Yp', input=inputs[0])
    model.add_node(Permute((2,1)), name="Yp", input='Y')
    model.add_node(Dense(N,W_regularizer=l2(l2reg)),name='Wh_n', input='h_n')
    model.add_node(RepeatVector(spad), name='Wh_n_cross_e', input='Wh_n')
    model.add_node(TimeDistributedDense(N,W_regularizer=l2(l2reg)), name='WY', input='Yp')
    model.summary()
    model.add_node(Activation('tanh'), name='M', inputs=['Wh_n_cross_e', 'WY'], merge_mode='sum')
    model.add_node(TimeDistributedDense(1,activation='linear'), name='alpha', input='M')
    model.add_node(Lambda(get_R, output_shape=(N,1)), name='_r', inputs=['Yp','alpha'], merge_mode='join')
    model.add_node(Permute((2,1)), name="_rp", input='_r')
    model.add_node(Flatten(input_shape = (N,1)),name='r', input='_rp')
    model.add_node(Dense(N,W_regularizer=l2(l2reg)), name='Wr', input='r')
    model.add_node(Dense(N,W_regularizer=l2(l2reg)), name='Wh', input='h_n')
    outputs=['Wr','Wh']
    return outputs


def prep_model(model, N, s0pad, s1pad, c):
    model.add_node(name="embmerge", inputs=['e0','e1'], merge_mode='concat', layer=Activation('linear'))
    B.rnn_input(model, N, s0pad, dropout=c['dropout'], dropoutfix_inp=c['dropoutfix_inp'], dropoutfix_rec=c['dropoutfix_rec'],
               sdim=2, rnnbidi=True, return_sequences=True,
               rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', rnnbidi_mode='sum',
               rnnlevels=1,
               inputs=['embmerge'])
    rnn_outputs=['e0s_', 'e1s_']
    #rnn_outputs=rnn_input(model,2*N,s0pad,'embmerge',c)
    #rnn_outputs=['embmerge','embmerge']
    outputs = entailment_embedding(model, rnn_outputs,2*N,s0pad,c['l2reg'])
    final_output = B.to_n_ptscorer(model, outputs, c['Ddim'], N, c['l2reg'], pfx="entail_out", output_dim=3)
    return final_output
