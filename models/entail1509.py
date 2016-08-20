"""
A model that is similar to the one from
Rocktaschel et al. "Reasoning about entailment with neural attention."
approaches 2015-state-of-art results on the anssel-wang task (with
token flags).

The implementation is inspired by https://github.com/shyamupa/snli-entailment/blob/master/amodel.py


"""


from keras.layers import GRU, Dropout, Lambda, Dense, RepeatVector, TimeDistributedDense, Activation, Reshape, Permute, Flatten
from keras.regularizers import l2
from keras import backend as K
import pysts.kerasts.blocks as B

def config(c):
    c['dropout'] = 4/5
    c['dropoutfix_inp'] = 0
    c['dropoutfix_rec'] = 0
    c['l2reg'] = 1e-4
    c['rnnact'] = 'tanh'
    c['rnninit'] = 'glorot_uniform'
    c['sdim'] = 2



def get_H_n(X):
    ans=X[:, -1, :]
    return ans


def generate_get_Y(size):
    def get_Y_generator(X):
        return get_Y(X,size)
    return get_Y_generator

def get_Y(X):
    return X[:, :60, :]

def get_H_0():
    return X[:, 0, :]


def get_R(X):
    Y, alpha = X.values()  # Y should be (None,L,k) and alpha should be (None,L,1) and ans should be (None, k,1)
    ans=K.T.batched_dot(Y,alpha)
    return ans

def rnn_input(model, N, spad, input,c):
        model.add_node(name='forward', input=input,
                          layer=GRU(input_dim=N, output_dim=N, input_length=2*spad,
                                    init=c['rnninit'], activation=c['rnnact'],
                                    return_sequences=True,
                                    dropout_W=c['dropoutfix_inp'], dropout_U=c['dropoutfix_rec']))

        model.add_node(name='backward', input=input,
                          layer=GRU(input_dim=N, output_dim=N, input_length=2*spad,
                                    init=c['rnninit'], activation=c['rnnact'],
                                    return_sequences=True, go_backwards=True,
                                    dropout_W=c['dropoutfix_inp'], dropout_U=c['dropoutfix_rec']))
        outputs=['e0s_', 'e1s_']
        model.add_node(name='rnndrop', inputs=['forward', 'backward'], merge_mode='concat' ,
                              layer=Dropout(c['dropout'], input_shape=(2*spad, int(N*c['sdim'])) ))
        return ['rnndrop']*2


def entailment_embedding(model, inputs,N=608, spad=60, l2reg=1e-4):
    model.add_node(Lambda(get_H_n, output_shape=(N,)), name='h_n', input=inputs[1])
    model.add_node(Lambda(get_Y, output_shape=(spad, N)), name='Y', input=inputs[0])
    model.add_node(Dense(N,W_regularizer=l2(l2reg)),name='Wh_n', input='h_n')
    model.add_node(RepeatVector(spad), name='Wh_n_cross_e', input='Wh_n')
    model.add_node(TimeDistributedDense(N,W_regularizer=l2(l2reg)), name='WY', input='Y')
    model.add_node(Activation('tanh'), name='M', inputs=['Wh_n_cross_e', 'WY'], merge_mode='sum')
    model.add_node(TimeDistributedDense(1,activation='softmax'), name='alpha', input='M')
    model.add_node(Permute((2,1)), name="Yp", input='Y')
    #model.add_node(name='_r',layer=B.dot_time_distributed_merge(model, ['Y','alpha'],
    #                                                      cos_norm=False)) 
    model.add_node(Lambda(get_R, output_shape=(N,1)), name='_r', inputs=['Yp','alpha'], merge_mode='join')
    model.add_node(Flatten(input_shape = (N,1)),name='r', input='_r')
    model.add_node(Dense(N,W_regularizer=l2(l2reg)), name='Wr', input='r')
    model.add_node(Dense(N,W_regularizer=l2(l2reg)), name='Wh', input='h_n')
    outputs=['Wr','Wh']
    return outputs


def prep_model(model, N, s0pad, s1pad, c):
    model.add_node(name="embmerge", inputs=['e0','e1'], merge_mode='concat', layer=Activation('linear'))
    rnn_outputs=rnn_input(model,N,s0pad,'embmerge',c)
    output = entailment_embedding(model, rnn_outputs,2*N,s0pad,c['l2reg'])
    return output
