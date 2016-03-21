import json
import pickle
import sys
from pysts.vocab import Vocabulary
from nltk.tokenize import word_tokenize


def sentence_gen(dsfiles):
    """ yield sentences from data files (train, validation) """
    i = 0
    for fname in dsfiles:
        with open(fname) as f:
            for l in f:
                d=json.loads(l)
                yield word_tokenize(d['sentence1'])
                yield word_tokenize(d['sentence2'])
                i += 1

if __name__ == "__main__":
    testset, valset, vocabf = sys.argv[1:3]
    vocab = Vocabulary(sentence_gen([testset,valset]), count_thres=2)
    pickle.dump(vocab, open(vocabf, "wb"))
