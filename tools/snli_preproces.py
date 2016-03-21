import json
import pickle
import sys
from pysts.vocab import Vocabulary


def sentence_gen(dsfiles):
    """ yield sentences from data files (train, validation) """
    i = 0
    for fname in dsfiles:
        with open(fname) as f:
            for l in f:
                d=json.loads(l)
                yield d['sentence1'].split(' ')
                yield d['sentence2'].split(' ')
                i += 1

if __name__ == "__main__":
    print sys.argv[1:]
    testset, valset, vocabf = sys.argv[1:]
    vocab = Vocabulary(sentence_gen([testset,valset]), count_thres=2)
    pickle.dump(vocab, open(vocabf, "wb"))
