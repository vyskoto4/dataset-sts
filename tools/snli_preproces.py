import json
import pickle
import sys
from pysts.vocab import Vocabulary


def sentence_gen(dsfiles):
    """ yield sentences from data file """
    i = 0
    for fname in dsfiles:
        with open(fname) as f:
            for l in f:
                d=json.loads(l)
                yield d['sentence1'].split(' ')
                yield d['sentence2'].split(' ')
                i += 1

def load_set(dsfile, vocab):
    s0i = []
    s1i = []
    labels = []
    lmappings={'contradiction': -1, 'neutral':0, 'entailment':1}
    i = 0
    skips=0
    with open(dsfile) as f:
        for l in f:
            d=json.loads(l)
            if i % 10000 == 0:
                print('%d samples, %d skips' % (i,skips))
            if len(d['gold_label'])<2: # some pairs are not labeled, skip them
                skips += 1
                continue
            s0 = d['sentence1']
            s1 = d['sentence2']
            si0 = vocab.vectorize([s0], spad=None)
            si1 = vocab.vectorize([s1], spad=None)
            label=lmappings[d['gold_label']]
            s0i.append(si0[0])
            s1i.append(si1[0])
            labels.append(int(label))
            i += 1
    return (s0i, s1i, labels)

if __name__ == "__main__":
    testset, vocabf = sys.argv[1:3]
    vocab = Vocabulary(sentence_gen([testset]), count_thres=2)
    pickle.dump(vocab, open(vocabf, "wb"))

    args = sys.argv[1:]
    if args[0] == '--revocab':
        revocab = True
        args = args[1:]
    else:
        revocab = False

    trainf,testf, dumpf, vocabf = args

    if revocab:
        vocab = Vocabulary(sentence_gen([trainf,testf]), count_thres=2)
        print('%d words' % (len(vocab.word_idx)))
        pickle.dump(vocab, open(vocabf, "wb"))
    else:
        vocab = pickle.load(open(vocabf, "rb"))
        print('%d words' % (len(vocab.word_idx)))

    s0i, s1i, f0, f1, labels = load_set(trainf, vocab)
    pickle.dump((s0i, s1i, f0, f1, labels), open(dumpf, "wb"))