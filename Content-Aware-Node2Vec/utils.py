import collections
import pickle
import re
from tqdm import tqdm
import numpy as np
import config
import os
from pprint import pprint
np.random.seed(1997)


class Utils(object):
    def __init__(self, walks, window_size, walk_length):
        self.phrase_dic = clean_dictionary(pickle.load(open(config.phrase_dic, 'rb')))
        self.walk_length = walk_length
        self.window_size = window_size
        if config.resume_training:
            print("Loading previous walks to continue training...")
            self.walks = pickle.load(open(os.path.join(config.checkpoint_dir, '{}_walks.p'.format(config.dataset_name)), 'rb'))
        else:
            self.walks = walks

        # when we evaluate we provide no walks..so we skip this part
        if self.walks is not None:
            self.frequencies, self.word2idx, self.idx2word = self.build_dataset(self.walks)
            self.vocabulary_size = len(self.word2idx)
            print("Total words: ", self.vocabulary_size)
            # the sample_table is used for negative sampling
            self.sample_table = self.create_sample_table()

    def build_word_vocab(self, walks):
        data_vocabulary = []  # in node2vec the words are nodeids and each walk represents a sentence (in word2vec terminology)
        word2idx = {}
        word2idx['PAD'] = 0
        word2idx['UNKN'] = len(word2idx)
        for walk in tqdm(walks):
            for nodeid in walk:
                data_vocabulary.append(nodeid)
                phrase = self.phrase_dic[int(nodeid)]
                for word in phrase:
                    try:
                        gb = word2idx[word]
                    except KeyError:
                        word2idx[word] = len(word2idx)
        idx2word = dict(zip(word2idx.values(), word2idx.keys()))
        return data_vocabulary, word2idx, idx2word

    def build_dataset(self, walks):
        print('Building dataset..')
        vocabulary, word2idx, idx2word = self.build_word_vocab(walks)
        count = []
        count.extend(collections.Counter(vocabulary).most_common())
        return count, word2idx, idx2word

    def create_sample_table(self):
        print('Creating sample table..')
        count = [element[1] for element in self.frequencies]
        pow_frequency = np.array(count) ** 0.75
        power = sum(pow_frequency)
        ratio = pow_frequency / power
        table_size = 1e8
        count = np.round(ratio * table_size)
        sample_table = []
        for idx, x in enumerate(count):
            sample = self.frequencies[idx]
            sample_table += [sample[0]] * int(x)
        return np.array(sample_table)


bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', ' ').replace('\\', '').replace("'", '').strip().lower()).split()


def get_index(w, vocab):
    try:
        return vocab[w]
    except KeyError:
        return vocab['UNKN']


def phr2idx(phr, word_vocab):
    p = [get_index(t, word_vocab) for t in phr.split()]
    return p


def clean_dictionary(phrase_dic):
    for nodeid, phrase in phrase_dic.items():
        # print("phrase before tokenize: ", phrase)
        # print("phrase after tokenize: ", tokenize(phrase) )
        # print("phrase after split: ", phrase.strip().lower().split() )
        # phrase_dic[nodeid] = tokenize(phrase)
        phrase_dic[nodeid] = phrase.strip().lower().split()
    return phrase_dic


def tokenize(x):
    return bioclean(x)
