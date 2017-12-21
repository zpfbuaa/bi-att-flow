import json
import os
import random
import itertools
import math
from collections import defaultdict

import numpy as np

from my.tensorflow import grouper
from my.utils import index


class Data(object):
    def get_size(self):
        raise NotImplementedError()

    def get_by_idxs(self, idxs):
        """
        Efficient way to obtain a batch of items from filesystem
        :param idxs:
        :return dict: {'X': [,], 'Y', }
        """
        data = defaultdict(list)
        for idx in idxs:
            each_data = self.get_one(idx)
            for key, val in each_data.items():
                data[key].append(val)
        return data

    def get_one(self, idx):
        raise NotImplementedError()

    def get_empty(self):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()


class DataSet(object):
    def __init__(self, data, data_type, shared=None, valid_idxs=None):
        self.data = data  # e.g. {'X': [0, 1, 2], 'Y': [2, 3, 4]}
        self.data_type = data_type
        self.shared = shared
        total_num_examples = self.get_data_size()
        self.valid_idxs = range(total_num_examples) if valid_idxs is None else valid_idxs
        self.num_examples = len(self.valid_idxs)

    def _sort_key(self, idx):
        rx = self.data['*x'][idx]
        x = self.shared['x'][rx[0]][rx[1]]
        return max(map(len, x))

    def get_data_size(self):
        if isinstance(self.data, dict):
            return len(next(iter(self.data.values())))
        elif isinstance(self.data, Data):
            return self.data.get_size()
        raise Exception()

    def get_by_idxs(self, idxs):
        if isinstance(self.data, dict):
            out = defaultdict(list)
            for key, val in self.data.items():
                out[key].extend(val[idx] for idx in idxs)
            return out
        elif isinstance(self.data, Data):
            return self.data.get_by_idxs(idxs)
        raise Exception()

    def get_batches(self, batch_size, num_batches=None, shuffle=False, cluster=False):
        """

        :param batch_size:
        :param num_batches:
        :param shuffle:
        :param cluster: cluster examples by their lengths; this might give performance boost (i.e. faster training).
        :return:
        """
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        if num_batches is None:
            num_batches = num_batches_per_epoch
        num_epochs = int(math.ceil(num_batches / num_batches_per_epoch))

        if shuffle:
            random_idxs = random.sample(self.valid_idxs, len(self.valid_idxs))
            if cluster:
                sorted_idxs = sorted(random_idxs, key=self._sort_key)
                sorted_grouped = lambda: list(grouper(sorted_idxs, batch_size))
                grouped = lambda: random.sample(sorted_grouped(), num_batches_per_epoch)
            else:
                random_grouped = lambda: list(grouper(random_idxs, batch_size))
                grouped = random_grouped
        else:
            raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
            grouped = raw_grouped

        batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        for _ in range(num_batches):
            batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
            batch_data = self.get_by_idxs(batch_idxs)
            shared_batch_data = {}
            for key, val in batch_data.items():
                if key.startswith('*'):
                    assert self.shared is not None
                    shared_key = key[1:]
                    shared_batch_data[shared_key] = [index(self.shared[shared_key], each) for each in val]
            batch_data.update(shared_batch_data)

            batch_ds = DataSet(batch_data, self.data_type, shared=self.shared)
            yield batch_idxs, batch_ds

    def get_multi_batches(self, batch_size, num_batches_per_step, num_steps=None, shuffle=False, cluster=False):
        batch_size_per_step = batch_size * num_batches_per_step
        batches = self.get_batches(batch_size_per_step, num_batches=num_steps, shuffle=shuffle, cluster=cluster)
        multi_batches = (tuple(zip(grouper(idxs, batch_size, shorten=True, num_groups=num_batches_per_step),
                         data_set.divide(num_batches_per_step))) for idxs, data_set in batches)
        return multi_batches

    def get_empty(self):
        if isinstance(self.data, dict):
            data = {key: [] for key in self.data}
        elif isinstance(self.data, Data):
            data = self.data.get_empty()
        else:
            raise Exception()
        return DataSet(data, self.data_type, shared=self.shared)

    def __add__(self, other):
        if isinstance(self.data, dict):
            data = {key: val + other.data[key] for key, val in self.data.items()}
        elif isinstance(self.data, Data):
            data = self.data + other.data
        else:
            raise Exception()

        valid_idxs = list(self.valid_idxs) + [valid_idx + self.num_examples for valid_idx in other.valid_idxs]
        return DataSet(data, self.data_type, shared=self.shared, valid_idxs=valid_idxs)

    def divide(self, integer):
        batch_size = int(math.ceil(self.num_examples / integer))
        idxs_gen = grouper(self.valid_idxs, batch_size, shorten=True, num_groups=integer)
        data_gen = (self.get_by_idxs(idxs) for idxs in idxs_gen)
        ds_tuple = tuple(DataSet(data, self.data_type, shared=self.shared) for data in data_gen)
        return ds_tuple


def load_metadata(config, data_type):
    metadata_path = os.path.join(config.data_dir, "metadata_{}.json".format(data_type))
    with open(metadata_path, 'r') as fh:
        metadata = json.load(fh)
        for key, val in metadata.items():
            config.__setattr__(key, val)
        return metadata


"""
# data_type : dev, train
# ref: whether load data from the prepro saved files.
# data_filter: to filter some data
"""
def read_data(config, data_type, ref, data_filter=None):
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type)) # data_dir default is 'data/squad'
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    num_examples = len(next(iter(data.values()))) # total number of question
    if data_filter is None:
        valid_idxs = range(num_examples) # if no data_filter passed in, then this valid_idxs is just a list from [0,1,2,3, ... , len(num_examples)
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each, shared))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))
    """
    # shared_path: default  ""
    # lower_word: default  True
    # lower_word: default  True
    # finetune: default False
    # known_if_glove: default True
    # use_glove_for_unk: default True
    # word_count_th: default 10 [ only a word appears time over this value can be added into data]
    # char_counter: default 50 [only a character appears time over 50 can be added into data]
    """
    shared_path = config.shared_path or os.path.join(config.out_dir, "shared.json")
    if not ref: # if not read from save file FIXME: to run this code piece, need to change ref to False [means: change config.load to False]
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec'] # back to file prepro.py
        word_counter = shared['lower_word_counter'] if config.lower_word else shared['word_counter']# back to file prepro.py
        char_counter = shared['char_counter']
        if config.finetune:
            shared['word2idx'] = {word: idx + 2 for idx, word in # FIXME: don't get why idx + 2
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th or (config.known_if_glove and word in word2vec_dict))}
        else:
            assert config.known_if_glove
            assert config.use_glove_for_unk
            shared['word2idx'] = {word: idx + 2 for idx, word in # FIXME: don't get why idx + 2
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th and word not in word2vec_dict)} # FIXME: don't underatand why word not in word2vec_dict
        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                        if count > config.char_count_th)}
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['word2idx'][NULL] = 0 # shared['word2idx'] type dict
        shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0 # shared['char2idx'] type dict
        shared['char2idx'][UNK] = 1 # TODO: unkonw character or NULL just to 1
        json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx']}, open(shared_path, 'w')) # add shared json to file
    else:
        new_shared = json.load(open(shared_path, 'r')) # just read shared_file
        for key, val in new_shared.items():
            shared[key] = val # read from file
    """
    # this handle the word in word2vec_dict but not appear in shared['word2idx']
    # word2vec_dict >= word2idx
    # 1. create a new_word2idx_dict
    # 2. calculate idx2vec_dict 
    # 3. change idx2vec_dict to numpy array (float)
    # 3. create a new key: new_emb_mat
    """
    if config.use_glove_for_unk:
        # create new word2idx and word2vec
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
        shared['new_word2idx'] = new_word2idx_dict
        offset = len(shared['word2idx'])
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec'] # FIXME: this is duplicate with the above code
        new_word2idx_dict = shared['new_word2idx'] # FIXME: whether thc code is useful?
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        shared['new_emb_mat'] = new_emb_mat

    data_set = DataSet(data, data_type, shared=shared, valid_idxs=valid_idxs) # self definition dataset
    return data_set


def get_squad_data_filter(config):
    def data_filter(data_point, shared): # TODO: complete to analyse the data_filter
        assert shared is not None
        """
        ############# [2017_12_21 15:35:07 zpf] #############
        Detail About the Data Type: 
        rx: [article_id, paragraph_id] (from 0 to article_num - 1 ) (from 0 to paragraph_num - 1)
        rcx: the same as rx
        q: [question_1_list_word_level, question_2_word_level_list ...]
        cq: [question_1_list_character_level_list, question_2_list_character_level_list ...]
        y: [ [[(0,answer1_start_idx),(-,answer1_end_idx)]], [[(0,answer2_start_idx),(-,answer2_end_idx)]] ... [[]] ]
        x: [ [[[article1_paragraph1_word1,paragraph1_word2 ...]] , [[paragraph2_word1, paragraph2_word2 ...]], ...], [[[]]] ]
        cx: [[ [[[paragraph1_word1_char1,paragraph1_word1_char2...], [paragraph1_word2_char1,paragraph1_word1_char2 ...]]] , [[[p2_w1_c1,p2_w1_c2...],[p2_w2_c1,p2_w2_c2...],]] ... [[]] ]]
        """

        rx, rcx, q, cq, y = (data_point[key] for key in ('*x', '*cx', 'q', 'cq', 'y'))
        x, cx = shared['x'], shared['cx']
        if len(q) > config.ques_size_th:
            return False

        # x filter
        xi = x[rx[0]][rx[1]]
        if config.squash:
            for start, stop in y:
                stop_offset = sum(map(len, xi[:stop[0]]))
                if stop_offset + stop[1] > config.para_size_th: # para_size_th default is 256
                    return False
            return True

        if config.single:
            for start, stop in y:
                if start[0] != stop[0]:
                    return False

        if config.data_filter == 'max':
            for start, stop in y:
                    if stop[0] >= config.num_sents_th: # num_sents_th default is 8
                        return False
                    if start[0] != stop[0]:
                        return False
                    if stop[1] >= config.sent_size_th: # sent_size_th default is 400
                        return False
        elif config.data_filter == 'valid':
            if len(xi) > config.num_sents_th:
                return False
            if any(len(xij) > config.sent_size_th for xij in xi):
                return False
        elif config.data_filter == 'semi':
            """
            Only answer sentence needs to be valid.
            """
            for start, stop in y:
                if stop[0] >= config.num_sents_th:
                    return False
                if start[0] != start[0]: # FIXME: start[0] != stop[0] this is wrong ? i am not sure.
                    return False
                if len(xi[start[0]]) > config.sent_size_th:
                    return False
        else:
            raise Exception()

        return True
    return data_filter

"""
# mode: default test 
# num_sents_th: default 8
# sent_size_th: default 400
# para_size_th: default 256
# word_size_th: default 16 len(internationally) = 15
"""
def update_config(config, data_sets):
    config.max_num_sents = 0
    config.max_sent_size = 0
    config.max_ques_size = 0
    config.max_word_size = 0
    config.max_para_size = 0
    for data_set in data_sets:
        data = data_set.data
        shared = data_set.shared
        for idx in data_set.valid_idxs: # total load question numbers !!!
            # TODO: this is quiet important! create pair (articel_id, such_article_paragraph_id) for each question
            # FIXME: actual rx is not a rx as before!!!
            rx = data['*x'][idx] # the idx's pair of article, paragraph. This means: this question belong to which paragraph and which article
            q = data['q'][idx] # the idx's question
            sents = shared['x'][rx[0]][rx[1]] # (article_id, paragraph_id)
            config.max_para_size = max(config.max_para_size, sum(map(len, sents))) # WARNNING: here is sum
            config.max_num_sents = max(config.max_num_sents, len(sents))
            config.max_sent_size = max(config.max_sent_size, max(map(len, sents))) # here is max
            config.max_word_size = max(config.max_word_size, max(len(word) for sent in sents for word in sent))
            if len(q) > 0:
                """
                ########## 2017_12_22 01:40:34 zpf ##########
                # find the max args
                # max_ques_size to max
                # 
                """
                config.max_ques_size = max(config.max_ques_size, len(q))
                config.max_word_size = max(config.max_word_size, max(len(word) for word in q))

    if config.mode == 'train':
        """
        ########## 2017_12_22 01:35:18 zpf ##########
        # update the config 
        """
        # TODO: find the min args!!!!
        config.max_num_sents = min(config.max_num_sents, config.num_sents_th)
        config.max_sent_size = min(config.max_sent_size, config.sent_size_th)
        config.max_para_size = min(config.max_para_size, config.para_size_th)

    # max_word_size to min
    # if larger than 16 then to 16
    config.max_word_size = min(config.max_word_size, config.word_size_th)

    """
    # char_vocab_size
    # word_emb_size
    # word_vocab_size
    """
    # TODO: analysis those parameters!
    config.char_vocab_size = len(data_sets[0].shared['char2idx'])
    config.word_emb_size = len(next(iter(data_sets[0].shared['word2vec'].values())))
    config.word_vocab_size = len(data_sets[0].shared['word2idx'])

    if config.single:
        config.max_num_sents = 1 # FIXME : this just confirm my upper guess!
    if config.squash:
        config.max_sent_size = config.max_para_size
        config.max_num_sents = 1
