
# -*- coding: utf-8 -*-
# @Time    : 20/12/2017 11:28 PM
# @Author  : 伊甸一点
# @FileName: test.py
# @Software: PyCharm
# @Blog    : http://zpfbuaa.github.io
import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from squad.utils import get_word_span, get_word_idx, process_tokens


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    print ('source_dir',source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, "train-v1.1.json")
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, "dev-v1.1.json")
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.mode == 'full': # default go into this
        prepro_each(args, 'train', out_name='train') # first is train data prepro
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'dev', out_name='test')
    elif args.mode == 'all':
        create_all(args)
        prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
        prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
        prepro_each(args, 'all', out_name='train')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'train', args.train_ratio, 1.0, out_name='dev')
        prepro_each(args, 'dev', out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    # elif args.tokenizer == 'Stanford':
    #     from my.corenlp_interface import CoreNLPInterface
    #     interface = CoreNLPInterface(args.url, args.port)
    #     sent_tokenize = interface.split_doc
    #     word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    # print type(sent_tokenize) # <type 'function'>
    # print type(word_tokenize) # <type 'function'>
    source_path = in_path or os.path.join(args.source_dir, "{}-v1.1.json".format(data_type)) # train or dev data
    source_data = json.load(open(source_path, 'r')) # default first is train

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio)) # the article start index
    stop_ai = int(round(len(source_data['data']) * stop_ratio)) # the article end index
    #print (start_ai,stop_ai) #(0,442) # total articles
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        x.append(xp) # append xp is changed as following codes, therefore in python the value of list is passed by reference
        cx.append(cxp)
        p.append(pp)
        for pi, para in enumerate(article['paragraphs']): # pi is idx para is dict
            # wordss
            #print ('pi',pi)
            context = para['context'] # context type : list
            context = context.replace("''", '" ')
            context = context.replace("``", '" ') # replace some special characters
            xi = list(map(word_tokenize, sent_tokenize(context)))
            # print ('len(xi)',len(xi)) # len = 1
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            # print ('xi',xi) #  FIXME: this result is really bad. we can improve this phrase
            # print ('cxi',cxi)
            #print ('context',context)
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)
            #print ('xi', xi)  # xi context word list
            for xij in xi:
                # print ('xij',xij) #xijk word vector
                for xijk in xij:
                    # print ('xijk',xijk) # xijk word level
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        # print ('xijkl', xijkl)  #xijkl character level
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi] # ai means title_id and pi means paragraphs_id
            # print ('ai',ai)
            # print ('pi',pi)
            # print ('rxi', rxi)
            # print ('len(x[ai]) - 1',len(x[ai]) - 1)
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                cqi = [list(qij) for qij in qi]
                # print ('qi',qi) # question word level
                print ('qi',qi)
                print ('type qi', type(qi))
                # print ('cqi',cqi) # character level
                yi = []
                cyi = []
                answers = []
                # print ('qa[\'answers\']',qa['answers'])
                # print ('type',type(qa['answers'])) # list
                # print ('len', len(qa['answers']))  # qa['answers'] len is 1
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    # print ('answers',answers)
                    #print ('answer_text',answer_text)
                    # print ('type answer_text', type(answer_text)) #  <type 'unicode'>
                    answer_start = answer['answer_start'] # the train data has several answer_start
                    # print ('answer_start',answer_start) # <type 'int'>
                    # print ('type answer_start', type(answer_start))
                    answer_stop = answer_start + len(answer_text) # calculate the length of answer
                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop) # start character index and end character index
                    # print ('yi0',yi0)
                    # print ('type yi0', type(yi0)) # <type 'tuple'>)
                    # print ('yi1', yi1)
                    # print ('type yi1', type(yi1))# <type 'tuple'>)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    w0 = xi[yi0[0]][yi0[1]]
                    w1 = xi[yi1[0]][yi1[1]-1]
                    # print ('(xi[0][1])',(xi[0][3]))
                    # print ('w0',w0)
                    # print ('type w0', type(w0)) # <type 'unicode'>
                    # print ('w1', w1)
                    # print ('type w1', type(w1)) # <type 'unicode'>
                    i0 = get_word_idx(context, xi, yi0)
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1))
                    # print ('answer_start', answer_start)  # <type 'int'>
                    # print ('i0', i0)
                    # print ('i1', i1)
                    # print ('type i0', type(i0))
                    # print ('type i1', type(i1))
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    # print ('cyi0',cyi0)
                    # print ('cyi1', cyi1)
                    # print ('w0',w0)
                    # print ('w1',w1)
                    # print('answer_text, w0[cyi0:],w1[:cyi1+1]',answer_text, w0[cyi0:], w1[:cyi1+1])
                    if cyi0 < 0:
                        print (answer_text, w0, cyi0)
                        continue
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1]) # answer start word(start,end).. answer end word(start,end)
                    cyi.append([cyi0, cyi1]) # final answer character level start index, end_index
                #print ('qi',qi)
                for qij in qi:
                    # print ('qij', qij)
                    word_counter[qij] += 1 # word level
                    lower_word_counter[qij.lower()] += 1 # lower word level
                    for qijk in qij: # character level
                        char_counter[qijk] += 1 # character appears times
                q.append(qi) # question list word by word
                cq.append(cqi) # question character by character
                y.append(yi) # yi is list[list] y is list[list[ list[ yi0 (start,end) , yi1 (start,end) ]  ] ]
                cy.append(cyi)# cyi is list [list] cy is list[list[list[cyi0 (start,end), cyi1 (start,end) ] ] ]
                rx.append(rxi)# rxi is list[title_idx, paragraph_idx]
                rcx.append(rxi) # the same as rx
                # print ('qa[\'id\']',qa['id'])
                ids.append(qa['id']) # qa['id'] store the question id, therefore ids are total question ids
                idxs.append(len(idxs)) # that means 0 , 1, 2, 3, ... n-1
                answerss.append(answers) # answers store the answer_text, therefore answerss store the total answers
                # print ('q',q)
                # print ('cq', cq)
                # print ('y', y)
                # print ('cy', cy)
                # print ('rx', rx)
                # print ('rcx', rcx)
                # print ('ids', ids)
                # print ('idxs', idxs)
                # print ('answerss', answerss)
                # print ('x', x)
                # print ('cx', cx)
                # print ('p', p)

            if args.debug:
                break

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)



if __name__ == "__main__":
    main()