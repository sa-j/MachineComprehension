# -*- coding: utf-8 -*-

# Created by junfeng, saj on 11/03/16.

# coding: utf-8

# In[1]:

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
import pandas as pd
import pickle
import json
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


# In[55]:

dataset_file = './SQuAD-raw/train-v1.1.json'
print('loading {} data ...'.format(dataset_file))
with open(dataset_file, 'r') as f:
    dataset_json = json.load(f)

dataset = dataset_json['data']

#ans_length = []
#for article in dataset:
#    for paragraph in article['paragraphs']:
#        for qa in paragraph['qas']:
#            qid = qa['id']
#            question = qa['question']
#            answers = qa['answers']
#            for ans in answers:
#                ans_length.append(len(word_tokenize(ans['text'])))
#                
#series = pd.Series(ans_length)
#series.value_counts()
#
#for article in dataset:
#    title = article['title']
#    # print('Title is {}'.format(title))
#    for paragraph in article['paragraphs']:
#        context = paragraph['context']
#        for qa in paragraph['qas']:
#            qid = qa['id']
#            question = qa['question']
#            answers = qa['answers']
#            for ans in answers:
#                ans_start = ans['answer_start']
#                ans_text = ans['text']
#                span = context[ans_start:ans_start + len(ans_text)]
#                assert span == ans_text, (context, qa)
#print('All answers can be found in context.')

#for article in dataset:
#    title = article['title']
#    # print('Title is {}'.format(title))
#    for paragraph in article['paragraphs']:
#        context = paragraph['context']
#        for qa in paragraph['qas']:
#            qid = qa['id']
#            question = qa['question']
#            answers = qa['answers']
#            for ans in answers:
#                ans_start = ans['answer_start']
#                ans_text = ans['text']
#                num_spans = len(context.split(ans_text)) - 1
#                assert num_spans == 1, (context, qa)
#
#print('Not all answers only appear once in context.')

qids = []
contexts = []
questions = []
tokenized_answers = []
answers_indices = []

for article in dataset:
    title = article['title']
    # print('Title is {}'.format(title))
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            qid = qa['id']
            question = qa['question']
            tokenized_question = word_tokenize(question)
#            answers = qa['answers']
#            for ans in answers:
            # no need for loop, as there is only 1 answer always in t
            ans = qa['answers'][0]
            ans_start = ans['answer_start']
            ans_text = ans['text']
            tokenized_ans_text = word_tokenize(ans_text)
            ctx_before_ans = word_tokenize(context[:ans_start])
            ctx_after_ans = word_tokenize(context[ans_start + len(ans_text):])
            tokenized_context = ctx_before_ans + tokenized_ans_text + ctx_after_ans
            start_index = len(ctx_before_ans)
            ans_indices = list(range(start_index, start_index + len(tokenized_ans_text)))
            qids.append(qid)
            contexts.append(tokenized_context)
            questions.append(tokenized_question)
            tokenized_answers.append(tokenized_ans_text)
            answers_indices.append(ans_indices)

df = pd.DataFrame(data=dict(qid=qids,
                            context=contexts,
                            question=questions,
                            answer=tokenized_answers,
                            ans_indices=answers_indices),
                 )
#with open('tokenize_suqad_train.pkl', 'wb') as fout:
#    pickle.dump(df, fout)
#print(len(qids))
#print(df.shape)
#print(df.head())
df_train = df[:77590]
df_dev = df[-10009:]

dataset_file = './SQuAD-raw/dev-v1.1.json'
print('loading {} data ...'.format(dataset_file))
with open(dataset_file, 'r') as f:
    dataset_json = json.load(f)

dataset_dev = dataset_json['data']

qids = []
contexts = []
questions = []
tokenized_answers = []
answers_indices = []

for article in dataset_dev:
    title = article['title']
    # print('Title is {}'.format(title))
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            qid = qa['id']
            question = qa['question']
            tokenized_question = word_tokenize(question)
#            answers = qa['answers']
#            for ans in answers:
            #FIXME fast solution: take the first answer only
            ans = qa['answers'][0]
            ans_start = ans['answer_start']
            ans_text = ans['text']
            tokenized_ans_text = word_tokenize(ans_text)
            ctx_before_ans = word_tokenize(context[:ans_start])
            ctx_after_ans = word_tokenize(context[ans_start + len(ans_text):])
            tokenized_context = ctx_before_ans + tokenized_ans_text + ctx_after_ans
            start_index = len(ctx_before_ans)
            ans_indices = list(range(start_index, start_index + len(tokenized_ans_text)))
            qids.append(qid)
            contexts.append(tokenized_context)
            questions.append(tokenized_question)
            tokenized_answers.append(tokenized_ans_text)
            answers_indices.append(ans_indices)

df_test = pd.DataFrame(data=dict(qid=qids,
                            context=contexts,
                            question=questions,
                            answer=tokenized_answers,
                            ans_indices=answers_indices),
                 )


# ### TODO
word2vec = Word2Vec.load_word2vec_format('/home/IAIS/sahmed/gensim/GoogleNews-vectors-negative300.bin', binary=True)


from collections import OrderedDict
#TODO improve pre-proc of worc2vec
inv_words, oov_words_in_train = OrderedDict(), set()
def check_sent(s):
    count = 0
    for r in s:
        #words = word_tokenize(r)
#        for w in words:
        for w in r:
            if type(w) != str:
                print(w)
                count += 1
                continue
            if w in inv_words or w in oov_words_in_train:
                continue
            if w not in word2vec:
                count += 1
                oov_words_in_train.add(w)
            else:
                inv_words[w] = word2vec.vocab[w].index
    return count

df_train[['context', 'question']].apply(check_sent)
print (len(inv_words))
print(len(oov_words_in_train))

oov_words_not_train = set()
def check_sent_dev_test(s):
    # s = s.str.translate(punctuation)
    count = 0
    for r in s:
        for w in r:
            if type(w) != str:
                print(w)
                count += 1
                continue
            if w in inv_words or w in oov_words_in_train or w in oov_words_not_train:
                continue
            if w not in word2vec:
                count += 1
                oov_words_not_train.add(w)
            else:
                inv_words[w] = word2vec.vocab[w].index
    return count

dev_test_df = pd.concat([df_dev, df_test], ignore_index=True)
print (dev_test_df.shape)
dev_test_df[['context', 'question']].apply(check_sent_dev_test)
print(len(inv_words))
print(len(oov_words_not_train))


# #### constructs words to ids dict
index = 0
dictionary = OrderedDict()
for k in inv_words:
    dictionary[k] = index
    index += 1
for k in oov_words_not_train:
    dictionary[k] = index
    index += 1
print(index)
for k in oov_words_in_train:
    dictionary[k] = index
    index += 1
print(index)

dictionary_filename = './SQUAD-data/dictionary_squad.pkl'
with open(dictionary_filename, 'wb') as f:
    pickle.dump(dictionary, f)


# #### constructs words enbedding W
inv_indices = list(inv_words.values())
inv_W = word2vec.syn0[inv_indices]

import numpy as np
rsg = np.random.RandomState(919)
oov_not_train_W = (rsg.rand(len(oov_words_not_train), word2vec.vector_size) - 0.5) / 10.0
unchanged_W = np.concatenate([inv_W, oov_not_train_W])
oov_in_train_W = (rsg.rand(len(oov_words_in_train), word2vec.vector_size) - 0.5) / 10.0
print(np.all([np.all(word2vec.syn0[i2] == unchanged_W[i1]) for i1, i2 in enumerate(inv_indices)]))

unchanged_W_filename = './SQUAD-data/unchanged_W_squad.pkl'
with open(unchanged_W_filename, 'wb') as f:
    pickle.dump(unchanged_W, f)
oov_in_train_W_filename = './SQUAD-data/oov_in_train_W_squad.pkl'
with open(oov_in_train_W_filename, 'wb') as f:
    pickle.dump(oov_in_train_W, f)

# #### convert sentence to list of words id
def to_ids(r):
#    premise_words = word_tokenize(r.question)
#    hypo_words = word_tokenize(r.context)
    premise_ids = []
    for w in r.question:
        premise_ids.append(dictionary[w])
    hypo_ids = []
    for w in r.context:
        hypo_ids.append(dictionary[w])
    r.loc['question'] = premise_ids
    r.loc['context'] = hypo_ids
    return r

df_train = df.fillna('')
converted_train = df_train.apply(to_ids, axis=1)
df_dev = df_dev.fillna('')
converted_dev = df_dev.apply(to_ids, axis=1)
df_test = df_test.fillna('')
converted_test = df_test.apply(to_ids, axis=1)

saved_columns = ['gold_label','answer','sentence1','qid','sentence2']
converted_train.columns = saved_columns
converted_dev.columns = saved_columns
converted_test.columns = saved_columns
#
with open('./SQUAD-data/converted_train_squad.pkl', 'wb') as f:
    pickle.dump(converted_train, f)
with open('./SQUAD-data/converted_dev_squad.pkl', 'wb') as f:
    pickle.dump(converted_dev, f)
with open('./SQUAD-data/converted_test_squad.pkl', 'wb') as f:
    pickle.dump(converted_test, f)
