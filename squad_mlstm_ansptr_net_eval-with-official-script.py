# -*- coding: utf-8 -*-

# Created by junfeng, saj on 11/04/16.


from __future__ import print_function

import json
import pickle
import time

import lasagne
import numpy
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from nltk.tokenize import word_tokenize

from custom_layers import CustomEmbedding, MatchLSTM, FakeFeatureDot2Layer, AnsPointerLayer, categorical_crossentropy, \
    categorical_accuracy
from squad_evaluate import evaluate


# In[2]:

def prepare(df, sequential=False):
    seqs_passage = []
    seqs_question = []
    seqs_answer = []
    for cc in df['sentence1']:
        seqs_passage.append(cc)
    for cc in df['sentence2']:
        seqs_question.append(cc)
    for cc in df['gold_label']:
        seqs_answer.append(cc)
    seqs_p = seqs_passage
    seqs_q = seqs_question
    seqs_a = seqs_answer

    lengths_p = [len(s) for s in seqs_p]
    lengths_q = [len(s) for s in seqs_q]
    lengths_a = [len(s) for s in seqs_a]

    n_samples = len(seqs_p)
    maxlen_p = numpy.max(lengths_p)
    # sequential or boundary
    maxlen_p += int(sequential)
    # question is to be attended, adds a NULL token
    maxlen_q = numpy.max(lengths_q) + 1

    lengths_a = np.array(lengths_a, dtype='int32')
    maxlen_a = numpy.max(lengths_a) + int(sequential)

    passages = numpy.zeros((n_samples, maxlen_p), dtype='int32')
    questions = numpy.zeros((n_samples, maxlen_q), dtype='int32')
    passage_masks = numpy.zeros((n_samples, maxlen_p), dtype='int32')
    question_masks = numpy.zeros((n_samples, maxlen_q), dtype='int32')
    answer = numpy.zeros((n_samples, maxlen_a), dtype='int32')
    answer_masks = np.zeros_like(answer)
    for idx, [s_p, s_q, s_a] in enumerate(zip(seqs_p, seqs_q, seqs_a)):
        assert lengths_q[idx] == len(s_q)
        passages[idx, :lengths_p[idx]] = s_p
        # id 0 is a special TOKEN NULL
        passage_masks[idx, :lengths_p[idx] + int(sequential)] = 1
        questions[idx, :lengths_q[idx]] = s_q
        question_masks[idx, :lengths_q[idx] + 1] = 1
        answer[idx, :lengths_a[idx]] = s_a
        if sequential:
            # End of passage index, index start from 0
            answer[idx, lengths_a[idx]] = lengths_p[idx]
        answer_masks[idx, :lengths_a[idx] + int(sequential)] = 1

    lengths_a += int(sequential)
    if not sequential:
        start_indices = answer[:, 0].reshape((-1, 1))
        # [start, end)
        # end_indices = answer[np.arange(n_samples), lengths_a - 1].reshape((-1, 1)) + 1
        end_indices = answer[np.arange(n_samples), lengths_a - 1].reshape((-1, 1))
        labels = np.concatenate([start_indices, end_indices], axis=1)
        return (passages, passage_masks,
                questions, question_masks,
                labels)
    else:
        return (passages, passage_masks,
                questions, question_masks,
                answer, maxlen_a, answer_masks, lengths_a)


# In[3]:

def find_indices(passage, answer):
    answer_indices = []
    start = 0
    for word in answer:
        try:
            word_index = passage.index(word, start)
        except:
            # print(passage)
            print('Warning: word {} not in raw context.'.format(word))
            word_index = 0
        answer_indices.append(word_index)
        start = word_index
    return answer_indices


def apply_answer(r):
    passage = r.sentence1
    answer = r.gold_label
    return find_indices(passage, answer)


def transform_index_ans_to_list(df):
    return df.index_answer.apply(lambda s: [s])


def check_which_is_dev(dev_real, converted):
    qid1 = set(dev_real.qid)
    qid2 = set(converted.qid)
    intersection = qid1.intersection(qid2)
    return len(qid1) == len(qid2) and len(qid1) == len(intersection)


with open('./SQUAD-data/dictionary_squad.pkl', 'rb') as f:
    print('Loading dictionary ...')
    dictionary = pickle.load(f, encoding='latin1')
    reversed_dict = {v: k for k, v in dictionary.items()}

with open('SQUAD-data/dev-v1.1.json', 'r') as f:
    print('Loading raw dev json file ...')
    dataset_json = json.load(f)
    dev_raw = dataset_json['data']
    del dataset_json


# convert dev_raw
def convert_dev_raw(dev_raw):
    qids = []
    contexts = []
    questions = []
    answers_indices = []
    for article in dev_raw:
        title = article['title']
        # print('Title is {}'.format(title))
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            tokenized_context = word_tokenize(context)
            for qa in paragraph['qas']:
                qid = qa['id']
                question = qa['question']
                answers = qa['answers']
                # uses first answer, just for prepare function work
                ans = answers[0]
                ans_text = ans['text']
                tokenized_ans_text = word_tokenize(ans_text)
                ans_indices = find_indices(tokenized_context, tokenized_ans_text)
                assert len(ans_indices) > 0, '{} doesn\'t appear in passage'.format(ans)
                tokenized_question = word_tokenize(question)
                qids.append(qid)
                contexts.append(list(map(lambda w: dictionary[w], tokenized_context)))
                questions.append(list(map(lambda w: dictionary[w], tokenized_question)))
                answers_indices.append(ans_indices)
    # constructs dataframe
    dev_real_df = pd.DataFrame(data=dict(qid=qids,
                                         sentence1=contexts,
                                         sentence2=questions,
                                         gold_label=answers_indices), )
    return dev_real_df
print('Convert dev json to right format')
dev_real_df = convert_dev_raw(dev_raw)

print('Loading data ...')
train_df, dev_df, test_df = (None, None, None)
with open('./SQUAD-data/converted_train_squad.pkl', 'rb') as f:
    print('Loading train ...')
    train_df = pickle.load(f, encoding='latin1')
    # print(len(train_df))
    # filtered_s2 = train_df.sentence2.apply(lambda s2: len(s2) != 0)
    # train_df = train_df[filtered_s2]
    # print(len(train_df))
    train_df = train_df.reset_index(drop=True)
    print('Max length passage: ', train_df.sentence1.apply(lambda s1: len(s1)).max())
    print('Max length question: ', train_df.sentence2.apply(lambda s2: len(s2)).max())
    print('Max length answer: ', train_df.gold_label.apply(lambda gl: len(gl)).max())
    print(len(train_df))
    # train_df['gold_label'] = train_df.apply(apply_answer, axis=1)
#    train_df['gold_label'] = transform_index_ans_to_list(train_df)
with open('./SQUAD-data/converted_dev_squad.pkl', 'rb') as f:
    print('Loading dev ...')
    dev_df = pickle.load(f, encoding='latin1')
    # print(len(dev_df))
    # filtered_s2 = dev_df.sentence2.apply(lambda s2: len(s2) != 0)
    # dev_df = dev_df[filtered_s2]
    # print(len(dev_df))
    dev_df = dev_df.reset_index(drop=True)
    print('Max length passage: ', dev_df.sentence1.apply(lambda s1: len(s1)).max())
    print('Max length question: ', dev_df.sentence2.apply(lambda s2: len(s2)).max())
    print('Max length answer: ', dev_df.gold_label.apply(lambda gl: len(gl)).max())
    print(len(dev_df))
    print('Is real dev set: {}'.format(check_which_is_dev(dev_real_df, dev_df)))
    # dev_df['gold_label'] = dev_df.apply(apply_answer, axis=1)
#    dev_df['gold_label'] = transform_index_ans_to_list(dev_df)
with open('./SQUAD-data/converted_test_squad.pkl', 'rb') as f:
    print('Loading test ...')
    test_df = pickle.load(f, encoding='latin1')
    # print(len(test_df))
    # filtered_s2 = test_df.sentence2.apply(lambda s2: len(s2) != 0)
    # test_df = test_df[filtered_s2]
    # print(len(test_df))
    test_df = test_df.reset_index(drop=True)
    print('Max length passage: ', test_df.sentence1.apply(lambda s1: len(s1)).max())
    print('Max length question: ', test_df.sentence2.apply(lambda s2: len(s2)).max())
    print('Max length answer: ', test_df.gold_label.apply(lambda gl: len(gl)).max())
    print(len(test_df))
    print('Is real dev set: {}'.format(check_which_is_dev(dev_real_df, test_df)))
    # test_df['gold_label'] = test_df.apply(apply_answer, axis=1)
#    test_df['gold_label'] = transform_index_ans_to_list(test_df)

print('Concat to use full train data')
train_df = pd.concat([train_df, dev_df], ignore_index=True)
print('Full train dataset info')
print('Max length passage: ', dev_df.sentence1.apply(lambda s1: len(s1)).max())
print('Max length question: ', dev_df.sentence2.apply(lambda s2: len(s2)).max())
print('Max length answer: ', dev_df.gold_label.apply(lambda gl: len(gl)).max())
# test_df is the real dev
dev_df = test_df
print('Real dev dataset info')
print('Max length passage: ', dev_df.sentence1.apply(lambda s1: len(s1)).max())
print('Max length question: ', dev_df.sentence2.apply(lambda s2: len(s2)).max())
print('Max length answer: ', dev_df.gold_label.apply(lambda gl: len(gl)).max())
print(len(dev_df))
del test_df

passage_max = 766 + 1  # 196
question_max = 60 + 1  # 33

# In[8]:

num_epochs = 1  # 20#10
k = 150
batch_size = 30  # 2#30
dropout_rate = 0.3
learning_rate = 0.001
l2_weight = 0.
sequential = False
display_freq = 100
save_freq = 1000
load_previous = False

print('num_epochs: {}'.format(num_epochs))
print('k: {}'.format(k))
print('batch_size: {}'.format(batch_size))
print('display_frequency: {}'.format(display_freq))
print('save_frequency: {}'.format(save_freq))
print('load previous: {}'.format(load_previous))
if sequential:
    suffix = 'sequential'
else:
    suffix = 'boundary'
print('Model: {}'.format(suffix))
save_filename = './SQUAD-data/mc_mlstm_ansptr_{}_model.npz'.format(suffix)
prepare(train_df.loc[:batch_size])
print("Building network ...")
passage_var = T.imatrix('passage_var')
passage_mask = T.imatrix('passage_mask')
question_var = T.imatrix('question_var')
question_mask = T.imatrix('question_mask')
max_steps_var = T.iscalar('max_steps_var')
ans_mask_var = T.imatrix('ans_mask_var')
ans_length_var = T.ivector('ans_length_var')
target_var = T.imatrix('target_var')

unchanged_W = pickle.load(open('./SQUAD-data/unchanged_W_squad.pkl', 'rb'), encoding='latin1')
unchanged_W = unchanged_W.astype('float32')
unchanged_W_shape = unchanged_W.shape
oov_in_train_W = pickle.load(open('./SQUAD-data/oov_in_train_W_squad.pkl', 'rb'), encoding='latin1')
oov_in_train_W = oov_in_train_W.astype('float32')
oov_in_train_W_shape = oov_in_train_W.shape
print('unchanged_W.shape: {0}'.format(unchanged_W_shape))
print('oov_in_train_W.shape: {0}'.format(oov_in_train_W_shape))

l_passage = lasagne.layers.InputLayer(shape=(None, passage_max + int(sequential)), input_var=passage_var)
l_passage_mask = lasagne.layers.InputLayer(shape=(None, passage_max + int(sequential)), input_var=passage_mask)
l_question = lasagne.layers.InputLayer(shape=(None, question_max + 1), input_var=question_var)
l_question_mask = lasagne.layers.InputLayer(shape=(None, question_max + 1), input_var=question_mask)

passage_embedding = CustomEmbedding(l_passage, unchanged_W, unchanged_W_shape,
                                    oov_in_train_W, oov_in_train_W_shape,
                                    p=dropout_rate)
# weights shared with passage_embedding
question_embedding = CustomEmbedding(l_question, unchanged_W=passage_embedding.unchanged_W,
                                     unchanged_W_shape=unchanged_W_shape,
                                     oov_in_train_W=passage_embedding.oov_in_train_W,
                                     oov_in_train_W_shape=oov_in_train_W_shape,
                                     p=dropout_rate,
                                     dropout_mask=passage_embedding.dropout_mask)
# LSTM layers
l_passage_lstm = lasagne.layers.LSTMLayer(passage_embedding, k, peepholes=False,
                                          mask_input=l_passage_mask,
                                          only_return_final=False)
l_question_lstm = lasagne.layers.LSTMLayer(question_embedding, k, peepholes=False,
                                           mask_input=l_question_mask,
                                           only_return_final=False)

l_passage_lstm = FakeFeatureDot2Layer(l_passage_lstm)
forward_mlstm = MatchLSTM(l_passage_lstm, k, peepholes=False, mask_input=l_passage_mask,
                          only_return_final=False,
                          encoder_input=l_question_lstm, encoder_mask_input=l_question_mask,
                          )
backward_mlstm = MatchLSTM(l_passage_lstm, k, peepholes=False, backwards=True,
                           mask_input=l_passage_mask, only_return_final=False,
                           encoder_input=l_question_lstm, encoder_mask_input=l_question_mask,
                           )
# concat on feature axis, not time step axis
mlstm = lasagne.layers.ConcatLayer([forward_mlstm, backward_mlstm], axis=-1)

# dropout_rate = 0.
if dropout_rate > 0.:
    print('apply dropout rate {} to decoder'.format(dropout_rate))
    mlstm = lasagne.layers.DropoutLayer(mlstm, dropout_rate)
if sequential:
    # maybe vary
    max_steps = max_steps_var
else:
    max_steps = 2
l_ans_softmax = AnsPointerLayer(mlstm, num_units=k,
                                max_steps=max_steps,
                                mask_input=l_passage_mask)
if load_previous:
    print('loading previous saved model ...')
    # And load them again later on like this:
    with np.load(save_filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(l_ans_softmax, param_values)

if not sequential:
    ans_mask = T.ones((1, 2))
    ans_length = T.constant(2)
else:
    ans_mask = ans_mask_var
    ans_length = ans_length_var

# lasagne.layers.get_output produces a variable for the output of the net
# prediction's shape is (n_batch, max_steps, passage_seq_len)
prediction = lasagne.layers.get_output(l_ans_softmax, deterministic=False)
loss, _ = categorical_crossentropy(prediction, target_var, ans_mask, ans_length)
cost = loss.mean()
if l2_weight > 0.:
    # apply l2 regularization
    print('apply l2 penalty to all layers, weight: {}'.format(l2_weight))
    l2_penalty = lasagne.regularization.regularize_network_params(l_ans_softmax,
                                                                  lasagne.regularization.l2) * l2_weight
    cost += l2_penalty
# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(l_ans_softmax, trainable=True)
# Compute adam updates for training
print("Computing updates ...")
updates = lasagne.updates.adam(cost, all_params, learning_rate=learning_rate)

# Again test_prediction's shape is (n_batch, max_steps, passage_seq_len)
test_prediction = lasagne.layers.get_output(l_ans_softmax, deterministic=True)
test_loss, test_probs = categorical_crossentropy(test_prediction, target_var, ans_mask, ans_length)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc, predicted_label = categorical_accuracy(test_probs, target_var, ans_mask, ans_length)
test_acc = T.mean(test_acc)

# Theano functions for training and computing cost
print("Compiling functions ...")
arguments_var = [passage_var, passage_mask,
                 question_var, question_mask,
                 target_var]
if sequential:
    arguments_var.extend([max_steps_var, ans_mask_var, ans_length_var])
train_fn = theano.function(arguments_var, cost, updates=updates)
val_fn = theano.function(arguments_var, [test_loss, test_acc])
predict_arguments_var = [passage_var, passage_mask,
                         question_var, question_mask]
if sequential:
    predict_arguments_var.append(max_steps_var)
predict_fn = theano.function(predict_arguments_var,
                             predicted_label)

print('train_df.shape: {0}'.format(train_df.shape))
print('dev_df.shape: {0}'.format(dev_df.shape))
# print('test_df.shape: {0}'.format(test_df.shape))

# initial evaluation
val_err = 0
val_acc = 0
val_batches = 0
predicted_dev_ans = dict()
for start_i in range(0, len(dev_df), batch_size):
    batched_df = dev_df[start_i:start_i + batch_size]
    batched = prepare(batched_df, sequential=sequential)
    err, acc = val_fn(*batched)
    pred_indic = predict_fn(*batched[:4])
    base_index = batched_df.index.values[0]
    for index, row in batched_df.iterrows():
        start_index, end_index = pred_indic[index - base_index]
        answer_ids = row['sentence1'][start_index:end_index + 1]
        answer_words = []
        for ans_id in answer_ids:
            answer_words.append(reversed_dict[ans_id])
        answer_text = ' '.join(answer_words)
        predicted_dev_ans[row['qid']] = answer_text
    val_err += err
    val_acc += acc
    val_batches += 1
eval_result = evaluate(dev_raw, predicted_dev_ans)
print('Initial evaluation')
print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
print("  validation accuracy:\t\t{:.2f} %".format(
    val_acc / val_batches * 100))
print('  validation exact_match: {}, f1: {}'.format(eval_result['exact_match'], eval_result['f1']))
try:
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        save_at = time.time()
        display_at = time.time()
        for start_i in range(0, len(shuffled_train_df), batch_size):
            batched_df = shuffled_train_df[start_i:start_i + batch_size]
            batched = prepare(batched_df, sequential=sequential)
            train_err += train_fn(*batched)
            err, acc = val_fn(*batched)
            train_acc += acc
            train_batches += 1
            # display
            if train_batches % display_freq == 0:
                print("Seen {:d} samples, time used: {:.3f}s".format(
                    start_i + batch_size, time.time() - display_at))
                print("  current training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  current training accuracy:\t\t{:.6f}".format(train_acc / train_batches))
                display_at = time.time()
            # do tmp save model
            if train_batches % save_freq == 0:
                print('saving to ..., time used: {:.3f}s'.format(time.time() - save_at))
                #                np.savez(save_filename,
                #                         *lasagne.layers.get_all_param_values(l_ans_softmax))
                save_at = time.time()

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        predicted_dev_ans = dict()
        for start_i in range(0, len(dev_df), batch_size):
            batched_df = dev_df[start_i:start_i + batch_size]
            batched = prepare(batched_df, sequential=sequential)
            err, acc = val_fn(*batched)
            pred_indic = predict_fn(*batched[:4])
            base_index = batched_df.index.values[0]
            for index, row in batched_df.iterrows():
                start_index, end_index = pred_indic[index - base_index]
                answer_ids = row['sentence1'][start_index:end_index + 1]
                answer_words = []
                for ans_id in answer_ids:
                    answer_words.append(reversed_dict[ans_id])
                answer_text = ' '.join(answer_words)
                predicted_dev_ans[row['qid']] = answer_text
            val_err += err
            val_acc += acc
            val_batches += 1
        eval_result = evaluate(dev_raw, predicted_dev_ans)
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        print('  validation exact_match: {}, f1: {}'.format(eval_result['exact_match'], eval_result['f1']))

        # After training, we compute and print the test error:
        # TODO, doesn't need
        # test_err = 0
        # test_acc = 0
        # test_batches = 0
        # answer_df = pd.DataFrame(columns=['qid', 'answer_text'])
        # for start_i in range(0, len(test_df), batch_size):
        #     batched_df = test_df[start_i:start_i + batch_size]
        #     batched = prepare(batched_df, sequential=sequential)
        #     err, acc = val_fn(*batched)
        #     pred_indic = predict_fn(*batched[:4])
        #     base_index=batched_df.index.values[0]
        #     for index, row in batched_df.iterrows():
        #         start_index, end_index = pred_indic[index-base_index]
        #         answer_ids = row['sentence1'][start_index:end_index+1]
        #         answer_words = []
        #         for ans_id in answer_ids:
        #             answer_words.append(reversed_dict[ans_id])
        #         answer_text = ' '.join(answer_words)
        #         answer_df.loc[len(answer_df)] = [row['qid'], answer_text]
        #     test_err += err
        #     test_acc += acc
        #     test_batches += 1
        # # print("Final results:")
        # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        # print("  test accuracy:\t\t{:.2f} %".format(
        #     test_acc / test_batches * 100))

        # Optionally, you could now dump the network weights to a file like this:
        # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        #
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)
except KeyboardInterrupt:
    print('exit ...')
