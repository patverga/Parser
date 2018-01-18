#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Timothy Dozat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter, defaultdict
import itertools
import sys

from lib.etc.k_means import KMeans
from configurable import Configurable
from vocab import Vocab
from metabucket import Metabucket

#***************************************************************
class Dataset(Configurable):
  """"""

  #=============================================================
  def __init__(self, rel_map, filename, vocabs, builder, *args, **kwargs):
    """"""

    super(Dataset, self).__init__(*args, **kwargs)
    self._file_iterator = self.file_iterator(filename)
    self._train = (filename == self.train_file)
    self._metabucket = Metabucket(self._config, n_bkts=self.n_bkts)
    self._data = None
    self.vocabs = vocabs
    self.reverse_vocabs = [{v: k for k, v in vocab.iteritems()} for vocab in vocabs]
    self.rel_map = rel_map
    self.keep_labels = {'B-Chemical', 'I-Chemical', 'B-Disease', 'I-Disease', 'B-Gene', 'I-Gene',
                        'B-Entity', 'I-Entity'}
    self.rebucket()

    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='inputs')
    self.targets = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='targets')

    # placeholders to gather entity pairs
    self.gather_idx = tf.placeholder(dtype=tf.int32, shape=(None,None), name='gather_idx')
    self.scatter_idx = tf.placeholder(dtype=tf.int32, shape=(None,None), name='scatter_idx')
    self.scatter_shape = tf.placeholder(dtype=tf.int32, shape=(None), name='scatter_shape')
    self.rel_labels = tf.placeholder(dtype=tf.int32, shape=(None), name='rel_targets')
    self.rel_eps = tf.placeholder(dtype=tf.string, shape=(None), name='rel_eps')

    # placeholders to gather individual entities
    self.entity_gather_idx = tf.placeholder(dtype=tf.int32, shape=(None,None), name='entity_gather_idx')
    self.entity_scatter_idx = tf.placeholder(dtype=tf.int32, shape=(None,None), name='entity_scatter_idx')
    self.entity_scatter_shape = tf.placeholder(dtype=tf.int32, shape=(None), name='entity_scatter_shape')
    self.entity_labels = tf.placeholder(dtype=tf.int32, shape=(None), name='entity_targets')
    self.builder = builder()

  #=============================================================
  def file_iterator(self, filename):
    """"""

    with open(filename) as f:
      if self.lines_per_buffer > 0:
        buff = [[]]
        while True:
          line = f.readline()
          while line:
            line = line.strip().split()
            if line:
              buff[-1].append(line)
            else:
              if len(buff) < self.lines_per_buffer:
                if buff[-1]:
                  buff.append([])
              else:
                break
            line = f.readline()
          if not line:
            f.seek(0)
          else:
            buff = self._process_buff(buff)
            yield buff
            line = line.strip().split()
            if line:
              buff = [[line]]
            else:
              buff = [[]]
      else:
        buff = [[]]
        for line in f:
          line = line.strip().split()
          if line:
            buff[-1].append(line)
          else:
            if buff[-1]:
              buff.append([])
        if buff[-1] == []:
          buff.pop()
        buff = self._process_buff(buff)
        while True:
          yield buff

  #=============================================================
  def _process_buff(self, buff):
    """"""

    words, tags, rels, _ = self.vocabs
    for i, sent in enumerate(buff):
      for j, token in enumerate(sent):
        # print(sent)
        word, tag1, rel = token[words.conll_idx], token[tags.conll_idx], token[rels.conll_idx]
        tag2 = 'O'
        try:
          docid = int(token[3])
        except Exception as e:
          print(e.message)
          print(token)
          sys.exit(1)
        buff[i][j] = (word,) + words[word] + tags[tag1] + tags[tag2] + (docid,) + rels[rel]
    return buff

  #=============================================================
  def reset(self, sizes):
    """"""
    self._data = []
    self._targets = []
    self._metabucket.reset(sizes)
    return

  #=============================================================
  def rebucket(self):
    """"""

    buff = self._file_iterator.next()
    len_cntr = Counter()

    for sent in buff:
      len_cntr[len(sent)] += 1
    self.reset(KMeans(self.n_bkts, len_cntr).splits)

    for sent in buff:
      self._metabucket.add(sent)
    self._finalize()
    return

  #=============================================================
  def _finalize(self):
    """"""
    self._metabucket._finalize()
    return


  #================
  def get_relation_gather_indices_from_tokens(self, data, sents):
    # TODO make sure that none of the entity ids are being set to UNK for some reason
    # print(sents.shape)
    # print(data.shape)
    all_ep_idx = []
    for batch_idx in range(data.shape[0]):
      # each element of seq has same docid, take first
      docid = data[batch_idx, 0, 4]
      # convert the ner label and entity id back to str using reverse dict
      entity_idx = [(seq_idx,
                     self.reverse_vocabs[2][data[batch_idx, seq_idx, 5]],
                     self.reverse_vocabs[1][data[batch_idx, seq_idx, 2]]
                     )
                    for seq_idx in range(data.shape[1])
                    if self.reverse_vocabs[2][data[batch_idx, seq_idx, 5]] in self.keep_labels]
      # print(entity_idx)
      ep_idx = [(batch_idx, docid, idx1, idx2, '%s::%s' % (e1, e2))
                for ((idx1, label1, e1), (idx2, label2, e2))
                in itertools.permutations(entity_idx, 2)]
      all_ep_idx.append(ep_idx)
    # end batch iteration

    all_ep_idx = [x for sublist in all_ep_idx for x in sublist]
    gather_idx = [(b, e1, e2) for b, docid, e1, e2, ep in all_ep_idx]
    # map each ep-doc to an id
    ep_doc_id_map = {ep: i for i, ep in enumerate({'%s::%s' % (docid, _ep) for b, docid, e1, e2, _ep in all_ep_idx})}
    ep_doc_count = defaultdict(int)
    scatter_idx = []
    max_ep_count = 0
    for b, docid, e1, e2, ep in all_ep_idx:
      ep_doc = '%s::%s' % (docid, ep)
      i = ep_doc_id_map[ep_doc]
      j = ep_doc_count[ep_doc]
      ep_doc_count[ep_doc] += 1
      if ep_doc_count[ep_doc] > max_ep_count:
        max_ep_count = ep_doc_count[ep_doc]
      scatter_idx.append((i, j))

    labels = np.ones(len(ep_doc_count))
    rel_eps = [''] * len(ep_doc_count)
    for ep_doc, i in ep_doc_id_map.iteritems():
      label = self.rel_map[ep_doc] if ep_doc in self.rel_map else self.vocabs[3]['Null']
      labels[i] = label
      rel_eps[i] = ep_doc

    scatter_shape = [len(ep_doc_count), max_ep_count, len(self.vocabs[3])]
    # print(np.array(gather_idx).shape, np.array(scatter_idx).shape, np.array(labels).shape)
    # sys.exit(1)

    return gather_idx, scatter_idx, scatter_shape, labels, rel_eps


  #================
  def get_entity_gather_indices(self, data, sents):
    # TODO make sure that none of the entity ids are being set to UNK for some reason
    gather_idx = []
    scatter_idx = []
    entity_ids = []
    docs = []
    ner_labels = []
    current_entity_idx = -1
    current_entity_token = 0
    max_token_len = 0

    for batch_idx in range(data.shape[0]):
      # each element of seq has same docid, take first
      docid = data[batch_idx, 0, 4]
      for seq_idx in range(data.shape[1]):
        ner_label = self.reverse_vocabs[2][data[batch_idx, seq_idx, 5]]

        # start of new entity
        if ner_label.startswith('B'):
          entity_id = data[batch_idx, seq_idx, 2]
          entity_id_str = self.reverse_vocabs[1][entity_id]
          if entity_id_str != 'UNK' and entity_id_str != '-1':
            current_entity_token = 0
            current_entity_idx += 1
            gather_idx.append((batch_idx, seq_idx))
            scatter_idx.append((current_entity_idx, current_entity_token))
            # keep track of kb id for this entity
            entity_ids.append(entity_id)
            docs.append(docid)
            ner_labels.append(ner_label)
        elif ner_label.startswith('I') and entity_id_str != 'UNK' and entity_id_str != '-1':
          current_entity_token += 1
          gather_idx.append((batch_idx, seq_idx))
          scatter_idx.append((current_entity_idx, current_entity_token))
        else:
          max_token_len = max(max_token_len, current_entity_token)
          current_entity_token = 0

    # shape of scattered matrix is entity count x max entity len x embed dimension
    hidden_size = self.num_heads * self.head_size
    scatter_shape = (len(entity_ids), max_token_len, hidden_size)
    # print(np.array(gather_idx).shape, np.array(scatter_idx).shape, np.array(labels).shape)
    return gather_idx, scatter_idx, scatter_shape, entity_ids, ner_labels, docs


  #================
  def get_relation_gather_indices_from_entities(self, entity_ids, entity_ner, entity_docs):
    gather_idx = []
    scatter_idx = []
    max_ep_count = 0
    doc_id = {}
    ep_map = {}
    ep_count = defaultdict(int)
    entity_info = zip(entity_ids, entity_ner, entity_docs)

    for i, (e1, ner1, docid1) in enumerate(entity_info):
      e1_str = self.reverse_vocabs[1][e1]
      for j, (e2, ner2, docid2) in enumerate(entity_info):
        e2_str = self.reverse_vocabs[1][e2]
        if docid1 == docid2 and ner1 in self.keep_labels and ner2 in self.keep_labels:
          if docid1 not in doc_id:
            doc_id[docid1] = len(doc_id)
          doc_ep = '%s::%s::%s' % (docid1, e1_str, e2_str)
          if doc_ep not in ep_map:
            ep_map[doc_ep] = len(ep_map)
          # get cell i,j from pairwise-score matrix
          gather_idx.append((i, j))
          # scatter to new matrix of ep x scores
          x = (ep_map[doc_ep], ep_count[doc_ep])
          # print(x)
          scatter_idx.append(x)
          ep_count[doc_ep] += 1
          max_ep_count = max(max_ep_count, ep_count[doc_ep])

    labels = np.ones(len(ep_count))
    rel_eps = [''] * len(ep_count)
    for ep_doc, i in ep_map.iteritems():
      label = self.rel_map[ep_doc] if ep_doc in self.rel_map else self.vocabs[3]['Null']
      labels[i] = label
      rel_eps[i] = ep_doc
      # print(ep_doc)

    scatter_shape = [len(ep_count), max_ep_count, len(self.vocabs[3])]
    return gather_idx, scatter_idx, scatter_shape, labels, rel_eps




  #=============================================================
  def get_minibatches(self, batch_size, input_idxs, target_idxs, shuffle=True):
    """"""

    minibatches = []
    for bkt_idx, bucket in enumerate(self._metabucket):
      if batch_size == 0:
        n_splits = 1
      else:
        n_tokens = len(bucket) * bucket.size
        n_splits = max(n_tokens // batch_size, 1)
      if shuffle:
        range_func = np.random.permutation
      else:
        range_func = np.arange
      arr_sp = np.array_split(range_func(len(bucket)), n_splits)
      for bkt_mb in arr_sp:
        minibatches.append( (bkt_idx, bkt_mb) )
    if shuffle:
      np.random.shuffle(minibatches)
    for bkt_idx, bkt_mb in minibatches:
      feed_dict = {}
      data = self[bkt_idx].data[bkt_mb]
      sents = self[bkt_idx].sents[bkt_mb]

      maxlen = np.max(np.sum(np.greater(data[:,:,0], 0), axis=1))
      entity_gather, entity_scatter, entity_scatter_shape, entity_labels, entity_ner, entity_docs = \
        self.get_entity_gather_indices(data[:, :maxlen, :], sents)
      # rel_gather, rel_scatter, rel_scatter_shape, rel_labels, rel_eps = \
      #   self.get_relation_gather_indices_from_tokens(data[:, :maxlen, :], sents)
      rel_gather, rel_scatter, rel_scatter_shape, rel_labels, rel_eps = \
        self.get_relation_gather_indices_from_entities(entity_labels, entity_ner, entity_docs)

      feed_dict.update({
        self.inputs: data[:,:maxlen,input_idxs],
        self.targets: data[:,:maxlen,target_idxs],
        self.entity_gather_idx: entity_gather,
        self.entity_scatter_idx: entity_scatter,
        self.entity_scatter_shape: entity_scatter_shape,
        self.entity_labels: entity_labels,
      })
      if self.rel_loss_weight > 0:
        feed_dict.update({
        self.gather_idx: rel_gather,
        self.scatter_idx: rel_scatter,
        self.scatter_shape: rel_scatter_shape,
        self.rel_labels: rel_labels,
        self.rel_eps: rel_eps,
        })
      yield feed_dict, sents

  #=============================================================
  @property
  def n_bkts(self):
    if self._train:
      return super(Dataset, self).n_bkts
    else:
      return super(Dataset, self).n_valid_bkts

  #=============================================================
  def __getitem__(self, key):
    return self._metabucket[key]
  def __len__(self):
    return len(self._metabucket)
