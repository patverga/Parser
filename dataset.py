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
    self.rebucket()
    
    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='inputs')
    self.targets = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='targets')
    self.gather_idx = tf.placeholder(dtype=tf.int32, shape=(None,None), name='gather_idx')
    self.scatter_idx = tf.placeholder(dtype=tf.int32, shape=(None,None), name='scatter_idx')
    self.scatter_shape = tf.placeholder(dtype=tf.int32, shape=(None), name='scatter_shape')
    self.rel_labels = tf.placeholder(dtype=tf.int32, shape=(None), name='rel_targets')
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
        except:
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
  def get_gather_indices(self, data, sents):
    # TODO make sure that none of the entity ids are being set to UNK for some reason
    keep_labels = {'B-Chemical', 'I-Chemical', 'B-Disease', 'I-Disease', 'B-Gene', 'I-Gene'}
    # print(sents.shape)
    # print(data.shape)
    all_ep_idx = []
    for batch_idx in range(data.shape[0]):
      # each element of seq has same docid, take first
      docid = data[batch_idx, 0, 4]
      for seq_idx in range(data.shape[1]):
        ner_label = data[batch_idx, seq_idx, 5]

      # convert the ner label and entity id back to str using reverse dict
      entity_idx = [(seq_idx,
                     self.reverse_vocabs[2][data[batch_idx, seq_idx, 5]],
                     self.reverse_vocabs[1][data[batch_idx, seq_idx, 2]]
                     )
                    for seq_idx in range(data.shape[1])
                    if self.reverse_vocabs[2][data[batch_idx, seq_idx, 5]] in keep_labels]
      # print(entity_idx)
      ep_idx = [(batch_idx, docid, idx1, idx2, '%s::%s' % (e1, e2))
                for ((idx1, label1, e1), (idx2, label2, e2))
                in itertools.permutations(entity_idx, 2)
                if e1 != e2 and e1 != 'UNK' and e1 != '-1' and e2 != 'UNK' and e2 != '-1']
      all_ep_idx.append(ep_idx)
    # end batch iteration

    all_ep_idx = [x for sublist in all_ep_idx for x in sublist]
    gather_idx = [(b, e1, e2) for b, docid, e1, e2, ep in all_ep_idx]
    ep_doc_id_map = {ep: i for i, ep in enumerate({'%s::%s' % (docid, _ep) for b, docid, e1, e2, _ep in all_ep_idx})}
    ep_doc_count = defaultdict(int)
    scatter_idx = []
    labels = []
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
    for ep_doc, i in ep_doc_id_map.iteritems():
      label = self.rel_map[ep_doc] if ep_doc in self.rel_map else self.vocabs[3]['Null']
      labels[i] = label

    scatter_shape = [len(ep_doc_count), max_ep_count, len(self.vocabs[3])]


    # print(len(gather_idx))
    # print(len(scatter_idx))
    # sys.exit(1)

    # print(gather_idx)
    # print(scatter_idx)
    return gather_idx, scatter_idx, scatter_shape, labels


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
      gather, scatter, scatter_shape, labels = self.get_gather_indices(data[:, :maxlen, :], sents)
      feed_dict.update({
        self.inputs: data[:,:maxlen,input_idxs],
        self.targets: data[:,:maxlen,target_idxs],
        self.gather_idx: gather,
        self.scatter_idx: scatter,
        self.scatter_shape: scatter_shape,
        self.rel_labels: labels
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
