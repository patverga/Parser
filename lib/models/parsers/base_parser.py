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

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models import NN

#***************************************************************
class BaseParser(NN):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    
    raise NotImplementedError
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep, n_cycles=-1, len_2_cycles=-1):
    """"""
    
    raise NotImplementedError
  
  #=============================================================
  def sanity_check(self, inputs, targets, predictions, vocabs, fileobject, feed_dict={}):
    """"""
    
    for tokens, golds, parse_preds, rel_preds in zip(inputs, targets, predictions[0], predictions[1]):
      for l, (token, gold, parse, rel) in enumerate(zip(tokens, golds, parse_preds, rel_preds)):
        if token[0] > 0:
          word = vocabs[0][token[0]]
          glove = vocabs[0].get_embed(token[1])
          tag = vocabs[1][token[2]]
          gold_tag = vocabs[1][gold[0]]
          pred_parse = parse
          pred_rel = vocabs[2][rel]
          gold_parse = gold[1]
          gold_rel = vocabs[2][gold[2]]
          fileobject.write('%d\t%s\t%s\t%s\t%s\t_\t%d\t%s\t%d\t%s\n' % (l, word, glove, tag, gold_tag, pred_parse, pred_rel, gold_parse, gold_rel))
      fileobject.write('\n')
    return
  
  #=============================================================
  def validate(self, mb_inputs, mb_targets, mb_probs, n_cycles, len_2_cycles):
    """"""
    
    sents = []
    mb_parse_probs, mb_rel_probs = mb_probs
    total_time = 0.0
    roots_lt_total = 0.
    roots_gt_total = 0.
    cycles_2_total = 0.
    cycles_n_total = 0.
    non_trees_total = 0.
    if np.all(n_cycles == -1):
        n_cycles = len_2_cycles = [-1] * len(mb_inputs)
    for inputs, targets, parse_probs, rel_probs, n_cycle, len_2_cycle in zip(mb_inputs, mb_targets, mb_parse_probs, mb_rel_probs, n_cycles, len_2_cycles):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      parse_preds, rel_preds, argmax_time, roots_lt, roots_gt = self.prob_argmax(parse_probs, rel_probs, tokens_to_keep, n_cycle, len_2_cycle)
      total_time += argmax_time
      roots_lt_total += roots_lt
      roots_gt_total += roots_gt
      cycles_2_total += len_2_cycle
      cycles_n_total += n_cycle
      if roots_lt or roots_gt or len_2_cycle or n_cycle:
        non_trees_total += 1.
      sent = -np.ones( (length, 9), dtype=int)
      tokens = np.arange(length)
      sent[:,0] = tokens
      sent[:,1:4] = inputs[tokens]
      sent[:,4] = targets[tokens,0]
      sent[:,5] = parse_preds[tokens]
      sent[:,6] = rel_preds[tokens]
      sent[:,7:] = targets[tokens, 1:]
      sents.append(sent)
    return sents, total_time, roots_lt_total, roots_gt_total, cycles_2_total, cycles_n_total, non_trees_total
  
  #=============================================================
  @staticmethod
  def evaluate(filename, punct=NN.PUNCT):
    """"""
    
    correct = {'UAS': [], 'LAS': []}
    with open(filename) as f:
      for line in f:
        line = line.strip().split('\t')
        if len(line) == 10 and line[4] not in punct:
          correct['UAS'].append(0)
          correct['LAS'].append(0)
          if line[6] == line[8]:
            correct['UAS'][-1] = 1
            if line[7] == line[9]:
              correct['LAS'][-1] = 1
    correct = {k:np.array(v) for k, v in correct.iteritems()}
    return 'UAS: %.2f    LAS: %.2f\n' % (np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100), correct

  # =============================================================
  @staticmethod
  def evaluate_by_len(filename, punct=NN.PUNCT):
    """"""
    # want UAS broken down by: sentence length, dep distance dep label
    # want LAS broken down by: dep label
    correct = {'UAS': [], 'LAS': []}
    correct_by_sent_len = {}
    correct_by_dep_len = {}
    correct_by_dep = {}
    uas_by_sent_len = {}
    uas_by_dep_len = {}
    las_by_dep = {}
    curr_sent_len = 0
    curr_sent_correct = 0
    curr_sent_pred = 0
    with open(filename) as f:
      for line in f:
        line = line.strip().split('\t')
        if len(line) == 10 and line[4] not in punct:
          correct['UAS'].append(0)
          correct['LAS'].append(0)
          if line[6] == line[8]:
            correct['UAS'][-1] = 1
            if line[7] == line[9]:
              correct['LAS'][-1] = 1
        # elif len(line) != 10:
        #   # update all the counts by sentence

    correct = {k: np.array(v) for k, v in correct.iteritems()}
    return 'UAS: %.2f    LAS: %.2f\n' % (np.mean(correct['UAS']) * 100, np.mean(correct['LAS']) * 100), correct

  #=============================================================
  @property
  def input_idxs(self):
    return (0, 1, 2)
  @property
  def target_idxs(self):
    return (3, 4, 5)
