#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf

from lib.models import nn

from vocab import Vocab
from lib.models.parsers.base_parser import BaseParser

#***************************************************************
class Parser(BaseParser):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets

    num_srl_classes = len(vocabs[3])

    # need to add batch dim for batch size 1
    # inputs = tf.Print(inputs, [tf.shape(inputs), tf.shape(targets)], summarize=10)

    reuse = (moving_params is not None)
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    
    word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
    tag_inputs = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    if self.add_to_pretrained:
      word_inputs += pret_inputs
    if self.word_l2_reg > 0:
      unk_mask = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,1], vocabs[0].UNK)),2)
      word_loss = self.word_l2_reg*tf.nn.l2_loss((word_inputs - pret_inputs) * unk_mask)
    embed_inputs = self.embed_concat(word_inputs, tag_inputs)
    
    top_recur = embed_inputs

    kernel = 3
    hidden_size = self.num_heads * self.head_size
    print("n_recur: ", self.n_recur)
    print("num heads: ", self.num_heads)
    print("cnn dim: ", self.cnn_dim)
    print("relu hidden size: ", self.relu_hidden_size)
    print("head size: ", self.head_size)

    print("cnn2d_layers: ", self.cnn2d_layers)
    print("cnn_dim_2d: ", self.cnn_dim_2d)

    # todo these are actually wrong because of nesting
    bilou_constraints = np.zeros((num_srl_classes, num_srl_classes))
    for s_str, s_idx in vocabs[3].iteritems():
      for e_str, e_idx in vocabs[3].iteritems():
        s_bilou = s_str[0]
        e_bilou = e_str[0]
        s_type = s_str[2:]
        e_type = e_str[2:]
        if (s_bilou == 'L' or s_bilou == 'U' or s_bilou == 'O') and (e_bilou == 'O' or e_bilou == 'B' or e_bilou == 'U'):
          bilou_constraints[s_idx, e_idx] = 1.0
        elif (s_bilou == 'B' or s_bilou == 'I') and s_type == e_type and (e_bilou == 'I' or e_bilou == 'L'):
          bilou_constraints[s_idx, e_idx] = 1.0

    attn_dropout = 0.67
    prepost_dropout = 0.67
    relu_dropout = 0.67
    # if moving_params is not None:
    #   attn_dropout = 1.0
    #   prepost_dropout = 1.0
    #   relu_dropout = 1.0
    #   self.recur_keep_prob = 1.0

    with tf.variable_scope("crf", reuse=reuse):  # to share parameters, change scope here
      if self.viterbi_train:
        transition_params = tf.get_variable("transitions", [num_srl_classes, num_srl_classes], initializer=tf.constant_initializer(bilou_constraints))
      elif self.viterbi_decode:
        transition_params = tf.get_variable("transitions", [num_srl_classes, num_srl_classes], initializer=tf.constant_initializer(bilou_constraints), trainable=False)
      else:
        transition_params = None

    assert (self.cnn_layers != 0 and self.n_recur != 0) or self.num_blocks == 1, "num_blocks should be 1 if cnn_layers or n_recur is 0"
    assert self.dist_model == 'bilstm' or self.dist_model == 'transformer', 'Model must be either "transformer" or "bilstm"'

    for b in range(self.num_blocks):
      with tf.variable_scope("block%d" % b, reuse=reuse):  # to share parameters, change scope here
        # Project for CNN input
        if self.cnn_layers > 0:
          with tf.variable_scope('proj0', reuse=reuse):
            top_recur = self.MLP(top_recur, self.cnn_dim, n_splits=1)

        ####### 1D CNN ########
        with tf.variable_scope('CNN', reuse=reuse):
          for i in xrange(self.cnn_layers):
            with tf.variable_scope('layer%d' % i, reuse=reuse):
              if self.cnn_residual:
                top_recur += self.CNN(top_recur, 1, kernel, self.cnn_dim, self.recur_keep_prob, self.info_func)
                top_recur = nn.layer_norm(top_recur, reuse)
              else:
                top_recur = self.CNN(top_recur, 1, kernel, self.cnn_dim, self.recur_keep_prob, self.info_func)
          if self.cnn_residual and self.n_recur > 0:
            top_recur = nn.layer_norm(top_recur, reuse)

        # Project for Tranformer / residual LSTM input
        if self.n_recur > 0:
          if self.dist_model == "transformer":
            with tf.variable_scope('proj1', reuse=reuse):
              top_recur = self.MLP(top_recur, hidden_size, n_splits=1)
          if self.lstm_residual and self.dist_model == "bilstm":
            with tf.variable_scope('proj1', reuse=reuse):
              top_recur = self.MLP(top_recur, (2 if self.recur_bidir else 1) * self.recur_size, n_splits=1)

        ##### Transformer #######
        if self.dist_model == 'transformer':
          with tf.variable_scope('Transformer', reuse=reuse):
            top_recur = nn.add_timing_signal_1d(top_recur)
            for i in range(self.n_recur):
              with tf.variable_scope('layer%d' % i, reuse=reuse):
                top_recur = self.transformer(top_recur, hidden_size, self.num_heads,
                                             attn_dropout, relu_dropout, prepost_dropout, self.relu_hidden_size,
                                             self.info_func, reuse)
            # if normalization is done in layer_preprocess, then it should also be done
            # on the output, since the output can grow very large, being the sum of
            # a whole stack of unnormalized layer outputs.
            if self.n_recur > 0:
              top_recur = nn.layer_norm(top_recur, reuse)

        ##### BiLSTM #######
        if self.dist_model == 'bilstm':
          with tf.variable_scope("BiLSTM", reuse=reuse):
            for i in range(self.n_recur):
              with tf.variable_scope('layer%d' % i, reuse=reuse):
                if self.lstm_residual:
                  top_recur_curr, _ = self.RNN(top_recur)
                  top_recur += top_recur_curr
                  # top_recur = nn.layer_norm(top_recur, reuse)
                else:
                  top_recur, _ = self.RNN(top_recur)
            # if self.lstm_residual and self.n_recur > 0:
            #   top_recur = nn.layer_norm(top_recur, reuse)
        if self.num_blocks > 1:
          top_recur = nn.layer_norm(top_recur, reuse)

    ####### 2D CNN ########
    if self.cnn2d_layers > 0:
      with tf.variable_scope('proj2', reuse=reuse):
        top_recur_rows, top_recur_cols = self.MLP(top_recur, self.cnn_dim_2d//2, n_splits=2)
        # top_recur_rows, top_recur_cols = self.MLP(top_recur, self.cnn_dim // 4, n_splits=2)

      top_recur_rows = nn.add_timing_signal_1d(top_recur_rows)
      top_recur_cols = nn.add_timing_signal_1d(top_recur_cols)

      with tf.variable_scope('2d', reuse=reuse):
        # set up input (split -> 2d)
        input_shape = tf.shape(embed_inputs)
        bucket_size = input_shape[1]
        top_recur_rows = tf.tile(tf.expand_dims(top_recur_rows, 1), [1, bucket_size, 1, 1])
        top_recur_cols = tf.tile(tf.expand_dims(top_recur_cols, 2), [1, 1, bucket_size, 1])
        top_recur_2d = tf.concat([top_recur_cols, top_recur_rows], axis=-1)

        # apply num_convs 2d conv layers (residual)
        for i in xrange(self.cnn2d_layers):  # todo pass this in
          with tf.variable_scope('CNN%d' % i, reuse=reuse):
            top_recur_2d += self.CNN(top_recur_2d, kernel, kernel, self.cnn_dim_2d,  # todo pass this in
                                    self.recur_keep_prob if i < self.cnn2d_layers - 1 else 1.0,
                                    self.info_func if i < self.cnn2d_layers - 1 else tf.identity)
            top_recur_2d = nn.layer_norm(top_recur_2d, reuse)

        with tf.variable_scope('Arcs', reuse=reuse):
          arc_logits = self.MLP(top_recur_2d, 1, n_splits=1)
          arc_logits = tf.squeeze(arc_logits, axis=-1)
          arc_output = self.output_svd(arc_logits, targets[:, :, 1])
          if moving_params is None:
            predictions = targets[:, :, 1]
          else:
            predictions = arc_output['predictions']

        # Project each predicted (or gold) edge into head and dep rel representations
        with tf.variable_scope('MLP', reuse=reuse):
          # flat_labels = tf.reshape(predictions, [-1])
          original_shape = tf.shape(arc_logits)
          batch_size = original_shape[0]
          bucket_size = original_shape[1]
          # num_classes = len(vocabs[2])
          i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(bucket_size), indexing="ij")
          targ = i1 * bucket_size * bucket_size + i2 * bucket_size + predictions
          idx = tf.reshape(targ, [-1])
          conditioned = tf.gather(tf.reshape(top_recur_2d, [-1, self.cnn_dim_2d]), idx)
          conditioned = tf.reshape(conditioned, [batch_size, bucket_size, self.cnn_dim_2d])
          dep_rel_mlp, head_rel_mlp = self.MLP(conditioned, self.class_mlp_size + self.attn_mlp_size, n_splits=2)
    else:
      with tf.variable_scope('MLP', reuse=reuse):
        dep_mlp, head_mlp = self.MLP(top_recur, self.class_mlp_size+self.attn_mlp_size, n_splits=2)
        dep_arc_mlp, dep_rel_mlp = dep_mlp[:,:,:self.attn_mlp_size], dep_mlp[:,:,self.attn_mlp_size:]
        head_arc_mlp, head_rel_mlp = head_mlp[:,:,:self.attn_mlp_size], head_mlp[:,:,self.attn_mlp_size:]

      with tf.variable_scope('Arcs', reuse=reuse):
        arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)

        arc_logits = tf.cond(tf.equal(tf.shape(tf.shape(arc_logits))[0], 2), lambda: tf.expand_dims(arc_logits, 0), lambda: arc_logits)
        # arc_logits = tf.Print(arc_logits, [tf.shape(arc_logits), tf.shape(tf.shape(arc_logits))])

        arc_output = self.output_svd(arc_logits, targets[:,:,1])
        if moving_params is None:
          predictions = targets[:,:,1]
        else:
          predictions = arc_output['predictions']

    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
      rel_output = self.output(rel_logits, targets[:, :, 2])
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)

    # predict SRL
    # Have a bunch of cols of labels, some of which are empty
    # need to sample from non-empty ones
    # rel_output = tf.Print(rel_output, [tf.reduce_any(targets[:, :, 3:] != 3, axis=1)], "targets", summarize=500)
    # arc_output['loss'] = tf.Print(arc_output['loss'], [tf.shape(targets), tf.reduce_any(tf.not_equal(targets, 3), axis=1)], "targets", summarize=500)
    with tf.variable_scope('SRL-MLP', reuse=reuse):
      trigger_role_mlp = self.MLP(top_recur, self.trigger_mlp_size + self.role_mlp_size, n_splits=1)
      trigger_mlp, role_mlp = trigger_role_mlp[:,:,:self.trigger_mlp_size], trigger_role_mlp[:,:,self.trigger_mlp_size:]

    with tf.variable_scope('SRL-Triggers', reuse=reuse):
      trigger_classifier_mlp = self.MLP(top_recur, self.trigger_mlp_size, n_splits=1)
      with tf.variable_scope('SRL-Triggers-Classifier', reuse=reuse):
        trigger_classifier = self.MLP(trigger_classifier_mlp, 2, n_splits=1)
      trigger_output = self.output_trigger(trigger_classifier, targets, vocabs[3][self.trigger_str][0])
      trigger_loss = trigger_output['loss']

    with tf.variable_scope('SRL-Arcs', reuse=reuse):
      srl_logits = self.bilinear_classifier_nary(trigger_mlp, role_mlp, num_srl_classes)
      if moving_params is None:
        trigger_predictions = trigger_output['targets']
      else:
        trigger_predictions = trigger_output['predictions']
      srl_output = self.output_srl(srl_logits, targets, vocabs[3][self.trigger_str][0], vocabs[3]["O"][0], transition_params if self.viterbi_train else None)

    # todo weight?
    srl_loss = srl_output['loss']


    output = {}
    output['probabilities'] = tf.tuple([arc_output['probabilities'],
                                        rel_output['probabilities']])
    output['predictions'] = tf.stack([arc_output['predictions'],
                                      rel_output['predictions']])
    output['correct'] = arc_output['correct'] * rel_output['correct']
    output['tokens'] = arc_output['tokens']
    output['n_correct'] = tf.reduce_sum(output['correct'])
    output['n_tokens'] = self.n_tokens
    output['accuracy'] = output['n_correct'] / output['n_tokens']
    output['loss'] = srl_loss + trigger_loss #arc_output['loss'] + rel_output['loss'] + srl_loss
    if self.word_l2_reg > 0:
      output['loss'] += word_loss

    output['embed'] = embed_inputs
    output['recur'] = top_recur
    # output['dep_arc'] = dep_arc_mlp
    # output['head_dep'] = head_arc_mlp
    output['dep_rel'] = dep_rel_mlp
    output['head_rel'] = head_rel_mlp
    output['arc_logits'] = arc_logits
    output['rel_logits'] = rel_logits

    output['rel_loss'] = rel_output['loss']
    output['log_loss'] = arc_output['log_loss']
    output['2cycle_loss'] = arc_output['2cycle_loss']
    output['roots_loss'] = arc_output['roots_loss']
    output['svd_loss'] = arc_output['svd_loss']
    output['n_cycles'] = arc_output['n_cycles']
    output['len_2_cycles'] = arc_output['len_2_cycles']

    output['srl_loss'] = srl_loss
    output['srl_preds'] = srl_output['predictions']
    output['srl_probs'] = srl_output['probabilities']
    output['srl_logits'] = srl_output['logits']
    output['srl_correct'] = srl_output['correct']
    output['srl_count'] = srl_output['count']
    output['transition_params'] = transition_params if transition_params is not None else tf.constant(bilou_constraints)
    output['srl_trigger'] = trigger_output['targets'] #trigger_predictions
    output['trigger_loss'] = trigger_loss
    output['trigger_count'] = trigger_output['count']
    output['trigger_correct'] = trigger_output['correct']



    #### OLD: TRANSFORMER ####
    # top_recur = nn.add_timing_signal_1d(top_recur)
    #
    # for i in xrange(self.n_recur):
    #   # RNN:
    #   # with tf.variable_scope('RNN%d' % i, reuse=reuse):
    #   #   top_recur, _ = self.RNN(top_recur)
    #
    #   # Transformer:
    #   with tf.variable_scope('Transformer%d' % i, reuse=reuse):
    #     top_recur = self.transformer(top_recur, hidden_size, self.num_heads,
    #                                  attn_dropout, relu_dropout, prepost_dropout, self.relu_hidden_size,
    #                                  self.info_func, reuse)
    # # if normalization is done in layer_preprocess, then it shuold also be done
    # # on the output, since the output can grow very large, being the sum of
    # # a whole stack of unnormalized layer outputs.
    # top_recur = nn.layer_norm(top_recur, reuse)
    #
    # with tf.variable_scope('MLP', reuse=reuse):
    #   dep_mlp, head_mlp = self.MLP(top_recur, self.class_mlp_size+self.attn_mlp_size, n_splits=2)
    #   dep_arc_mlp, dep_rel_mlp = dep_mlp[:,:,:self.attn_mlp_size], dep_mlp[:,:,self.attn_mlp_size:]
    #   head_arc_mlp, head_rel_mlp = head_mlp[:,:,:self.attn_mlp_size], head_mlp[:,:,self.attn_mlp_size:]
    #
    # with tf.variable_scope('Arcs', reuse=reuse):
    #   arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)
    #   # arc_output = self.output(arc_logits, targets[:,:,1])
    #   arc_output = self.output_svd(arc_logits, targets[:,:,1])
    #   if moving_params is None:
    #     predictions = targets[:,:,1]
    #   else:
    #     predictions = arc_output['predictions']
    # with tf.variable_scope('Rels', reuse=reuse):
    #   rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
    #   rel_output = self.output(rel_logits, targets[:,:,2])
    #   rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
    #
    # output = {}
    # output['probabilities'] = tf.tuple([arc_output['probabilities'],
    #                                     rel_output['probabilities']])
    # output['predictions'] = tf.stack([arc_output['predictions'],
    #                                  rel_output['predictions']])
    # output['correct'] = arc_output['correct'] * rel_output['correct']
    # output['tokens'] = arc_output['tokens']
    # output['n_correct'] = tf.reduce_sum(output['correct'])
    # output['n_tokens'] = self.n_tokens
    # output['accuracy'] = output['n_correct'] / output['n_tokens']
    # output['loss'] = arc_output['loss'] + rel_output['loss']
    # if self.word_l2_reg > 0:
    #   output['loss'] += word_loss
    #
    # output['embed'] = embed_inputs
    # output['recur'] = top_recur
    # output['dep_arc'] = dep_arc_mlp
    # output['head_dep'] = head_arc_mlp
    # output['dep_rel'] = dep_rel_mlp
    # output['head_rel'] = head_rel_mlp
    # output['arc_logits'] = arc_logits
    # output['rel_logits'] = rel_logits
    #
    # output['rel_loss'] = rel_output['loss']
    # output['log_loss'] = arc_output['log_loss']
    # output['2cycle_loss'] = arc_output['2cycle_loss']
    # output['roots_loss'] = arc_output['roots_loss']
    # output['svd_loss'] = arc_output['svd_loss']
    return output
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep, n_cycles=-1, len_2_cycles=-1):
    """"""
    start_time = time.time()
    parse_preds, roots_lt, roots_gt = self.parse_argmax(parse_probs, tokens_to_keep, n_cycles, len_2_cycles)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    total_time = time.time() - start_time
    return parse_preds, rel_preds, total_time, roots_lt, roots_gt
