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

    self.multi_penalties = {k: float(v) for k, v in map(lambda s: s.split(':'), self.multitask_penalties.split(';'))} if self.multitask_penalties else {}
    self.multi_layers = {k: set(map(int, v.split(','))) for k, v in map(lambda s: s.split(':'), self.multitask_layers.split(';'))} if self.multitask_layers else {}
    
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

    #### OLD ####
    # word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
    # tag_inputs = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    # if self.add_to_pretrained:
    #   word_inputs += pret_inputs
    # if self.word_l2_reg > 0:
    #   unk_mask = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,1], vocabs[0].UNK)),2)
    #   word_loss = self.word_l2_reg*tf.nn.l2_loss((word_inputs - pret_inputs) * unk_mask)
    # embed_inputs = self.embed_concat(word_inputs, tag_inputs)

    word_inputs, pret_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
    if self.add_to_pretrained:
      word_inputs += pret_inputs
    if self.word_l2_reg > 0:
      unk_mask = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,1], vocabs[0].UNK)), 2)
      word_loss = self.word_l2_reg*tf.nn.l2_loss((word_inputs - pret_inputs) * unk_mask)
    inputs_to_embed = [word_inputs]
    if self.add_pos_to_input:
      pos_inputs = vocabs[1].embedding_lookup(inputs[:, :, 2], moving_params=self.moving_params)
      inputs_to_embed.append(pos_inputs)
    if self.add_triggers_to_input:
      trigger_inputs = vocabs[4].embedding_lookup(inputs[:, :, 3], moving_params=self.moving_params)
      inputs_to_embed.append(trigger_inputs)

    embed_inputs = self.embed_concat(*inputs_to_embed)
    
    top_recur = embed_inputs

    attn_weights_by_layer = {}

    kernel = 3
    hidden_size = self.num_heads * self.head_size
    print("n_recur: ", self.n_recur)
    print("num heads: ", self.num_heads)
    print("cnn dim: ", self.cnn_dim)
    print("relu hidden size: ", self.relu_hidden_size)
    print("head size: ", self.head_size)

    print("cnn2d_layers: ", self.cnn2d_layers)
    print("cnn_dim_2d: ", self.cnn_dim_2d)

    print("multitask penalties: ", self.multi_penalties)
    print("multitask layers: ", self.multi_layers)

    trigger_indices = [i for s, i in vocabs[3].iteritems() if self.trigger_str in s]

    # do parse update if the random ~ unif(0,1) < proportion
    # otherwise, do srl update
    do_parse_update = tf.less_equal(np.random.rand(), self.parse_update_proportion)
    # do_arc_update = tf.not_equal(self.arc_loss_penalty, 0.)
    # do_rel_update = tf.not_equal(self.rel_loss_penalty, 0.)
    # do_parse_update = tf.logical_and(do_arc_update, do_rel_update)

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

    ###### stuff for multitask attention ######
    multitask_targets = {}

    mask = self.tokens_to_keep3D * tf.transpose(self.tokens_to_keep3D, [0, 2, 1])

    # compute targets adj matrix
    shape = tf.shape(targets[:, :, 1])
    batch_size = shape[0]
    bucket_size = shape[1]
    i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(bucket_size), indexing="ij")
    idx = tf.stack([i1, i2, targets[:, :, 1]], axis=-1)
    adj = tf.scatter_nd(idx, tf.ones([batch_size, bucket_size]), [batch_size, bucket_size, bucket_size])
    adj = adj * mask

    roots_mask = 1. - tf.expand_dims(tf.eye(bucket_size), 0)

    # create parents targets
    parents = targets[:, :, 1]
    multitask_targets['parents'] = parents

    # create children targets
    multitask_targets['children'] = tf.transpose(adj, [0, 2, 1]) * roots_mask

    # create grandparents targets
    i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(bucket_size), indexing="ij")
    idx = tf.reshape(tf.stack([i1, tf.nn.relu(parents)], axis=-1), [-1, 2])
    grandparents = tf.reshape(tf.gather_nd(parents, idx), [batch_size, bucket_size])
    multitask_targets['grandparents'] = grandparents
    grand_idx = tf.stack([i1, i2, grandparents], axis=-1)
    grand_adj = tf.scatter_nd(grand_idx, tf.ones([batch_size, bucket_size]), [batch_size, bucket_size, bucket_size])
    grand_adj = grand_adj * mask

    ###########################################

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
                if self.inject_manual_attn and moving_params is None and 'parents' in self.multi_layers.keys() and i in \
                    self.multi_layers['parents']:
                  manual_attn = adj
                  top_recur, attn_weights = self.transformer(top_recur, hidden_size, self.num_heads,
                                                             attn_dropout, relu_dropout, prepost_dropout,
                                                             self.relu_hidden_size,
                                                             self.info_func, reuse, manual_attn)
                elif self.inject_manual_attn and moving_params is None and 'grandparents' in self.multi_layers.keys() and i in \
                    self.multi_layers['grandparents']:
                  manual_attn = grand_adj
                  top_recur, attn_weights = self.transformer(top_recur, hidden_size, self.num_heads,
                                                             attn_dropout, relu_dropout, prepost_dropout,
                                                             self.relu_hidden_size,
                                                             self.info_func, reuse, manual_attn)
                else:
                  top_recur, attn_weights = self.transformer(top_recur, hidden_size, self.num_heads,
                                                             attn_dropout, relu_dropout, prepost_dropout,
                                                             self.relu_hidden_size, self.info_func, reuse)
                # head x batch x seq_len x seq_len
                attn_weights_by_layer[i] = tf.transpose(attn_weights, [1, 0, 2, 3])

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
      ######## do parse-specific stuff (arcs) ########
      with tf.variable_scope('MLP', reuse=reuse):
        dep_mlp, head_mlp = self.MLP(top_recur, self.class_mlp_size+self.attn_mlp_size, n_splits=2)
        dep_arc_mlp, dep_rel_mlp = dep_mlp[:,:,:self.attn_mlp_size], dep_mlp[:,:,self.attn_mlp_size:]
        head_arc_mlp, head_rel_mlp = head_mlp[:,:,:self.attn_mlp_size], head_mlp[:,:,self.attn_mlp_size:]

      with tf.variable_scope('Arcs', reuse=reuse):
        arc_logits = self.bilinear_classifier(dep_arc_mlp, head_arc_mlp)

        # arc_logits = tf.cond(tf.less_equal(tf.shape(tf.shape(arc_logits))[0], 2), lambda: tf.reshape(arc_logits, [tf.shape(arc_logits)[0], 1, 1]), lambda: arc_logits)

        arc_output = self.output_svd(arc_logits, targets[:, :, 1])
        if moving_params is None:
          predictions = targets[:, :, 1]
        else:
          predictions = arc_output['predictions']

    ######## do parse-specific stuff (rels) ########
    with tf.variable_scope('Rels', reuse=reuse):
      rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)

      # rel_logits = tf.Print(rel_logits, [embed_inputs], "embed_inputs", summarize=5000)
      #
      # rel_logits = tf.Print(rel_logits, [arc_logits], "arc_logits", summarize=5000)
      #
      # rel_logits = tf.Print(rel_logits, [predictions], "predictions", summarize=5000)
      #
      # rel_logits = tf.Print(rel_logits, [rel_logits], "rel_logits", summarize=5000)
      # rel_logits = tf.Print(rel_logits, [targets[:, :, 2]], "targets[:, :, 2]", summarize=5000)

      rel_output = self.output(rel_logits, targets[:, :, 2])
      rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)

    # def compute_rels_output():
    #   with tf.variable_scope('Rels', reuse=reuse):
    #     rel_logits, rel_logits_cond = self.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, len(vocabs[2]), predictions)
    #     rel_output = self.output(rel_logits, targets[:, :, 2])
    #     rel_output['probabilities'] = self.conditional_probabilities(rel_logits_cond)
    #     return rel_output
    # def dummy_compute_rels_output():

    multitask_losses = {}
    multitask_loss_sum = 0
    for l, attn_weights in attn_weights_by_layer.iteritems():
      # attn_weights is: head x batch x seq_len x seq_len
      # idx into attention heads
      attn_idx = 0
      if 'parents' in self.multi_layers.keys() and l in self.multi_layers['parents']:
        outputs = self.output_svd(attn_weights[attn_idx], multitask_targets['parents'])
        attn_idx += 1
        loss = self.multi_penalties['parents'] * outputs['loss']
        multitask_losses['parents%s' % l] = loss
        multitask_loss_sum += loss
      if 'grandparents' in self.multi_layers.keys() and l in self.multi_layers['grandparents']:
        outputs = self.output_svd(attn_weights[attn_idx], multitask_targets['grandparents'])
        attn_idx += 1
        loss = self.multi_penalties['grandparents'] * outputs['loss']
        multitask_losses['grandparents%s' % l] = loss
        multitask_loss_sum += loss
      if 'children' in self.multi_layers.keys() and l in self.multi_layers['children']:
        outputs = self.output_multi(attn_weights[attn_idx], multitask_targets['children'])
        attn_idx += 1
        loss = self.multi_penalties['children'] * outputs['loss']
        multitask_losses['children%s' % l] = loss
        multitask_loss_sum += loss


    ######## do SRL-specific stuff (rels) ########
    with tf.variable_scope('SRL-MLP', reuse=reuse):
      trigger_role_mlp = self.MLP(top_recur, self.trigger_mlp_size + self.role_mlp_size, n_splits=1)
      trigger_mlp, role_mlp = trigger_role_mlp[:,:,:self.trigger_mlp_size], trigger_role_mlp[:,:,self.trigger_mlp_size:]

    with tf.variable_scope('SRL-Triggers', reuse=reuse):
      trigger_classifier_mlp = self.MLP(top_recur, self.trigger_pred_mlp_size, n_splits=1)
      with tf.variable_scope('SRL-Triggers-Classifier', reuse=reuse):
        trigger_classifier = self.MLP(trigger_classifier_mlp, 2, n_splits=1)
      trigger_output = self.output_trigger(trigger_classifier, targets, trigger_indices)

    with tf.variable_scope('SRL-Arcs', reuse=reuse):
      srl_logits = self.bilinear_classifier_nary(trigger_mlp, role_mlp, num_srl_classes)
      if moving_params is None or self.add_triggers_to_input:
        trigger_predictions = trigger_output['targets']
      else:
        trigger_predictions = trigger_output['predictions']
      srl_output = self.output_srl(srl_logits, targets, trigger_indices, vocabs[3]["O"][0], transition_params if self.viterbi_train else None)

    trigger_loss = self.trigger_loss_penalty * trigger_output['loss']
    srl_loss = self.role_loss_penalty * srl_output['loss']
    arc_loss = self.arc_loss_penalty * arc_output['loss']
    rel_loss = self.rel_loss_penalty * rel_output['loss']

    if self.add_triggers_to_input:
      trigger_loss = tf.constant(0.)

    # if this is a parse update, then actual parse loss equal to sum of rel loss and arc loss
    actual_parse_loss = tf.cond(do_parse_update, lambda: tf.add(rel_loss, arc_loss), lambda: tf.constant(0.))

    # if this is a parse update and the parse proportion is not one, then no srl update. otherwise,
    # srl update equal to sum of srl_loss, trigger_loss
    actual_srl_loss = tf.cond(tf.logical_and(do_parse_update, tf.not_equal(self.parse_update_proportion, 1.0)), lambda: tf.constant(0.), lambda: tf.add(srl_loss, trigger_loss))

    output = {}

    output['multitask_losses'] = multitask_losses

    output['probabilities'] = tf.tuple([arc_output['probabilities'],
                                        rel_output['probabilities']])
    output['predictions'] = tf.stack([arc_output['predictions'],
                                      rel_output['predictions']])
    output['correct'] = arc_output['correct'] * rel_output['correct']
    output['tokens'] = arc_output['tokens']
    output['n_correct'] = tf.reduce_sum(output['correct'])
    output['n_tokens'] = self.n_tokens
    output['accuracy'] = output['n_correct'] / output['n_tokens']

    # output['loss'] = srl_loss + trigger_loss + actual_parse_loss
    # output['loss'] = actual_srl_loss + arc_loss + rel_loss
    output['loss'] = actual_srl_loss + actual_parse_loss + multitask_loss_sum

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

    output['rel_loss'] = rel_loss # rel_output['loss']
    output['log_loss'] = arc_loss # arc_output['log_loss']
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
    output['srl_trigger'] = trigger_predictions
    output['srl_trigger_targets'] = trigger_output['targets']
    output['trigger_loss'] = trigger_loss
    output['trigger_count'] = trigger_output['count']
    output['trigger_correct'] = trigger_output['correct']

    # transpose and softmax attn weights
    attn_weights_by_layer_softmaxed = {k: tf.transpose(tf.nn.softmax(v), [1, 0, 2, 3]) for k, v in
                                       attn_weights_by_layer.iteritems()}

    output['attn_weights'] = attn_weights_by_layer_softmaxed

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
