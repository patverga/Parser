from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import string

data = np.load("attn_weights.npz")
lines = map(lambda x: x.split('\t'), open("sanitycheck.txt", 'r').readlines())
save_dir = "attn_plots"
sentences = []
current_sent = []
for line in lines:
    if len(line) < 10:
        sentences.append(map(list, zip(*current_sent)))
        current_sent = []
    else:
        current_sent.append(map(string.strip, (line[1], line[6], line[7], line[8], line[9])))
sentences.append(map(list, zip(*current_sent)))
max_layer = 3

batch_sum = 0
fig, axes = plt.subplots(nrows=2, ncols=4)
# For each batch+layer
for arr_name in sorted(data.files):
    print("Processing %s" % arr_name)
    batch_size = data[arr_name].shape[0]
    batch = int(arr_name[1])
    layer = int(arr_name.split(':')[1][-1])
    idx_in_batch = 0
    # For each element in the batch (one layer)
    # if layer == max_layer and batch > 0:
    for b_i, arrays in enumerate(data[arr_name]):
        plt.clf()

        sentence_idx = batch_sum + b_i

        width = arrays.shape[-1]
        name = "sentence%d_layer%d" % (sentence_idx, layer)
        print("Batch: %d, sentence: %d, layer: %d" % (batch, sentence_idx, layer))

        sentence = sentences[sentence_idx]
        words = sentence[0]
        pred_deps = map(int, sentence[1])
        pred_labels = sentence[2]
        gold_deps = map(int, sentence[3])
        gold_labels = sentence[4]
        text = words + ['PAD'] * (width - len(words))
        print(' '.join(text))

        gold_deps_xy = list(enumerate(gold_deps))
        pred_deps_xy = list(enumerate(pred_deps))

        correct_dir = "correct" if gold_deps == pred_deps and gold_labels == pred_labels else "incorrect"

        fig.suptitle(name, fontsize=16)
        # For each attention head
        for arr, ax in zip(arrays, axes.flat):
            res = ax.imshow(arr, cmap=plt.cm.viridis, interpolation='nearest')
            ax.set_xticks(range(len(text)))
            ax.set_yticks(range(len(text)))
            ax.set_xticklabels(text, rotation=75, fontsize=4)
            ax.set_yticklabels(text, fontsize=4)

            map(lambda x: ax.text(x[0][1], x[0][0], x[1], ha="center", va="center", fontsize=2), zip(gold_deps_xy, gold_labels))
            map(lambda x: ax.text(x[0][1], x[0][0], x[1], ha="center", va="center", fontsize=2, color='red'), zip(pred_deps_xy, pred_labels))

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, correct_dir, name + ".pdf"))
    if layer == max_layer:
        batch_sum += batch_size

# plt.show()