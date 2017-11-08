from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import string

plt.ioff()

data = np.load("attn_weights.npz")
lines = map(lambda x: x.split('\t'), open("sanitycheck.txt", 'r').readlines())
save_dir = "attn_plots3"
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

remove_padding = True
plot = False

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
        sentence_idx = batch_sum + b_i

        width = arrays.shape[-1]
        name = "sentence%d_layer%d" % (sentence_idx, layer)
        print("Batch: %d, sentence: %d, layer: %d" % (batch, sentence_idx, layer))

        sentence = sentences[sentence_idx]
        words = sentence[0]
        pred_deps = np.array(map(int, sentence[1]))
        pred_labels = sentence[2]
        gold_deps = np.array(map(int, sentence[3]))
        gold_labels = sentence[4]
        sent_len = len(words)
        text = words + [] if remove_padding else (['PAD'] * (width - sent_len))

        gold_deps_xy = np.array(list(enumerate(gold_deps)))
        pred_deps_xy = np.array(list(enumerate(pred_deps)))

        labels_incorrect = map(lambda x: x[0] != x[1], zip(pred_labels, gold_labels))
        incorrect_indices = np.where((pred_deps != gold_deps) | labels_incorrect)

        pred_deps_xy_incorrect = pred_deps_xy[incorrect_indices]
        pred_labels_incorrect = np.array(pred_labels)[incorrect_indices]

        if 'prep' in pred_labels_incorrect:
            print(' '.join(text))
            print(' '.join(pred_labels))
            print(' '.join(gold_labels))

        if plot:
            correct_dir = "correct" if len(incorrect_indices[0]) == 0 else "incorrect"

            fig.suptitle(name, fontsize=16)
            # For each attention head
            for arr, ax in zip(arrays, axes.flat):
                res = ax.imshow(arr[:sent_len, :sent_len], cmap=plt.cm.viridis, interpolation=None)
                ax.set_xticks(range(sent_len))
                ax.set_yticks(range(sent_len))
                ax.set_xticklabels(text, rotation=75, fontsize=2)
                ax.set_yticklabels(text, fontsize=2)

                map(lambda x: ax.text(x[0][1], x[0][0], x[1], ha="center", va="center", fontsize=1), zip(gold_deps_xy, gold_labels))
                map(lambda x: ax.text(x[0][1], x[0][0], x[1], ha="center", va="bottom", fontsize=1, color='red'), zip(pred_deps_xy_incorrect, pred_labels_incorrect))

            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, correct_dir, name + ".pdf"))
            map(lambda x: x.clear(), axes.flat)

    if layer == max_layer:
        batch_sum += batch_size
