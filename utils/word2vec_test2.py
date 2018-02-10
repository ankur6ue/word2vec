import zipfile
import collections
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time
from os.path import isfile, join
from inputdata import Options, scorefunction
from model import skipgram
from sklearn.manifold import TSNE

# This code loads the word2vec model and corresponding vocabulary and computes the difference of the embedding vectors
# for words that are in the CS vocab but not in the stat vocab and computes the norm of the differences. This is to confirm
# our intuition that words more salient to
# stat (the smaller training set) should "move" more than words that are more general. This is expected as the smaller,
# more specialized training set provides additional context information not present in the general training set that
# was used to initialize the embedding matrices for training on the smaller set.

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def read_vocabulary(path):
  vocab_file_name = os.path.join(path, 'vocab.txt')
  with open(vocab_file_name) as f:
    line = f.readline()
    vocab = []
    wordindex = dict()
    index = 0
    while line:
      line_contents = line.strip().split()
      if (len(line_contents) == 2):
        word = line_contents[0]
        index = line_contents[1]
        wordindex[word] = int(index)
      line = f.readline()
  f.close()
  reversed_dictionary = dict(zip(wordindex.values(), wordindex.keys()))
  return wordindex, reversed_dictionary

if __name__ == '__main__':

    # Path for loading the cs model (model trained on the CS arxiv abstract data) vocab. Vocab is in the same directory
    # as the model
    path_cs_vocab = os.path.join(__location__, 'cs_model_plain')
    vocab_cs = read_vocabulary(path_cs_vocab)

    # load the vocab for the stat model - the model trained on the stat.ML abstract data
    path_stat_vocab = os.path.join(__location__, 'stat_model_plain')
    vocab_stat = read_vocabulary(path_stat_vocab)

    # words that are in the cs vocab but not in the stat vocab
    diff_list = list(set(vocab_cs[0]) - set(vocab_stat[0]))

    # load the model initialized using the cs model and then trained on stat data
    model_path = join(__location__, 'stat_model_cs_init')
    model = torch.load(join(model_path, 'skipgram.epoch4.batch73671'))
    all_embds_stat = model['u_embeddings.weight']

    # load the "large model", i.e., the model trained on the CS abstract data
    model_path = join(__location__, 'cs_model_plain')
    model = torch.load(join(model_path, 'skipgram.epoch9.batch854390'))
    all_embds_cs = model['u_embeddings.weight']



    distances = list()
    for word in vocab_cs[0]:
        if word not in diff_list:
            word_idx = vocab_cs[0][word]
            e1 = all_embds_cs[word_idx, :]
            e2 = all_embds_stat[word_idx, :]
            dist = torch.norm(e1 - e2)
            distances.append(tuple((dist, word)))







