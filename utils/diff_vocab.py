import zipfile
import collections
import numpy as np

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

from inputdata import Options, scorefunction
from model import skipgram

# This code can be used to read two vocabulary files and create a list of words that are in one vocab but not the other
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def read_vocabulary(path):
  vocab_file_name = os.path.join(path, 'vocab.txt')
  with open(vocab_file_name) as f:
    line = f.readline()
    vocab = []
    wordindex = dict()
    words = list()
    index = 0
    while line:
      line_contents = line.strip().split()
      if (len(line_contents) == 2):
        word = line_contents[0]
        words.append(word)
        index = line_contents[1]
        wordindex[word] = int(index)
      line = f.readline()
  f.close()
  reversed_dictionary = dict(zip(wordindex.values(), wordindex.keys()))
  return words, wordindex, reversed_dictionary

if __name__ == '__main__':
  path_cs_vocab = os.path.join(__location__, 'cs_model_plain')
  vocab_cs = read_vocabulary(path_cs_vocab)

  path_stat_vocab = os.path.join(__location__, 'stat_model_plain')
  vocab_stat = read_vocabulary(path_stat_vocab)
  list(set(vocab_cs[0]) - set(vocab_stat[0]))














