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

# This code derives from the following repo:
# https://github.com/fanglanting/skip-gram-pytorch
# I have added some comments, additional text pre-processing options and added the ability to train a word2vec model
# from an existing model and vocabulary.
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

class word2vec:
  def __init__(self, inputfile, pre_trained_vocab_reverse = {}, pre_trained_vocab = {}, vocabulary_size=300000, embedding_dim=200, epoch_num=5, batch_size=16, windows_size=5,neg_sample_num=10):
    self.op = Options(inputfile, pre_trained_vocab_reverse, pre_trained_vocab, vocabulary_size)
    self.embedding_dim = embedding_dim
    self.windows_size = windows_size
    self.vocabulary_size = len(self.op.vocab_words)
    self.batch_size = batch_size
    self.epoch_num = epoch_num
    self.neg_sample_num = neg_sample_num


  def train(self, pre_trained_model):

    model = skipgram(self.vocabulary_size, self.embedding_dim, pre_trained_model)
    if torch.cuda.is_available():
      model.cuda()
    optimizer = optim.SGD(model.parameters(),lr=0.2)
    loss_history = list()
    for epoch in range(self.epoch_num):
      start = time.time()     
      self.op.process = True
      batch_num = 0
      batch_new = 0

      while self.op.process:
        pos_u, pos_v, neg_v = self.op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_num)

        pos_u = Variable(torch.LongTensor(pos_u))
        pos_v = Variable(torch.LongTensor(pos_v))
        neg_v = Variable(torch.LongTensor(neg_v))


        if torch.cuda.is_available():
          pos_u = pos_u.cuda()
          pos_v = pos_v.cuda()
          neg_v = neg_v.cuda()

        optimizer.zero_grad()
        loss = model(pos_u, pos_v, neg_v,self.batch_size)

        loss.backward()
   
        optimizer.step()

        if batch_num % 10 == 0:
          loss_history.append(loss.data[0])

        if batch_num % 2000 == 0:
          end = time.time()
          word_embeddings = model.input_embeddings()
          # sp1, sp2 = scorefunction(word_embeddings)
          print('epoch,batch=%2d %5d:  pair/sec = %4.2f loss=%4.3f\r', \
                epoch, batch_num, (batch_num - batch_new) * self.batch_size / (end - start), loss.data[0])
          batch_new = batch_num
          start = time.time()
        batch_num = batch_num + 1
      print()
      torch.save(model.state_dict(), __location__ + '/skipgram.epoch{}.batch{}'.format(epoch, batch_num))

    plt.plot(loss_history[::100])
    plt.ylabel('loss (stat.ML)')
    plt.show()
    print("Optimization Finished!")

  
if __name__ == '__main__':
  __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

  path = os.path.join(__location__, '..\\arxivscraper-master\\arxiv\\abstracts\\stat.ML')
  model_path = os.path.join(__location__, '..\\skip-gram-pytorch-master\\cs_model_plain')
  # if you wish to use a pre-trained model to initialize the embedding matrices, use the code below. You'll need to
  # provide both a vocabulary and a model.
  #pre_trained_model = torch.load(os.path.join(model_path, 'skipgram.epoch9.batch854390'))
  #pre_trained_vocab_reverse, pre_trained_vocab = read_vocabulary(model_path)
  #wc= word2vec(path, pre_trained_vocab_reverse, pre_trained_vocab)
  #wc.train(pre_trained_model)

  # if you are training from scratch, leave the vocabulary and pre-trained model blank.
  wc = word2vec(path,{}, {})
  wc.train({})















