import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inputdata import Options, scorefunction
import os

class skipgram(nn.Module):
  def __init__(self, vocab_size, embedding_dim, pre_trained_model):
    super(skipgram, self).__init__()
    self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)   
    self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True) 
    self.embedding_dim = embedding_dim
    self.init_emb(pre_trained_model)
  def init_emb(self, pre_trained_model):
    initrange = 0.5 / self.embedding_dim
    if (pre_trained_model):
      u_embds = pre_trained_model['u_embeddings.weight']
      v_embds = pre_trained_model['v_embeddings.weight']
      del self.u_embeddings.weight # release memory associated with self.u_embeddings.weight
      self.u_embeddings.weight = nn.Parameter(u_embds)
      del self.v_embeddings.weight
      self.v_embeddings.weight = nn.Parameter(v_embds)
    else:
      self.u_embeddings.weight.data.uniform_(-initrange, initrange)
      self.v_embeddings.weight.data.uniform_(-0, 0)

  def forward(self, u_pos, v_pos, v_neg, batch_size):

    embed_u = self.u_embeddings(u_pos)
    embed_v = self.v_embeddings(v_pos)

    score  = torch.mul(embed_u, embed_v)
    score = torch.sum(score, dim=1)
    log_target = F.logsigmoid(score).squeeze()
    
    neg_embed_v = self.v_embeddings(v_neg)
    
    neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
    neg_score = torch.sum(neg_score, dim=1)
    sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

    loss = log_target + sum_log_sampled

    return -1*loss.sum()/batch_size
  def input_embeddings(self):
    return self.u_embeddings.weight.data.cpu().numpy()
  def save_embedding(self, file_name, id2word):
    embeds = self.u_embeddings.weight.data
    fo = open(file_name, 'w')
    for idx in range(len(embeds)):
      word = id2word(idx)
      embed = ' '.join(embeds[idx])
      fo.write(word+' '+embed+'\n')
