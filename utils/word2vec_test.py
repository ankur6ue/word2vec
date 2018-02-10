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
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats

# This code reads the vocabulary and model files and looks up the closest embedding vectors for a target word. It then
# computes lower dimensional representations for the embedding vectors using PCA and t-SNE and plots them in 2D an 3D. It also
# shows how to calculate kendall-tau distances for the ranked lists produced by PCA and t-SNE against the gold standard rankings
# produced by the original embedding vectors.

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

# Scale and visualize the embedding vectors
def plot_embedding_2d(X, labels, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    plt.figure()
    ax = plt.subplot(111)
    plt.xlim(x_min[0], x_max[0])
    plt.ylim(x_min[1], x_max[1])
    # Draw the first label in bold red
    plt.text(X[0, 0], X[0, 1], labels[0], color='red', fontdict={'weight': 'bold', 'size': 9})
    for i in range(1, len(X)):
        plt.text(X[i, 0], X[i, 1], labels[i])

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

# Scale and visualize the embedding vectors in 3D
def plot_embedding_3d(X, labels, title=None):
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(x_min[0], x_max[0])
    ax.set_ylim3d(x_min[1], x_max[1])
    ax.set_zlim3d(x_min[2], x_max[2])
    # Draw the first label in bold red
    ax.text(X[0, 0], X[0, 1], X[0, 2], labels[0], color='red', fontdict={'weight': 'bold', 'size': 9})
    # Draw the remaining labels in normal color
    for i in range(1, len(X)):
        ax.text(X[i, 0], X[i, 1], X[i, 2], labels[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    if title is not None:
        ax.set_title(title)

# Calculate Kendall Tau Distance
def calc_kendall_tau(x1, x2, numElem=30):
    mat = np.asmatrix(x1[1:, :])
    vec = np.asmatrix(x1[0, :])
    dist = np.squeeze(np.asarray(np.dot(vec, np.transpose(mat))))
    sort_indicies = np.argsort(-dist)
    tau, pval = stats.kendalltau(sort_indicies[0:numElem], x2[0:numElem])
    return tau, pval

# Plot kendall-tau against variation explained by PCA. The two plots should have a similar trend line, although
# Kendall-tau plot will be a lot noisier.
def plot_var_tau():
    tau_pca_arr = list()
    var_explained_arr = list()
    for dim in range(2, 60):
        pca = PCA(n_components=dim)
        embds_pca = pca.fit_transform(top_embds)
        tau_pca, pval = calc_kendall_tau(embds_pca, range(0, 30), 30)
        tau_pca_arr.append(tau_pca)
        var_explained_arr.append(np.sum(pca.explained_variance_ratio_))
    fig = plt.figure()
    plt.plot(var_explained_arr)
    plt.plot(tau_pca_arr)

if __name__ == '__main__':

    # Path for loading the stat model. Corresponding vocabulary file is in the same directory
    # as the model
    model_path = join(__location__, 'stat_model_plain')
    #model_path = join(__location__, 'stat_model_cs_init')
    #model = torch.load(join(model_path, 'skipgram.epoch4.batch73671'))
    model = torch.load(join(model_path, 'skipgram.epoch8.batch73626'))
    all_embds = model['u_embeddings.weight']
    # load vocabulary
    words, indx_to_words = read_vocabulary(model_path)
    search_words = ["embedding", "learning", "optimization", "loss"]

    for search_word in search_words:
        # find the index of the word
        search_indx = words[search_word]
        # find the correspondong embedding vector (target embedding vector)
        search_embd = all_embds[search_indx,:]
        # rank all other words in order of embedding distance
        # the unsqueeze is to add a singleton dimension so the dot product (mm) has the correct dimensions
        distances = torch.mm(all_embds, torch.unsqueeze(search_embd, 1))
        distance_np = distances.cpu().numpy().squeeze()
        sort_indicies = np.argsort(-distance_np)
        top_50_scores = distance_np[sort_indicies[:50]]
        top_scores = distance_np[sort_indicies[1:]]
        plt.plot(top_scores)
        print(search_word)
        for i in range(1, 10):
            print(indx_to_words[sort_indicies[i]])

        # make a labels array for plotting purposes
        labels = list()
        for i in range(0, 100):
            labels.append(indx_to_words[sort_indicies[i]])

        # Since t-SNE is a bit slow, only pass the top 1000 vectors to t-SNE
        top_embds = all_embds[sort_indicies[:1000], :].cpu().numpy()
        pca = PCA(n_components=50)
        # Perform principal component analysis. The number of dimensions desired can be changed easily
        embds_pca = pca.fit_transform(top_embds)
        # Kendall tau distance. The first two arguments are the two lists to be compared, last argument is the number of
        # elements to be considered. The second argument is the "gold standard" list, the original rankings produced by the
        # embedding vectors
        tau_pca, pval = calc_kendall_tau(embds_pca, range(0, 30), 30)
        #plot_embedding_2d(embds_pca[0:25], labels, "2D PCA Plot for target word \"embedding\"")
        t_sne = TSNE(n_components=3)
        embds_t_sne = t_sne.fit_transform(top_embds)

        tau_sne, pval = calc_kendall_tau(embds_t_sne, range(0, 30), 30)
        #plot_embedding_2d(t_sne[0:30], labels, "2D t-SNE Plot")
    print('\n')











