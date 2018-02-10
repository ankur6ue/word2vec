import collections
import numpy as np
import math
import os
import random
import string
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from os import listdir
from os.path import isfile, join
data_index = 0

class Options(object):
  def __init__(self, path, pre_trained_vocab_reverse, pre_trained_vocab, vocabulary_size):
    self.vocabulary_size = vocabulary_size
    self.save_path = path
    self.do_stemming = False
    # create English stop words list
    self.en_stop = get_stop_words('en')
    self.remove_stop_words = False
    # vocabulary contains a mapping between word and its index
    # vocabulary_reverse contains the opposite mapping - between the index and the word
    self.pre_trained_vocab = pre_trained_vocab
    self.pre_trained_vocab_reverse = pre_trained_vocab_reverse

    self.vocabulary = self.read_data(path)
    # There are two vocabularies:
    # 1. Current vocabulary: maps indicies to words in the current corpus. These indicies are index of the words
    # organized by their frequencies. For example, if "the" followed by "of" are the more frequent words, the vocabulary
    # contain 1: "the", 2: "of" and so on
    # 2. Pre-Trained Vocabularly: The user may also supply a vocabulary for a pre-trained model that will be used to initialize
    # the embedding matrices. When training with such a vocabulary, we must use the word indicies from this vocabularly while
    # still using word frequencies from the current corpus for frequent word subsampling etc.

    # data_or_ contains the current corpus where each word is replaced by its index in the current vocabulary
    # data_or contains the current corpus where each word is replaced by its index in the provided vocabulary. In case
    # no external vocabularly is provided, data_or = data_or_
    # dictionary: contains the reverse map., i.e from word to its index in the provided vocabularly. If no external vocabulary is provided
    # index in the current vocabulary is returned
    # vocab_words: contains the mapping between indicies (from external vocabulary) to words in the current corpus
    # vocab_words_: contains the mapping between indicies (from current vocabulary) to words in the current corpus. If
    # no external corpus is provided vocab_words = vocab_words_
    # It is possible that not every word in the corpus is in vocabulary. This is determined by vocabulary_size parameter.
    # Only the first vocabulary_size words are included in the vocabulary (from list of words sorted by frequency)
    data_or_, data_or, dictionary, self.count, self.vocab_words, self.vocab_words_ = self.build_dataset(self.vocabulary, pre_trained_vocab_reverse,
                                                              self.vocabulary_size)
    # performs frequent word subsampling - words are discarded in the proportion to their frequency, i.e., frequent words
    # are discarded more often. The intuition being that less frequent words are more meaningful. When an external vocabulary is
    # provided, we need both the indicies from the current vocabulary and the external vocabulary. train_data contains
    # selected words (that passed the subsampling) replaced by their indicies in the external vocabulary
    self.train_data, self.train_data_set = self.subsampling(data_or_, data_or)

    #self.save_vocab()
    #self.train_data = data_or

    # This function applies a transformation to the original word frequencies and then repeats the indicies according to
    # the transformed frequencies so that when this array containing the repeated frequencies is uniformly sampled, it
    # results in sampling according to the transformation. During this process, the correct word frequencies (stored in count)
    # must be used while the indicies that are repeated must come from the correct vocabulary - external or current vocabulary.
    self.sample_table = self.init_sample_table(dictionary)



  def read_data(self,path):
    data = []
    non_words = []
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    p_stemmer = PorterStemmer()
    # we want to ignore numerals, -, ?, \, {}, ()
    chars = set('0123456789-?\\(){}')
    MAX_NUM_FILES = 500000
    for i in range(0, min(MAX_NUM_FILES, len(onlyfiles))):
      if 'vocab' not in onlyfiles[i]: # don't need to read the vocabulary file which is saved in the same dir
        filename = path + '\\' + onlyfiles[i]
        with open(filename) as f:
          _data = f.read().split()
          for x in _data:
            if (x != 'eoood'):
              if ('$' in x):
                x = "LATEX"
                data.append(x)
              elif ('http' in x or 'www' in x):
                x = "LINK"
                data.append(x)
              elif not any((c in chars) for c in x): # only interested in words without numbers
                # strip punctuation marks - .,; etc
                x  = x.translate(str.maketrans('','',string.punctuation))
                x = x.lower()
                # stop word removal if requested
                if (self.remove_stop_words):
                  if x in self.en_stop:
                    continue
                if (self.do_stemming):
                  x = p_stemmer.stem(x)  # convert to lower case
                data.append(x)

    return data

  # this function takes the corpus and ranks the unique words in this corpus in order of frequency. A vocabulary and
  # reverse vocabulary is constructed that contains a mapping between words and the indicies in the frequency table.
  # If the value of n_words is lower than the number of unique words in the corpus, not all words in the corpus will be
  # part of the vocabulary. A pre-trained vocabulary can also be provided. If so, the existing indicies from this vocabulary
  # will be used
  def build_dataset(self,words, pre_trained_vocab_rev, n_words):

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary_ = {}
    dictionary = {}
    for word, _ in count:
      dictionary_[word] = len(dictionary_)
    if (pre_trained_vocab_rev):
      dictionary = pre_trained_vocab_rev
    else:
      dictionary = dictionary_
    data = list()
    data_ = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
        index_ = dictionary_[word]
      else:
        index = 0  # dictionary['UNK']
        index_ = 0
        unk_count += 1
      data.append(index)
      data_.append(index_)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    reversed_dictionary_ = dict(zip(dictionary_.values(), dictionary_.keys()))
    return data_, data, dictionary, count, reversed_dictionary, reversed_dictionary_

  def save_vocab(self):
    with open(os.path.join(self.save_path, "vocab.txt"), "w") as f:
      for idx in self.train_data_set:
        word = self.vocab_words[idx]
        f.write("%s %d\n" % (word, idx))

  # This is used for negative word sampling. A transformation is applied to the original word frequencies and the index of each word
  # is repeated in proportion to its frequency. When this array of indicies is randomly sampled, the probablity distribution of the
  # sampled words will be the transformed frequency distribution.
  def init_sample_table(self, dictionary):
    count = [ele[1] for ele in self.count]
    pow_frequency = np.array(count)**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency/ power
    table_size = 1e8
    count = np.round(ratio*table_size)
    sample_table = []
    for idx, x in enumerate(count):
      word = self.vocab_words_[idx]
      # index into the original dictionary
      # check if the word is in the dictionary - if an external dictionary is provided, the word
      # may not be in it. A word may also be missing from a dictionary constructed from the current corpus
      # if a low value of vocabulary_size parameter (see build_dataset function) is chosen and not all words in the current
      # corpus are added to the dictionary.
      if (word in dictionary):
        idx_ = dictionary[word]
        sample_table += [idx_]*int(x)
    return np.array(sample_table)

  def weight_table(self):
    count = [ele[1] for ele in self.count]
    pow_frequency = np.array(count)**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency/ power
    return np.array(ratio)

  # frequent word subsampling. For a simple explanation, see
  # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
  # basically keep words that have a low frequency using a probability distribution
  # data_ contains the original corpus where the words are replaced by indicies into the current vocabulary
  # data contains the original corpus where the words are replaced by indicies into the external vocabulary. If no external
  # vocabulary is provided, the two are identical
  # the function returns sampled data where the word indices are picked from the external vocabulary. If no external vocabulary
  # is provided, indicies into the current vocabulary are returned.
  def subsampling(self, data_, data):
    count = [ele[1] for ele in self.count]
    frequency = np.array(count)/sum(count)
    P = dict()
    for idx, x in enumerate(frequency):
      y = (math.sqrt(x/0.001)+1)*0.001/x
      P[idx] = y
    subsampled_data = list()
    subsampled_data_set = set()
    for idx, word_idx in enumerate(data):
      # word_idx ordered by frequency: needed for subsampling
      word_idx_ = data_[idx]
      if random.random()<P[word_idx_]:
        subsampled_data.append(word_idx)
        subsampled_data_set.add(word_idx)
    return subsampled_data, subsampled_data_set



  def generate_batch2(self, skip_window, batch_size):
    global data_index
    data = self.train_data
    batch = np.ndarray(shape=(batch_size), dtype=np.int64)
    labels = np.ndarray(shape=(batch_size, 2 * skip_window), dtype=np.int64)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    
    if data_index + span > len(data):
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size):
      batch[i] = buffer[skip_window]
      targets = [x for x in range(skip_window)]+[x for x in range(skip_window+1,span)]
      for idj, j in enumerate(targets):
        labels[i,idj] = buffer[j]
      if data_index == len(data):
        buffer.extend(data[:span])
        data_index = span
        self.process = False
      else:
        buffer.append(data[data_index])
        data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

  def generate_batch(self, window_size, batch_size, count):
    data = self.train_data
    global data_index
    span = 2 * window_size + 1
    context = np.ndarray(shape=(batch_size,2 * window_size), dtype=np.int64)
    labels = np.ndarray(shape=(batch_size), dtype=np.int64)
    pos_pair = []
    
    if data_index + span > len(data):
      data_index = 0
      self.process = False  
    buffer = data[data_index:data_index + span]
    pos_u = []
    pos_v = []

    for i in range(batch_size):
      data_index += 1
      context[i,:] = buffer[:window_size]+buffer[window_size+1:]
      labels[i] = buffer[window_size]
      if data_index + span > len(data):
        buffer[:] = data[:span]
        data_index = 0
        self.process = False
      else:
        buffer = data[data_index:data_index + span]

      for j in range(span-1):
        pos_u.append(labels[i])
        pos_v.append(context[i,j])
    neg_v = np.random.choice(self.sample_table, size=(batch_size*2*window_size,count))
    return np.array(pos_u), np.array(pos_v), neg_v




import json, csv
from scipy.stats import spearmanr
import math
def cosine_similarity(v1,v2):
  "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
  sumxx, sumxy, sumyy = 0, 0, 0
  for i in range(len(v1)):
    x = v1[i]; y = v2[i]
    sumxx += x*x
    sumyy += y*y
    sumxy += x*y
  return sumxy/math.sqrt(sumxx*sumyy)

def scorefunction(embed):
  f = open('./tmp/vocab.txt')
  line = f.readline()
  vocab = []
  wordindex = dict()
  index = 0
  while line:
    word = line.strip().split()[0]
    wordindex[word] = index
    index = index +1
    line = f.readline()
  f.close()
  ze = []
  with open('./wordsim353/combined.csv') as csvfile:
    filein = csv.reader(csvfile)
    index = 0
    consim = []
    humansim = []
    for eles in filein:
      if index==0:
        index = 1
        continue
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue

      word1 = int(wordindex[eles[0]])
      word2 = int(wordindex[eles[1]])
      humansim.append(float(eles[2]))


      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      score = cosine_similarity(value1, value2)
      consim.append(score)


  cor1, pvalue1 = spearmanr(humansim, consim)

  if 1==1:
    lines = open('./rw/rw.txt','r').readlines()
    index = 0
    consim = []
    humansim = []
    for line in lines:
      eles = line.strip().split()
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue
      word1 = int(wordindex[eles[0]])
      word2 = int(wordindex[eles[1]])
      humansim.append(float(eles[2]))
      
      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      score = cosine_similarity(value1, value2)
      consim.append(score)


  cor2, pvalue2 = spearmanr(humansim, consim)


  return cor1,cor2
