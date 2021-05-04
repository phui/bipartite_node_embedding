#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple word2vec from scratch with Python
#   2018-FEB
#
#------------------------------------------------------------------------------+
#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import numpy as np
from collections import defaultdict

# SOFTMAX ACTIVATION FUNCTION
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Word2Vec():
    def __init__ (self, corpus, d=10, learning_rate=0.001, epochs=20,
                  window=1, negative_sample=1):
        self.d = d
        self.eta = learning_rate
        self.epochs = epochs
        self.window = window
        self.ng = negative_sample
        self.corpus = corpus

        # GENERATE WORD COUNTS
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(word_counts.keys()),reverse=False)
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))



    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return np.array(word_vec)


    # FORWARD PASS
    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = softmax(u)
        return y_c, h, u


    # BACKPROPAGATION
    def backprop(self, e, h, x, scale=1.0):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (scale * self.eta * dl_dw1)
        self.w2 = self.w2 - (scale * self.eta * dl_dw2)


    # TRAIN W2V model
    def train(self):
        # INITIALIZE WEIGHT MATRICES
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.d))     # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.d, self.v_count))     # context matrix

        # CYCLE THROUGH EACH EPOCH
        for ep in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for sentence in self.corpus:
                sent_len = len(sentence)

                # CYCLE THROUGH EACH WORD IN SENTENCE
                for i, word in enumerate(sentence):

                    w_t = self.word2onehot(sentence[i])

                    # CYCLE THROUGH CONTEXT WINDOW
                    w_c = []
                    for j in range(i-self.window, i+self.window+1):
                        if j!=i and j<=sent_len-1 and j>=0:
                            w_c.append(self.word2onehot(sentence[j]))

                    # GENERATE RANDOM NEGATIVE SAMPLES
                    negative_samples = np.random.choice(self.words_list, self.ng)
                    w_n = []
                    for word in negative_samples:
                        w_n.append(self.word2onehot(word))

                    # FORWARD PASS
                    y_pred, h, u = self.forward_pass(w_t)

                    # BACKPROPAGATION
                    # CONTEXT
                    EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                    self.backprop(EI, h, w_t)
                    # NEGATIVE SAMPLE
                    ET = np.sum([np.subtract(word, y_pred) for word in w_n], axis=0)
                    self.backprop(EI, h, w_t)

                    # CALCULATE LOSS
                    self.loss -= np.sum([u[list(word).index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                    self.loss += np.sum([u[list(word).index(1)] for word in w_n]) + len(w_n) * np.log(np.sum(np.exp(u)))

            print('EPOCH:', ep, 'LOSS:', self.loss)


    # input a word, returns a vector (if available)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w


    # input a vector, returns nearest word(s)
    def vec_sim(self, vec, top_n):

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda item : item[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)


    # input word, returns top [n] most similar words
    def word_sim(self, word, top_n):

        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda item: item[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)

