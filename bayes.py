#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time
import math
np.set_printoptions(threshold=np.inf)


class MyBayesClassifier():
    # For graduate and undergraduate students to implement Bernoulli Bayes
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    def train(self, X, y):

        class_counts = []
        for cl in y:
            if cl not in self._Ncls:
                self._Ncls.append(cl)
                # initialize the class counts with 0
                class_counts.append(0)

        # Calculate P(y) for each class y
        self._class_prob = np.zeros(len(self._Ncls))
        for class_index, cl in enumerate(self._Ncls):
            counter = 0
            for y_val in y:
                if (y_val == cl):
                    counter += 1
            self._class_prob[class_index] = ((float(counter) + self._smooth)/(len(y) + self._smooth*2))
            class_counts[class_index] = float(counter)

        #Calculate P(xi=0|y) and for each y and every feature xi
        self._Nfeat = np.zeros((len(self._Ncls), len(X[0])));
        self._feat_prob = np.zeros((len(self._Ncls), len(X[0])));

        for class_index, cl in enumerate(self._Ncls):
            for x_idx, x_val in enumerate(X[0]):
                self._feat_prob[class_index][x_idx] = (class_counts[class_index] + (self._smooth*2));
 
        for class_index, class_val in enumerate(self._Ncls): #looping over classes

            for y_idx, y_val in enumerate(y): #looping over the documents classifications
                if (y_val == class_val):
                    #add up number of '1's in the Xs corresponding to that y index
                    for x_idx, xi in enumerate(X[y_idx]): #looping over a documents words
                        if(xi > 0):
                            self._Nfeat[class_index][x_idx] += 1

        self._Nfeat = np.add(self._Nfeat, self._smooth)
        self._feat_prob = np.true_divide(self._Nfeat, self._feat_prob)
        return 


    def predict(self, X):
        # Calculate  P(0 | X), P(1| X), P (2 | X), etc
        # Given that the class=0, whats the prob that xi=0, xi=1?
        # Then total = add all of them together, divide: each by total to get percentages, choose highest

        x_prob = np.zeros((len(self._Ncls), len(X)))
        x_percentages = np.zeros((len(self._Ncls), len(X)))

        for class_idx, cl in enumerate(self._Ncls):
            for xlist_idx, x_list in enumerate(X): #looping over documents in document list
                xi_probs = math.log(self._class_prob[class_idx])
                for xval_idx, x_val in enumerate(x_list):   #looping over words in document
                    if(x_val > 0):
                        xi_probs += math.log(self._feat_prob[class_idx][xval_idx]) #P(xi=1 | Class=cl)
                    else: 
                        xi_probs += math.log(1 - self._feat_prob[class_idx][xval_idx]) #P(xi=0 | Class=cl)
                x_prob[class_idx][xlist_idx] = xi_probs

        class_predictions = np.zeros(len(X))
        for xlist_idx, x_list in enumerate(X):
            highest_number = x_prob[0][xlist_idx]
            highest_number_class = self._Ncls[0]
            for class_idx, cl in enumerate(self._Ncls):
                new_number = x_prob[class_idx][xlist_idx]
                if (new_number > highest_number):
                    highest_number = new_number
                    highest_number_class = cl
            class_predictions[xlist_idx] = highest_number_class

        return class_predictions


class MyMultinomialBayesClassifier():
    # For graduate students only
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    # Train the classifier using features in X and class labels in Y
    def train(self, X, y):
        # Your code goes here.
        return

    # should return an array of predictions, one for each row in X
    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        return np.zeros([X.shape[0],1])
        


""" 
Here is the calling code

"""

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
#print('data loaded')

y_train, y_test = data_train.target, data_test.target

#print("Extracting features from the training data using a count vectorizer")
t0 = time()

vectorizer = CountVectorizer(stop_words='english', binary=True)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 0.0
while alpha < 1.01:
    clf = MyBayesClassifier(alpha)
    clf.train(X_train,y_train)
    y_pred = clf.predict(X_test)
    print '%f %f' %(alpha, np.mean((y_test-y_pred)==0))
    alpha += 0.01
