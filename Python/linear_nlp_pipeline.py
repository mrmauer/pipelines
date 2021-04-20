'''
A library of objects for NLP text classifaction.
Included:
- a model for Linear Classification that trains on SSD with either hinge or perceptron loss.
- a flexible TF-IDF vectorizer
- a Unigram Binary vectorizer

Author: Matt Mauer
Date: 10/29/2020
'''

import numpy as np
import time
import random
from operator import itemgetter
import json


class LinearClassifier:
    '''
    A supervised learner for multi-label classification using SSD with perceptron or
    hinge loss.
    '''
    
    def __init__(self, Yspace=[0, 1, 2], features=set(), eta=0.01, initW=0, lossF="perceptron"):
        self._Yspace = Yspace
        self._eta = eta
        self._initW = initW
        self._Nepochs = 0
        self.loss = lossF
        self._features = features
        self._training_time = 0
        
    @property
    def loss(self):
        return self._loss
    
    @loss.setter
    def loss(self, lossF):
        if lossF == 'perceptron':
            self._loss = self._perceptron
        elif lossF == 'hinge':
            self._loss = self._hinge
        else:
            raise ValueError('Invalid name for loss function.')
    
    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, newW):
        self._W = newW
        
    def _perceptron(self, sample, gold_standard):
        
        # predict class
        # if correct -> return
        # if incorrect ->
            # W = W0 + (eta * W1) for gold standard
            # W = W0 - (eta * W1) for false prediction
            
        pred_class = self.predict_one(sample)
        
        if pred_class == gold_standard:
            return
        else:
            for token in sample:
                self._W[pred_class][token] -= self._eta * sample[token]
                self._W[gold_standard][token] += self._eta * sample[token]
    
    def _hinge(self, sample, gold_standard):
        
        # determine the CostClass (the label with the highest score + cost)
        # if the CostClass is the Gold Standard -> return
        # if CostClass isn't the Gold Standard ->
            # W = W0 + (eta * W1) for gold standard weights
            # W = W0 - (eta * W1) for false prediction
            
        fear_label = self._costClassify(sample, gold_standard)
        
        if fear_label == gold_standard:
            return
        else:
            for token in sample:
                self._W[fear_label][token] -= self._eta * sample[token]
                self._W[gold_standard][token] += self._eta * sample[token]
        
    def _costClassify(self, x, gold_standard):
        
        label_scores = {}
        
        for label in self._W:
            score = sum(map(lambda i: self._W[label][i] * x[i], x))
            if label == gold_standard:
                label_scores[score] = label
            else:
                label_scores[score + 1] = label
        
        fear_label = label_scores[max(label_scores)]
        
        return fear_label
            
    def train(self, X, Y, Xdev=None, Ydev=None, Xtest=None, Ytest=None, epochs=1, online=False, verbose=True):
        
        training_start = time.time()
        self._best_dev_accuracy = 0
        self._best_test_accuracy = 0
        
        if not online:
            self.W = {y : dict.fromkeys(self._features, self._initW) for y in self._Yspace}
        
        for epoch in range(epochs):
            self._Nepochs += 1
            for i, x in enumerate(X):
                self.loss(x, Y[i])
                
            if Xdev:
                dev_accuracy = self.test(Xdev, Ydev)
                if verbose:
                    print(f"After epoch {self._Nepochs}, accuracy is {round(dev_accuracy, 4)} on the DEV data.")
                
                if dev_accuracy > self._best_dev_accuracy:
                    self._best_dev_accuracy = dev_accuracy
                    
                    if Xtest:
                        test_accuracy = self.test(Xtest, Ytest)
                        if verbose:
                            print(f"After epoch {self._Nepochs}, accuracy is {round(test_accuracy, 4)} on the TEST data.")
                        
                        if test_accuracy > self._best_test_accuracy:
                            self._best_test_accuracy = test_accuracy
                
                
    
        self._training_time += time.time() - training_start
        print(f"A total of {round(self._training_time, 2)} seconds has been spent training.")
    
    def predict_one(self, x):
        
        label_scores = {}
        
        for label in self._W:
            score = sum(map(lambda i: self._W[label][i] * x[i], x))
            label_scores[score] = label
        
        prediction = label_scores[max(label_scores)]
        
        return prediction
    
    def predict(self, X):
        
        if isinstance(X, dict):
            return self.predict_one(X)
        else:
            labels = list(map(self.predict_one, X))
            
            return labels
            
    
    def test(self, X, Y):
        
        predictions = self.predict(X)
        
        accuracy = (Y == predictions).mean()
        return accuracy
    
    def feature_importance(self, n=10, pretty_print=False):
        
        label_features = {}
        
        for label in self._W:
            topn = dict(sorted(self._W[label].items(), key = itemgetter(1), reverse = True)[:n])
            label_features[label] = topn
            
        if pretty_print:
            print(json.dumps(label_features, indent=4))
            
        return label_features
                
        
class BinaryUnigramVectorizer:
    '''
    A vectorizer for text data. Accepts an array/Series of text samples.
    Ngrams of size up to 5. 
    '''
    
    def __init__(self, n=1, label_name="_Y", vocab=set(), features='_BINARY', lemmatizer=None, stemmer=None, cutoff=1):
        
        if n > 5:
            raise ValueError("n must be less than 6.")
            
        self._lemmatizer = lemmatizer
        self._stemmer = stemmer
        self.vocab = vocab
        
    @property
    def vocab(self):
        return self._vocab
    
    @vocab.setter
    def vocab(self, V):
        self._vocab = V
    
    def convert_text_fit(self, text):
        
        d = {}
        for t in text.split(' '):
            d[t] = 1
            self._vocab.add(t)
            
        return d
        
    def fit_transform(self, X):
        
        X = [self.convert_text_fit(x) for x in X]
        
        return X
    
    def convert_text_transform(self, text):
        
        d = {}
        for t in text.split(' '):
            if t in self.vocab:
                d[t] = 1
            
        return d
    
    def transform(self, X, _internal=False):
        
        X = [self.convert_text_transform(x) for x in X]
        
        return X
            

class TfIdfVectorizer:
    '''Class for more vectorizing TF-IDF values for a corpus.'''

    def __init__(self, ngram_size=1, stop_words=set(), lemmatizer=None): 
        
        self.preprocess = (stop_words, lemmatizer)
        self._ngram_size = ngram_size
        
    @property
    def vocab(self):
        return self._DF.keys()
        
    @property
    def preprocess(self):
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, processing_tools):
        
        stop_words, lemmatizer = processing_tools
        
        if lemmatizer:
            f = lemmatizer
        else:
            f = lambda doc: doc.split(' ')
        
        if stop_words:
            self._preprocess = lambda doc: list(filter(lambda t: t not in stop_words, f(doc)))
        else:
            self._preprocess = f
        
    def fit_transform_one(self, doc):
        
        tf = {}
        doc = self.preprocess(doc)
        
        for i in range(self._ngram_size-1, len(doc)-self._ngram_size+1):
            ngram = ' '.join(doc[i : i+self._ngram_size])
  
            if ngram in tf:
                tf[ngram] += 1
            else:
                tf[ngram] = 1
                
                if ngram in self._DF: 
                    self._DF[ngram] += 1
                else:
                    self._DF[ngram] = 1
                
        return tf
    
    def _inverse(self, TF):
    
        for i, sample in enumerate(TF):
            
            for ngram in sample:
                sample[ngram] /= self._DF[ngram]
                
            TF[i] = sample 
                
        return TF
    
    def fit_transform(self, corpus):
        
        self._DF = {}
        TF = [self.fit_transform_one(doc) for doc in corpus]
        TfIdf = self._inverse(TF)
        
        return TfIdf

    def _transform_one(self, doc):
        
        doc = self.preprocess(doc)
        tfidf = {}
        
        for i in range(self._ngram_size-1, len(doc)-self._ngram_size+1):
            ngram = ' '.join(doc[i : i+self._ngram_size])
            
            if ngram in self._DF:
                if ngram in tfidf:
                    tfidf[ngram] += 1/self._DF[ngram]
                else:
                    tfidf[ngram] = 1/self._DF[ngram]
                
        return tfidf
    
    def transform(self, corpus):
        TfIdf = [self._transform_one(doc) for doc in corpus]
        return TfIdf
            
