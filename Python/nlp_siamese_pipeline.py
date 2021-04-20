'''
A library for Paraphrase Classification.

Matt Mauer
'''

import random
import itertools
import re
import math
import time
import pandas as pd
import numpy as np
import csv

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from textdistance import jaccard

from spacy.lang.en import English

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import vocab



class FeatureEngineer:
    '''
    Defined to convert instances into a Fx1 Array where F is the number of manual features.
    Manual features include: 
    '''
    
    def __init__(self, features=['tfidf_sim', 'jaccard5', 'negation_diff', 'number_equiv'],
                 tfidf_vectorizer=None, tfidf_corpus=None, negations=set()):
        
        self.feature_index = {}
        self.feature_constructors = []
        
        if not negations:
            self.negations = {"no", "not", "but", "rather", "instead", "never", 
                              "none", "nobody", "nothing", "neither", "nowhere"}
        
        if tfidf_vectorizer:
            self.tfidf_vectorizer = tfidf_vectorizer
        elif tfidf_corpus:
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_vectorizer.fit(tfidf_corpus)
        elif 'tfidf_sim' in features:
            raise ValueError("tfidf_sim feature requested but no vectorizer or corpus is provided.")
        
        for i, feature in enumerate(features):
            
            self.feature_index[feature] = i
            self.feature_constructors.append(self.initialize_constructor(feature))
        
    
    def construct_features(self, docs=()):
        
        features = []
        
        for constructor in self.feature_constructors:
            features.append(constructor(self, docs))
        
        return features
    
    def initialize_constructor(self, feature):
        
        if feature=='tfidf_sim':
            
            def constructor(self, docs=()):
                
                V = self.tfidf_vectorizer.transform(docs)
                similarity = cosine_similarity(V[0], V[1]).item()
                return similarity
                
        elif feature=='jaccard5':
            
            ngramize = lambda doc, n: {doc[i:i+n] for i in range(len(doc)-n+1)}
            
            def constructor(self, docs=()):
                
                n = 5
                shortest_len = min(len(docs[0]), len(docs[1]))
                
                ngrams1 = ngramize(docs[0], min(n, shortest_len))
                ngrams2 = ngramize(docs[1], min(n, shortest_len))
                
                similarity = jaccard.normalized_similarity(ngrams1, ngrams2)
                return similarity
            
        elif feature=='negation_diff':
            
            def constructor(self, docs=()):
                
                text0 = docs[0].lower()
                n0 = text0.count("n't")
                text0 = text0.split()
                
                text1 = docs[1].lower()
                n1 = text1.count("n't")
                text1 = text1.split()
                
                for negation in self.negations:
                    n0 += text0.count(negation)
                    n1 += text1.count(negation)
                
                return abs(n1 - n0) / 10
            
        elif feature=='number_equiv':
            
            def constructor(self, docs=()):
                
                number_pattern = r"\d+\.?\d*"
                
                nums0 = set(re.findall(number_pattern, docs[0]))
                nums1 = set(re.findall(number_pattern, docs[1]))
                
                return len(nums0 ^ nums1) / 10
            
        else:
            raise ValueError("""Invalid feature constructor. 
                             Valid inputs are: ['tfidf_sim', 'jaccard5', 'negation_diff', 'synonym_ident', 'number_equiv']""")
            
        return constructor

    
class ParaphraseClassifier:
    
    def __init__(
        self, parser=None, embeddings=None, lr=0.01, learning_noise=0.10, 
        net=None, optimizer=None, weight_decay=0, loss_F=None, 
        feature_engineer=None
    ):
        
        if parser:
            self.parser = parser
        else:
            self.parser = English()
        
        if embeddings:
            self.embeddings = embeddings
        else:
            self.embeddings = vocab.GloVe(name='6B', dim=300)
            
        self.vocab = self.embeddings.itos
        self.learning_noise = learning_noise
        
        if net:
            self.net = net
        else:
            raise ValueError('No neural network supplied to classifier.')

        if optimizer:
            self.optimizer = optimizer
        elif weight_decay:
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr)

        if loss_F:
            self.loss_F = loss_F
        else:
            self.loss_F = nn.CrossEntropyLoss()

        self.feature_engineer = feature_engineer

        self.training_time = 0
        self.dev_score = 0
        self.test_score = 0
        self.training_samples = 0
        self.peak_training_samples = 0
        self.peak_epoch = 0
    
    def get_embeds(self, doc):
        
        doc = [token.text for token in self.parser(doc)]
        embeds = self.embeddings.get_vecs_by_tokens(doc, lower_case_backup=True)
        return embeds
        
    def get_noisy_embeds(self, doc):
        '''Replace a single random token with a random token pulled from the embedding library, and return 
        the resulting embeddings.'''
        
        doc = [token.text for token in self.parser(doc)]
        replaced_word = int(random.uniform(0, len(doc)))
        doc[replaced_word] = random.choice(self.vocab)
            
        embeds = self.embeddings.get_vecs_by_tokens(doc, lower_case_backup=True)
        return embeds
    
    def train(self, XYtrain, XYdev, XYtest, epochs=1, verbose=True):
        
        start = time.time()
        
        # for each epoch
        for epoch in range(1, epochs+1):
            self.epochs = 1
            i = 0
            for doc1, doc2 in XYtrain:
                i += 1
                
                embeds1 = self.get_embeds(doc1)
                
                if i % 2 == 0:
                    y = 0
                    p = random.random()
                    if p > 0.5:
                        embeds2 = self.get_noisy_embeds(doc2)
                    else:
                        doc2 = random.choice(XYtrain)[1]
                        embeds2 = self.get_embeds(doc2)
                else:
                    y = 1
                    embeds2 = self.get_embeds(doc2)

                if self.feature_engineer:
                    features = self.feature_engineer.construct_features((doc1, doc2))
                else:
                    features = []
                
                # reset gradients to zero
                self.net.zero_grad()
                
                # feed forward - randomly selecting order of inputs
                if random.random() > 0.5:
                    output = self.net(embeds1, embeds2, features)
                else:
                    output = self.net(embeds2, embeds1, features)
                
                # compute loss -- NEEDS WORK: 
                loss = self.loss_F(output, torch.tensor([y], dtype=torch.long))
                
                # backprop and update weights
                loss.backward()
                self.optimizer.step()

                self.training_samples += 1
            
                if verbose and i%50000==0:

                    # test on dev data
                    dev_score = self.test(XYdev)
                    print(f"After {self.training_samples} training samples, accuracy is {round(dev_score, 4)} on the DEV data.")

                    if dev_score > self.dev_score:
                        self.dev_score = dev_score
                        test_score = self.test(XYtest)
                        print(f"After {self.training_samples} training samples, accuracy is {round(test_score, 4)} on the DEVTEST data.")

                        if test_score > self.test_score:
                            self.test_score = test_score
                            self.peak_training_samples = self.training_samples
                
        self.training_time += time.time() - start
        
        if verbose:
            print("\n------------------------------------------------------------------\n")
            print(f"{round(self.training_time, 1)} seconds has been spent training.")
            print(f"The best score on DEVTEST data was {round(self.test_score, 4)} after {self.peak_training_samples} training samples.")
                
                    
    def predict(self, docs):
        
        embeds1 = self.get_embeds(docs[0])
        embeds2 = self.get_embeds(docs[1])

        if self.feature_engineer:
            features = self.feature_engineer.construct_features(docs)
        else:
            features = []

        probs = self.net(embeds1, embeds2, features)
        
        prediction = torch.argmax(probs).item()
        
        return prediction
        
    def test(self, XY):
        
        Ncorrect = 0
        Ntotal = len(XY)
        
        for doc1, doc2, y in XY:
            
            predicted_label = self.predict((doc1, doc2))
            
            if y == predicted_label:
                Ncorrect += 1
                
        return Ncorrect / Ntotal

class SupervisedParaphraseClassifier(ParaphraseClassifier):
    '''
    Same as the ParaphraseClassifier with one exception. It expects fully 
    labeled training date including negative and positive samples. This means
    no negatives need to be created.
    '''
    def train(self, XYtrain, XYdev, epochs=1, verbose=True):
        
        start = time.time()
        
        # for each epoch
        for epoch in range(1, epochs+1):
            self.epochs = 1
            for doc1, doc2, y in XYtrain:
                
                embeds1 = self.get_embeds(doc1)
                embeds2 = self.get_embeds(doc2)

                if self.feature_engineer:
                    features = self.feature_engineer.construct_features((doc1, doc2))
                else:
                    features = []
                
                # reset gradients to zero
                self.net.zero_grad()
                
                # feed forward - randomly selecting order of inputs
                if random.random() > 0.5:
                    output = self.net(embeds1, embeds2, features)
                else:
                    output = self.net(embeds2, embeds1, features)
                
                # compute loss -- NEEDS WORK: 
                loss = self.loss_F(output, torch.tensor([y], dtype=torch.long))
                
                # backprop and update weights
                loss.backward()
                self.optimizer.step()

                self.training_samples += 1
            
            self.epochs += 1

            # test on dev data
            dev_score = self.test(XYdev)
            print(f"After {self.training_samples} training samples, accuracy is {round(dev_score, 4)} on the DEV data.")

            if dev_score > self.dev_score:
                self.dev_score = dev_score
                self.peak_training_samples = self.training_samples
                
        self.training_time += time.time() - start
        
        if verbose:
            print("\n------------------------------------------------------------------\n")
            print(f"{round(self.training_time, 1)} seconds has been spent training.")
            print(f"The best score on DEV data was {round(self.dev_score, 4)} after {self.peak_training_samples} training samples over {self.epochs} epochs.")
                

class StackedClassifier():

    def __init__(self, siamese_classifier, feature_engineer, super_model):

        self.siamese_classifier = siamese_classifier
        self.feature_engineer = feature_engineer
        self.super_model = super_model
        self.n_training_samples = 0
        self.training_time = 0

    def train(self, XYtrain):

        self.processed_training_data = []
        start = time.time()

        for doc1, doc2, y in XYtrain:

            man_features = self.feature_engineer.construct_features((doc1, doc2))

            embeds1 = self.siamese_classifier.get_embeds(doc1)
            embeds2 = self.siamese_classifier.get_embeds(doc2)

            nn_features = self.siamese_classifier.net(embeds1, embeds2).tolist()

            row = man_features + nn_features[0] + [y]

            self.processed_training_data.append(row)

            self.n_training_samples += 1


        self.processed_training_matrix = np.array(self.processed_training_data)

        self.super_model.fit(
            X = self.processed_training_matrix[:,:-1],
            y = self.processed_training_matrix[:,-1]
        )

        self.training_time += time.time() - start

        self.training_score = self.super_model.score(
            X = self.processed_training_matrix[:,:-1],
            y = self.processed_training_matrix[:,-1]
        )

        print(f"""After training on {self.n_training_samples} training samples and {self.training_time} seconds of training, \
the stacked model has an accuracy of {self.training_score} on the training data.""")

    def predict(self, docs):

        doc1, doc2 = docs

        man_features = self.feature_engineer.construct_features((doc1, doc2))

        embeds1 = self.siamese_classifier.get_embeds(doc1)
        embeds2 = self.siamese_classifier.get_embeds(doc2)

        nn_features = self.siamese_classifier.net(embeds1, embeds2).tolist()

        x = np.array(man_features + nn_features[0]).reshape(1, -1)

        prediction = int(self.super_model.predict(x).item())

        return prediction

    def test(self, XY):
        
        Ncorrect = 0
        Ntotal = len(XY)
        
        for doc1, doc2, y in XY:
            
            predicted_label = self.predict((doc1, doc2))
            
            if y == predicted_label:
                Ncorrect += 1
                
        return Ncorrect / Ntotal


class RootSiameseNet(nn.Module):
    
    def __init__(
        self, embbeding_dim=300, h1_size=128, h2_size=128, num_labels=2,
        h1_expansion=1, xtra_depth=0, n_man_features=0
    ):
        
        super(RootSiameseNet, self).__init__()

        self.embedding_dim = embbeding_dim
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.n_man_features = n_man_features
        self.xtra_depth = xtra_depth

        self.lstm = nn.LSTM(self.embedding_dim, h1_size)
        self.linear1 = nn.Linear(h1_size*h1_expansion+n_man_features, h2_size)
        
        self.h0 = torch.randn(1, 1, self.h1_size)
        self.c0 = torch.randn(1, 1, self.h1_size)

        self.linearXtra = []
        for i in range(xtra_depth):
            self.linearXtra.append(nn.Linear(h2_size, h2_size))
        
        self.classifier = nn.Linear(h2_size, num_labels)
                

        # nn.init.uniform_(self.linear1.weight, a=-0.01, b=0.01)
        # nn.init.uniform_(self.linear1.bias, a=-0.01, b=0.01)
        # nn.init.uniform_(self.linear2.weight, a=-0.01, b=0.01)
        # nn.init.uniform_(self.linear2.bias, a=-0.01, b=0.01)
        

class AbsDiffSiamese(RootSiameseNet):

    def __init__(
        self, embbeding_dim=300, h1_size=128, h2_size=128, num_labels=2,
        xtra_depth=1, n_man_features=0, activation_F=None
    ):

        super(AbsDiffSiamese, self).__init__(
            embbeding_dim=embbeding_dim, h1_size=h1_size, h2_size=h2_size, 
            num_labels=num_labels, h1_expansion=1, xtra_depth=xtra_depth, 
            n_man_features=n_man_features
        )

        if activation_F:
            self.activation_F = activation_F
        else:
            self.activation_F = torch.tanh


    def forward(self, embeds1, embeds2, features=[]):

        h0c0 = (self.h0, self.c0)

        # left LSTM 
        _, (hL, _) = self.lstm(embeds1.view((-1, 1, 300)), h0c0)
        
        # right LSTM
        _, (hR, _) = self.lstm(embeds2.view((-1, 1, 300)), h0c0)
        
        # combination of LSTM outputs
        if features:
            H = torch.cat((
                torch.abs(hL-hR), 
                torch.tensor(features).view(1,1,self.n_man_features)
            ), dim=2)
        else:
            H = torch.abs(hL-hR)
        
        # MLP
        H = self.activation_F(self.linear1(H.view((1,-1))))

        for linear in self.linearXtra:
            H = self.activation_F(linear(H.view((1,-1))))

        out = self.classifier(H)

        return out
    
class ProductSiamese(RootSiameseNet):

    def __init__(
        self, embbeding_dim=300, h1_size=128, h2_size=128, num_labels=2,
        xtra_depth=1, n_man_features=0, activation_F=None
    ):

        super(ProductSiamese, self).__init__(
            embbeding_dim=embbeding_dim, h1_size=h1_size, h2_size=h2_size, 
            num_labels=num_labels, h1_expansion=1, xtra_depth=xtra_depth, 
            n_man_features=n_man_features
        )

        if activation_F:
            self.activation_F = activation_F
        else:
            self.activation_F = torch.tanh


    def forward(self, embeds1, embeds2, features=[]):

        h0c0 = (self.h0, self.c0)

        # left LSTM 
        _, (hL, _) = self.lstm(embeds1.view((-1, 1, 300)), h0c0)
        
        # right LSTM
        _, (hR, _) = self.lstm(embeds2.view((-1, 1, 300)), h0c0)
        
        # combination of LSTM outputs
        if features:
            H = torch.cat((
                hL*hR, 
                torch.tensor(features).view(1,1,self.n_man_features)
            ), dim=2)
        else:
            H = hL*hR
        
        # MLP
        H = self.activation_F(self.linear1(H.view((1,-1))))

        for linear in self.linearXtra:
            H = self.activation_F(linear(H.view((1,-1))))

        out = self.classifier(H)

        return out


def write_results(fp, model, data_dict):
    with open(fp, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["ID", "Category"])
        for i in data_dict:
            prediction = model.predict(data_dict[i])
            writer.writerow([i, prediction])


    