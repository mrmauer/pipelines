# A library for streamlining ML processes
# by Matthew Mauer
# last editted 2020-05-10

'''
    EDITS TO COME:
        - more exception handling!!!
        - more Grid Parameters in SupervisedLearner
        - an UnsupervervisedLearner...
'''

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, \
                            accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, \
                                  PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, \
                                 LinearRegression, Lasso
from sklearn.svm import LinearSVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_ridge import KernelRidge

# read in and display the data
# needs more sophisticated API... later
def read(path):
    df = pd.read_csv(path)
    print(df.shape)
    df.sample(5)
    return df

# a function to handle idiosyncracies of data set before any analysis
# MUST BE EDITED FOR EACH APPLICATION...
def clean(df):
    # impute coded -666666 value with NaN
    df.loc[df.med_inc < 0,'med_inc'] = np.nan
    return df

# Perform a broad analysis on all the continuous variables
# If columns aren't passed in, it is assumed all variables are continuous
def see_Cdistribution(df, X=None):
    if X:
        df = df[X]

    print(df.describe(), '\n')
    print("Correlation Table:\n", df.corr(), '\n')
    sns.pairplot(df, diag_kind='kde')

# Describe the categorical variables and render a frequency bar plot for each
def see_Ddistribution(df, X=None):
    if X:
        df = df[X]
    else:
        X = df.columns

    print(df.describe(), '\n')
    
    for Xj in X:
        plt.figure(figsize=(16, 6))
        sns.set()
        sns.countplot(x=Xj, data=df)
        plt.xticks(rotation=90)

# Set seed. Generate a random array of booleans and use it to filter into training/test data
def TTsplit(df, train_size=0.8):
    # sklearn.model_selection.train_test_split() # alternative
    np.random.seed(12244)
    msk = np.random.random(len(df)) < train_size
    train = df[msk]
    test = df[~msk]
    print(f'''There are {train.shape[0]} observations in the training set
        and {test.shape[0]} in the test set.''')
    return train, test

# Convert stated boolean type columns to int {0,1} for all listed dataframes
def bool_to_01(dfs=[], cols=[]):
    converted_dfs = []
    for df in dfs:
        for col in cols:
            df[col] = df[col] * 1
            print(f'{(df[col].mean() * 100).round(1)}% resulted in an arrest.')
        converted_dfs.append(df)
    return converted_dfs

# Fill NaN values with the training sets column mean for all conitnuous vars
def impute(train, test, cols=[]):
    # mean_medinc = df.loc[df.med_inc>=0,'med_inc'].mean()
    avgs = [train[col].mean() for col in cols]
    for i, col in enumerate(cols):
        train[col] = train[col].fillna(value=avgs[i])
        test[col] = test[col].fillna(value=avgs[i])
    return train, test

# leans on helper function to scale continuous variables oftraining and 
# test data by parameters from the training set
def normalize(train, test, cvars=[]):
    train[cvars], scaler = _norm_features(train[cvars])
    test[cvars], _ = _norm_features(test[cvars])
    return train, test

# helper function
# If scaler is not none, use given scaler's means and sds to normalize 
# (used for test set case)
def _norm_features(df, scaler=None):
    # Normalizing training set
    if not scaler:
      scaler = StandardScaler()
      normalized_features = pd.DataFrame(scaler.fit_transform(df))

    # Normalizing test set with the values (ie scaler) from on the training set
    else:
      normalized_features = pd.DataFrame(scaler.transform(df))
      
    # Recover the original indices and column names                                          
    normalized_features.index = df.index
    normalized_features.columns = df.columns

    return normalized_features, scaler

# performing one hot encoding using pandas methods and preserving
# transformation from training set across test sets
def OHE(train, test, dfeatures=[]):
    # get dummies for taining data
    train = pd.get_dummies(train, prefix_sep='__', columns=dfeatures)

    # grab all training dummy col names
    dummies = [col for col in train 
                if "__" in col 
                and col.split("__")[0] in dfeatures]
    # and all training columns (for order preservation)
    all_columns = list(train.columns)

    # get dummies for test data
    test = pd.get_dummies(test, prefix_sep="__", columns=dfeatures)

    # Remove extraneous columns
    for col in test.columns:
        if ("__" in col) and (col.split("__")[0] in dfeatures) and \
                                                            col not in dummies:
            print("Removing additional feature {}".format(col))
            test.drop(col, axis=1, inplace=True)

    # Add column of 0s to test set that correspond with missing cols from
    # the training set
    for col in dummies:
        if col not in test.columns:
            print("Adding missing feature {}".format(col))
            test[col] = 0

    # to rearrange the test set column order to match the training set
    test = test[all_columns]

    return train, test


# transform discrete/categorical variables to dummies for every value
# using scikit learn OneHotEncoder object
def OHE_skl(train, test, dvars=[]):

    # create an encoder and fit the it to the training sets discrete variables
    enc = OneHotEncoder(handle_unknown = 'ignore')
    # flag NaN values 
    train[dvars] = train[dvars].fillna('erase')
    train_dummies = enc.fit_transform(train[dvars]).toarray()

    # grab the new dummy feature names, and replace the discrete features in the 
    # training set
    all_dummies = enc.get_feature_names(dvars)
    new_features = [col for col in all_dummies\
                        if not (col.split("_")[-1] == "erase")]
    train[new_features] = pd.DataFrame(train_dummies,
                                        columns=all_dummies)[new_features]
    train.drop(columns=dvars, inplace=True)

    # using the same feature names, transform and replace f
    # or the test set as well
    test[dvars] = test[dvars].fillna('erase')
    test_dummies = enc.transform(test[dvars]).toarray()
    test[new_features] = pd.DataFrame(test_dummies,
                                        columns=all_dummies)[new_features]
    test.drop(columns=dvars, inplace=True)

    return train, test


# train a model
# display the amount of time training required
class SupervisedLearner():
    # Config: Dictionaries of models and hyperparameters
    MODELS = {
        'LogisticRegression': LogisticRegression(solver='lbfgs'), 
        'LinearSVC': LinearSVC(), 
        'GaussianNB': GaussianNB(),
        'OLS': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'KernelRidge': KernelRidge(),
    }

    GRID = {
        'LogisticRegression': [{'penalty': x, 'C': y, 'random_state': 0} 
                               for x in ('l2', 'none') \
                               for y in (0.01, 0.1, 1, 10, 100)],
        'GaussianNB': [{'priors': None}],
        'LinearSVC': [{'C': x, 'random_state': 0} \
                      for x in (0.01, 0.1, 1, 10, 100)]
    }

    def __init__(self, train, test, target='target', model_type='regression'):
        self.Ytrain = train[target]
        self.Xtrain = train.drop(columns=target)
        self.Ytest = test[target]
        self.Xtest = test.drop(columns=target)
        self.model_type = model_type

        # subset the class models and grid by type 
        if model_type == 'linear_regression':
            self.model_selection = ['OLS', 'Lasso', 'Ridge']
        elif model_type == 'regression':
            self.model_selection = ['OLS', 'KernelRidge']
        elif model_type == 'classification':
            self.model_selection = ['LogisticRegression', 'LinearSVC', 'GaussianNB']
        else:
            raise Exception('''Invalid model type for evaluation.
                                options are regression, classification.''')

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        self._model_type = value

    @property
    def Ytrain(self):
        return self._Ytrain
    
    @Ytrain.setter
    def Ytrain(self, value):
        self._Ytrain = value

    @property
    def Xtrain(self):
        return self._Xtrain
    
    @Xtrain.setter
    def Xtrain(self, value):
        self._Xtrain = value

    @property
    def Ytest(self):
        return self._Ytest
    
    @Ytest.setter
    def Ytest(self, value):
        self._Ytest = value

    @property
    def Xtest(self):
        return self._Xtest
    

    @Xtest.setter
    def Xtest(self, value):
        self._Xtest = value

    @property
    def model_selection(self):
        return self._model_selection

    @model_selection.setter
    def model_selection(self, value):
        self._model_selection = value
   
    @property
    def results_GS(self):
        return self._results_GS

    @results_GS.setter
    def results_GS(self, df):
        self._results_GS = df
    

    # train, transform, and avaluate for one specific model 
    def single_model(self, model_name, params, see_ftrs=10):

        # create, customize, and train model and predict Y for test set
        model = self.MODELS[model_name]
        model.set_params(**params)
        model.fit(self.Xtrain, self.Ytrain)
        Ypred = model.predict(self.Xtest)

        # evaluate the model with visualizations
        self.eval(Ypred, model=model, visual=True)

        # display the 10 most weighted coefficients
        coeff = pd.DataFrame(self.Xtrain.columns, columns=['feature'])
        coeff['weights'] = model.coef_.reshape(-1,1)
        print(coeff.sort_values(by='weights', ascending=False).head(see_ftrs))

        return model

    # fit, predict, and evaluate for every model/parameter combination and 
    # and return evaluation results
    def grid_search(self):

        # Begin timer 
        start = datetime.datetime.now()

        # Initialize results data frame 
        results_df = pd.DataFrame()

        # Loop over models 
        for model_key in self.model_selection: 
            
            # Loop over parameters 
            for params in self.GRID[model_key]: 
                print(f"Training model: {model_key} | {params}")
                
                # Create model 
                model = self.MODELS[model_key]
                model.set_params(**params)
                
                # Fit model on training set 
                model.fit(self.Xtrain, self.Ytrain)
                
                # Predict on testing set 
                Ypred = model.predict(self.Xtest)
                
                row = {'model':model_key}
                row.update(params)

                # Evaluate predictions by model type
                # complete the results row for this model + parameters
                if 'regression' in self.model_type:
                    mae, mse, r2 = self.eval(Ypred, model=model, output=False)
                    row.update({'MAE':mae, 'MSE':mse, 'R^2':r2})  

                elif self.model_type == 'classification':
                    accuracy = self.eval(Ypred, output=False)
                    row.update({'Accuracy':accuracy})

                else:
                    raise Exception('''Invalid model type for grid search.
                                Options are regression, classification.''')

                # Store results in your results data frame 
                results_df = results_df.append(row, ignore_index=True)
                
        # End timer
        stop = datetime.datetime.now()
        print("Time Elapsed:", stop - start)

        self.results_GS = results_df
        return self.results_GS       

    # evaluate an individual supervised learning model depending on its type
    # vizualize the meric of the model if specified
    def eval(self, Ypred, model=None, output=True, visual=False):

        if 'regression' in self.model_type:
            mae = mean_absolute_error(Ypred, self.Ytest)
            mse = mean_squared_error(Ypred, self.Ytest)
            r2 = model.score(self.Xtrain, self.Ytrain)
            
            if output:
                print("Mean Absolute error: %.2f" % mae)
                print("Mean squared error: %.2f" % mse)
                print("R-squared: %.2f \n" % r2)
            
            return (mae, mse, r2)

        elif self.model_type == 'classification':
            accuracy = accuracy_score(self.Ytest, Ypred)

            if output:
                print("Accuracy Score: %.2f" % accuracy)

            if visual:
                sns.heatmap(confusion_matrix(self.Ytest, Ypred), annot=True)

            return accuracy

        else:
            raise Exception('''Invalid model type for evaluation.
                                options are regression, classification.''')

