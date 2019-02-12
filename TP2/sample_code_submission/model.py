'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator

class model (BaseEstimator):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = np.zeros([num_test_samples, self.num_labels])
        # If you uncomment the next line, you get pretty good results for the Iris data :-)
        y = np.round(X[:,3])
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

    
import numpy as np
from sklearn.base import BaseEstimator

class oneR(BaseEstimator):
    ''' One Rule classifier '''
    def __init__(self):
        ''' The "constructor" initializes the parameters '''
        self.selected_feat = 0 	# The chosen variable/feature
        self.theta1 = 0 		# The first threshold
        self.theta2 = 0			# The second threshold
        self.is_trained=False
        
    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
    
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def fit(self, X, Y, F=[]):
        ''' The method "fit" trains a super-simple classifier '''
        if not F: F=[str(item) for item in range(X.shape[1])]
        # First it selects the feature most correlated to the target
        correlations = np.corrcoef(X, Y, rowvar=0)
        self.selected_feat = np.argmax(correlations[0:-1, -1])
        best_feat = X[:, self.selected_feat]
        print('Feature selected = ' +  F[self.selected_feat])
        # Then it computes the average values of the 3 classes
        mu0 = np.median(best_feat[Y==0])
        mu1 = np.median(best_feat[Y==1])
        mu2 = np.median(best_feat[Y==2])
        # Finally is sets two decision thresholds
        self.theta1 = (mu0+mu1)/2.
        self.theta2 = (mu1+mu2)/2.
        self.is_trained=True

    def predict(self, X):
        ''' The method "predict" classifies new test examples '''
        # Select the values of the correct feature
        best_feat = X[:, self.selected_feat]
        # Initialize an array to hold the predicted values
        Yhat = np.copy(best_feat)				# By copying best_fit we get an array of same dim
        # then classify using the selected feature according to the cutoff thresholds
        Yhat[best_feat<self.theta1] = 0											# Class 0
        Yhat[np.all([self.theta1<=best_feat, best_feat<=self.theta2], 0)] = 1	# Class 1
        Yhat[best_feat>self.theta2] = 2 										# Class 2
        return Yhat
    