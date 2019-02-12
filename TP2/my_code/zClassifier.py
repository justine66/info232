"""
Created on Sat Mar 11 08:04:23 2017
Last revised: Feb 2, 2019

@author: isabelleguyon

This is an example of classifier program, we show how to combine
a classifier and a preprocessor with a pipeline. 

IMPORTANT: when you submit your solution to Codalab, the program run.py 
should be able to find your classifier. Currently it loads "classifier.py"
from the sample_code/ directory. If you do not want to modify run.py, 
copy all your code to sample_code/ and rename zClassifier.py to Classifier.py.
Alternatively, make sure that the path makes it possible
to find your code and that you import your own version of "Classifier":
in run.py, add
lib_dir2 = os.path.join(run_dir, "my_code")
path.append (lib_dir2)
and change 
from data_manager import DataManager 
from classifier import Classifier    
to
from zDataManager import DataManager 
from zClassifier import Classifier    
"""

from sys import argv
from sys import path

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import numpy as np
    import pickle
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import Perceptron
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_score
    from zPreprocessor import Preprocessor

def ebar(score, sample_num):
    '''ebar calculates the error bar for the classification score (accuracy or error rate)
    for sample_num examples'''
    return np.sqrt(1.*score*(1-score)/sample_num)

class RandomPredictor(BaseEstimator):
    ''' Make random predictions.'''	
    def __init__(self):
        self.target_num=0
        return
        
    def __repr__(self):
        return "RandomPredictor"

    def __str__(self):
        return "RandomPredictor"
	
    def fit(self, X, Y):
        if Y.ndim == 1:
            self.target_num=len(set(Y))
        else:
            self.target_num==Y.shape[1]
        return self
		
    def predict_proba(self, X):
        prob = np.random.rand(X.shape[0],self.target_num)
        return prob	
    
    def predict(self, X):
        prob = self.predict_proba(X)
        yhat = [np.argmax(prob[i,:]) for i in range(prob.shape[0])]
        return np.array(yhat)

class BasicClassifier(BaseEstimator):
    '''BasicClassifier: modify this class to create a simple classifier of
    your choice. This could be your own algorithm, of one for the scikit-learn
    classfiers, with a given choice of hyper-parameters.'''
    def __init__(self):
        '''This method initializes the parameters. This is where you could replace
        RandomForestClassifier by something else or provide arguments, e.g.
        RandomForestClassifier(n_estimators=100, max_depth=2)'''
        self.clf = RandomForestClassifier(random_state=1)

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        return self.clf.fit(X, y)

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
    
class MonsterClassifier(BaseEstimator):
    '''MonsterClassifier: This is a more complex example that shows how you can combine
    basic modules (you can create many), in parallel (by voting using ensemble methods)
    of in sequence, by using pipelines.'''
    def __init__(self):
        '''You may here define the structure of your model. You can create your own type
        of ensemble. You can make ensembles of pipelines or pipelines of ensembles.
        This example votes among two classifiers: BasicClassifier and a pipeline
        whose classifier is itself an ensemble of GaussianNB classifiers.'''
        fancy_classifier = Pipeline([
					('preprocessing', Preprocessor()),
					('classification', BaggingClassifier(base_estimator=GaussianNB(),random_state=1))
					])
        self.clf = VotingClassifier(estimators=[
					('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
					('Gaussian Classifier', GaussianNB()),
					('Support Vector Machine', SVC(probability=True)),
					('Fancy Classifier', fancy_classifier)],
					voting='soft')   
        
    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        return self.clf.fit(X, y)

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
        
def compute_accuracy(M, D, classifier_name):
	'''Evaluate the accuracy of M on D'''
	# Train
	Ytrue_tr = D.data['Y_train']
	M.fit(D.data['X_train'], Ytrue_tr)
    
	# Making classification predictions (the output is a vector of class IDs)
	Ypred_tr = M.predict(D.data['X_train'])
	Ypred_va = M.predict(D.data['X_valid'])
	Ypred_te = M.predict(D.data['X_test'])  
    
	# Training success rate and error bar:
	acc_tr = accuracy_score(Ytrue_tr, Ypred_tr)
        
	# Cross-validation performance:
	acc_cv = cross_val_score(M, D.data['X_train'], Ytrue_tr, cv=5, scoring='accuracy')
        
	print("%s\ttrain_acc=%5.2f(%5.2f)\tCV_acc=%5.2f(%5.2f)" % (classifier_name, acc_tr, ebar(acc_tr, Ytrue_tr.shape[0]), acc_cv.mean(), acc_cv.std()))
	# Note: we do not know Ytrue_va and Ytrue_te so we cannot compute validation and test accuracy
	return acc_tr

def ClfScatter(self, clf, dim1=0, dim2=1, title=''):
        '''(self, clf, dim1=0, dim2=1, title='')
        Split the training data into 1/2 for training and 1/2 for testing.
        Display decision function and training or test examples.
        clf: a classifier with at least a fit and a predict method
        like a sckit-learn classifier.
        dim1 and dim2: chosen features.
        title: Figure title.
        Returns: Test accuracy.
        '''
        X = self.data['X_train']
        Y = self.data['Y_train']
        F = self.feat_name
        # Split the data
        ntr=round(X.shape[0]/2)
        nte=X.shape[0]-ntr
        Xtr = X[0:ntr, (dim1,dim2)]
        Ytr = Y[0:ntr]
        Xte = X[ntr+1:ntr+nte, (dim1,dim2)]
        Yte = Y[ntr+1:ntr+nte]
        # Fit model in chosen dimensions
        clf.fit(Xtr, Ytr)
        # Compute the training score
        Yhat_tr = clf.predict(Xtr) 
        training_accuracy = accuracy_score(Ytr, Yhat_tr)
        # Compute the test score
        Yhat_te = clf.predict(Xte)  
        test_accuracy = accuracy_score(Yte, Yhat_te)       
        # Define a mesh    
        x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
        y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
        h = 0.1 # step
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
							 np.arange(y_min, y_max, h))
        Xgene = np.c_[xx.ravel(), yy.ravel()]
        # Make your predictions on all mesh grid points (test points)
        Yhat = clf.predict(Xgene) 
        # Make contour plot for all points in mesh
        Yhat = Yhat.reshape(xx.shape)
        plt.subplot(1, 2, 1)
        plt.contourf(xx, yy, Yhat, cmap=plt.cm.Paired)
        # Overlay scatter plot of training examples
        plt.scatter(Xtr[:, 0], Xtr[:, 1], c=Ytr, cmap=cm)   
        plt.title('{}: training accuracy = {:5.2f}'.format(title, training_accuracy))
        plt.xlabel(F[dim1])
        plt.ylabel(F[dim2])
        plt.subplot(1, 2, 2)
        plt.contourf(xx, yy, Yhat, cmap=plt.cm.Paired)
        # Overlay scatter plot of test examples
        plt.scatter(Xte[:, 0], Xte[:, 1], c=Yte, cmap=cm)   
        plt.title('{}: test accuracy = {:5.2f}'.format(title, test_accuracy))
        plt.xlabel(F[dim1])
        plt.ylabel(F[dim2])
        plt.subplots_adjust(left  = 0, right = 1.5, bottom=0, top = 1, wspace=0.2)
        plt.show()
        return test_accuracy

def test(D):  
    '''Function to try some examples classifiers'''    
    classifier_dict = {
            '1. MonsterClassifier': MonsterClassifier(),
            '2. SimplePipeline': Pipeline([('prepro', Preprocessor()), ('classif', BasicClassifier())]),
            '3. RandomPred': RandomPredictor(),
            '4. Linear Discriminant Analysis': LinearDiscriminantAnalysis()}
            
    for key in classifier_dict:
        myclassifier = classifier_dict[key]
        acc = D.ClfScatter (myclassifier,dim1=0, dim2=1, title=key) # Replace by a call to ClfScatter
              
    return acc # Return the last accuracy (important to get the correct answer in the TP)
    
if __name__=="__main__":
    # We can use this function to test the Classifier
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../results"
        score_dir = "../scoring_program"
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        score_dir = argv[3]
                            
	# The M2 may have prepared challenges using sometimes AutoML challenge metrics
    path.append(score_dir)
    
    from zDataManager import DataManager # The class provided by binome 1
    
    basename = 'Iris'
    D = DataManager(basename, input_dir) # Load data
    print(D)
    test(D)
 
