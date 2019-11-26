"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.

    Parameters
    --------------------
        fname  -- string, filename

    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.

    Parameters
    --------------------
        infile    -- string, filename

    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """

    word_list = {}
    with open(infile, 'r') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        for line in fid:
            words = extract_words(line)
            for word in words:
                if word not in word_list:
                    word_list[word] = len(word_list)-1
        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.

    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)

    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """

    num_lines = sum(1 for line in open(infile,'r'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))

    with open(infile, 'r') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        for i, line in enumerate(fid):
            words = set(extract_words(line))
            for word in words:
                feature_matrix[i, word_list[word]] = 1
        ### ========== TODO : END ========== ###
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1

    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == 'accuracy':
        return metrics.accuracy_score(y_true, y_label)
    elif metric == 'f1_score':
        return metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
        return metrics.roc_auc_score(y_true, y_label)
    return 0
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance
    score = 0
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.decision_function(X_test)
        score += performance(y_test, y_pred, metric)
    return score / kf.get_n_splits()
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """

    print ('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)

    ### ========== TODO : START ========== ###
    # part 2: select optimal hyperparameter using cross-validation
    best_c = None
    best_perf = float('-inf')
    for c in C_range:
        clf = SVC(kernel='linear', C=c)
        perf = cv_performance(clf, X, y, kf, metric)
        print(c, perf)
        if perf > best_perf:
            best_perf = perf
            best_c = c

    return best_c
    ### ========== TODO : END ========== ###



def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 3: return performance on test data by first computing predictions and then calling performance
    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric)
    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    np.random.seed(1234)

    # read the tweets and its labels
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')

    metric_list = ["accuracy", "f1_score", "auroc"]

    ### ========== TODO : START ========== ###
    print(X.shape)
    # part 1: split data into training (training + cross-validation) and testing set
    X_train, X_test = X[:560, :], X[560:, :]
    y_train, y_test = y[:560], y[560:]

    # part 2: create stratified folds (5-fold CV)
    kf = StratifiedKFold(n_splits=5)

    # part 2: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    optimal_c = dict()
    for metric in metric_list:
        optimal_c[metric] = select_param_linear(X_train, y_train, kf, metric=metric)
    print(optimal_c)
    # part 3: train linear-kernel SVMs with selected hyperparameters

    # part 3: report performance on test data
    for metric in metric_list:
        clf = SVC(kernel='linear', C=optimal_c[metric])
        clf.fit(X_train, y_train)
        print(metric, performance_test(clf, X_test, y_test, metric))


    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
