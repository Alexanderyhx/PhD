""" Weighted Weighted K-Nearest Neighbors (WWKNN)"""
# Authors: Sugam Budhraja <sugam11nov@gmail.com>    Editted by Alexander Hui Xiang Yang  
# License: Open

import warnings
import numpy as np
import scipy as sp
import pandas as pd
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

class WWKNeighborsClassifier():

    """Classifier implementing the weighted distance, weighted variables K-nearest neighbors vote (WWKNN).
    Read more in the paper https://kedri.aut.ac.nz/__data/assets/pdf_file/0009/90846/KASABOV_07.pdf.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.

    neighbors_type : {'total', 'per_class'}, default = 'per_class'
        Method for choosing nearest neighbors.
        -'total' : closest n_neighbors will be chosen that can be 
          from any class.
        -'per_class' : closest n_neighbors will be chosen for each
          class
    
    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and when p = 2, this is 
        euclidean_distance (l2). For arbitrary p, minkowski_distance (l_p) is 
        used.

    initial_weights : {'uniform', 'snr', 'l1reg', 'rfe'} or custom array, default='uniform'
        Feature weights set initially, that are used during distance 
        calculation to determine neighborhood. Possible values:
        - 'uniform' : uniform weights. All features are given equal 
          importance.
        - 'snr' : Signal-to-Noise ratio is used to calculate the weight
          for each feature.
        - 'l1reg' : Logistic Regression penalized with L1 norm. Coefficient 
          of each feature is used as feature weight. Using L1 leads to 
          sparse solutions (many coefficients are zero).
        - 'rfe' : Recursive Feature Elimination using SVM classifier.
          Known to work well on high dimensional data.
        - [custom] : a user-defined weights array which has size equal 
          to the number of features in the data.
    
    voting_type : {'uniform', 'inverse', 'minmax'} or callable, default='minmax'
        Importance of neighbors during voting.  Possible values:
        - 'uniform' : uniform weights. All neighbors are weighted equally.
        - 'inverse' : neighbors are weighted by the inverse of their distance.
          In this case, closer neighbors of a query will have a greater 
          influence than neighbors which are further away.
        - 'minmax' : neighbors are weighted using the following function:
          (max_dist - (neighbor - min_dist)) / max_dist
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    voting_weights : {'uniform', 'snr', 'l1reg', 'rfe'} or custom array, default='snr'
        Feature weights used during voting to determine prediction. 
        Possible values:
        - 'uniform' : uniform weights. All features are given equal 
          importance.
        - 'snr' : Signal-to-Noise ratio is used to calculate the weight
          for each feature.
        - 'rfe' : Recursive Feature Elimination using SVM classifier.
          Known to work well on high dimensional data.
        - [custom] : a user-defined weights array which has size equal 
          to the number of features in the data.

    n_features_to_select: int, default=None
        Number of features to be selected for voting. All features are ranked
        using method specified in voting_weights and top features with higher
        weights are selected.
    
    proba : bool, default=False
        Return class probabilities. False by default.

    plot: str, default=None
        Plot gene profile for test samples, stored with given name. None by default


    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> from wwknn import WWKNeighborsClassifier
    >>> neigh = WWKNeighborsClassifier(n_neighbors=3)
    >>> neigh.fit(X, y)
    WWKNeighborsClassifier(...)
    >>> print(neigh.predict([[1.1]]))
    [0]
    
    Notes
    -----
    .. warning::
       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances
       but different labels, the results will depend on the ordering of the
       training data.
    
    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """
    
    def __init__(self, n_neighbors=5, neighbors_type='per_class', p=2, initial_weights='uniform', voting_type='minmax', voting_weights='snr', n_features_to_select=None, proba=False, plot=None):
        self.n_neighbors = n_neighbors
        self.neighbors_type = neighbors_type
        self.p = p
        self.initial_weights = initial_weights
        self.voting_type = voting_type
        self.voting_weights = voting_weights
        self.n_features_to_select = n_features_to_select
        self.proba = proba
        self.plot = plot


    def fit(self, X, y):
        """Fit the weighted weighted k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        
        y : {array-like, sparse matrix} of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """

        self.n_samples = X.shape[0]
        if self.n_samples == 0:
            raise ValueError("n_samples must be greater than 0")
        self.n_features = X.shape[1]
        if self.n_features == 0:
            raise ValueError("n_features must be greater than 0")
        self.classes = np.unique(y)
        self.n_classes = self.classes.size
        
        # Calculating feature importance (weight)
        if self.initial_weights == 'uniform':
            self.weights = np.ones(self.n_features)
        
        elif self.initial_weights == 'snr':
            class_mean = np.empty((self.n_classes, self.n_features))
            class_std = np.empty((self.n_classes, self.n_features))
            
            for class_i in np.arange(self.n_classes):
                X_class = X[y == self.classes[class_i]]
                class_mean[class_i] = np.mean(X_class, axis=0)
                class_std[class_i] = np.std(X_class, axis=0, ddof=1)
            
            # Calculating weights
            if self.n_classes > 2:        # multiclass
                for class_i in np.arange(self.n_classes):
                    self.weights = (np.abs(class_mean[class_i] - np.sum(class_mean[~class_i], axis=0)) / np.sum(class_std, axis=0))
                self.weights /= self.n_classes
            elif self.n_classes == 2:     # binary
                self.weights = np.abs(class_mean[0] - class_mean[1]) / (class_std[0] + class_std[1])
            else:                         # one class
                self.weights = class_mean[0] / class_std[0]
        
        elif self.initial_weights == 'l1reg':
            clf = LogisticRegression(C = 10000, penalty='l1', solver='liblinear')
            clf = clf.fit(X, y)
            self.weights = np.abs(clf.coef_).mean(axis=0)
        
        elif self.initial_weights == 'rfe':
            estimator = SVC(kernel="linear", C=2000, gamma=0.1)
            self.selector = RFE(estimator, n_features_to_select=1)
            self.selector = self.selector.fit(X, y)
            # X = self.selector.transform(X)
            self.weights = np.float_power(self.selector.ranking_, -0.1)
            # self.n_features = 35
            # self.weights = np.ones(self.n_features)

        else:
            self.weights = self.initial_weights

        self.weights = np.nan_to_num(self.weights, nan=0, posinf=1)
        
        if self.weights.size != self.n_features:
            raise ValueError("size of weights array does not match n_features")

        self.X_train = X.copy()
        self.y_train = y.copy()
        
        return self

        

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.
        
        Returns
        -------
        y : ndarray of shape (n_queries,)
            Class labels for each data sample.
        """

        n_queries = X.shape[0]
        if self.initial_weights == 'rfe':
            X = self.selector.transform(X)
        y_pred = np.empty(n_queries, dtype=self.y_train.dtype) # TODO generalize dtype for normal lists
        probabilities = []
        
        for query in np.arange(n_queries):
            distances = np.empty(self.n_samples) # stores distance of current test sample to every training sample
            
            for sample in np.arange(self.n_samples):
                distances[sample] = sp.spatial.distance.minkowski(self.X_train[sample], X[query], self.p, self.weights)
            
            neighbors = np.empty(0)
            if self.neighbors_type == 'total': 
                neighbors = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors] # O(n), faster than argsort O(nlog(n))
            elif self.neighbors_type == 'per_class':
                for class_i in np.arange(self.n_classes):
                    class_samples = np.where(self.y_train == self.classes[class_i])[0]
                    neighbors_class = np.argpartition(distances[self.y_train == self.classes[class_i]], self.n_neighbors)[:self.n_neighbors]
                    neighbors = np.concatenate([neighbors, class_samples[neighbors_class]])
            else:
                raise ValueError("neighbor_type not valid. Choose 'total' or 'per_class'")
            neighbors = neighbors.astype(int)

            # Calculating feature importance (weight)
            if self.voting_weights == 'uniform':
                feature_weights = np.ones(self.n_features)

            elif self.voting_weights == 'snr':
                class_mean = np.empty((self.n_classes, self.n_features))
                class_std = np.empty((self.n_classes, self.n_features))
                
                if self.neighbors_type == 'total': # TODO Improve
                    X_class = []
                    for class_i in np.arange(self.n_classes):
                        X_class.append([[0]*self.n_features])
                    for neighbor in neighbors:
                        X_class[np.where(self.classes == self.y_train[neighbor])[0][0]].append(self.X_train[neighbor])
                    for class_i in np.arange(self.n_classes):
                        X_class[class_i] = np.asarray(X_class[class_i])
                        class_mean[class_i] = np.mean(X_class[class_i], axis=0)
                        class_std[class_i] = np.std(X_class[class_i], axis=0, ddof=1)
                elif self.neighbors_type == 'per_class':
                    for class_i in np.arange(self.n_classes): # ! if n_neighor > number of samples in class, it will cause errors
                        X_class = self.X_train[neighbors[class_i*self.n_neighbors:(class_i+1)*self.n_neighbors]]
                        class_mean[class_i] = np.mean(X_class, axis=0)
                        class_std[class_i] = np.std(X_class, axis=0, ddof=1)
                
                # Calculating weights
                if self.n_classes > 2:        # multiclass
                    for class_i in np.arange(self.n_classes):
                        feature_weights = (np.abs(class_mean[class_i] - np.sum(class_mean[~class_i], axis=0)) / np.sum(class_std, axis=0)) # Might not work
                    feature_weights /= self.n_classes
                elif self.n_classes == 2:     # binary
                    feature_weights = np.abs(class_mean[0] - class_mean[1]) / (class_std[0] + class_std[1])
                else:                         # one class
                    feature_weights = class_mean[0] / class_std[0]
                feature_weights = np.nan_to_num(feature_weights, nan=0, posinf=1)
                # feature_weights = np.square(feature_weights)
                # feature_weights[feature_weights > 1] = 1

            elif self.voting_weights == 'l1reg':
                clf = LogisticRegression(C = 10000, penalty='l1', solver='liblinear')
                clf = clf.fit(self.X_train[neighbors], self.y_train[neighbors])
                feature_weights = np.abs(clf.coef_).mean(axis=0)

            elif self.voting_weights=='rfe':
                # estimator = LogisticRegression(C=10000)
                estimator = SVC(kernel="linear", C=1000, gamma=0.1)
                selector = RFE(estimator, n_features_to_select=1)
                selector = selector.fit(self.X_train[neighbors], self.y_train[neighbors])
                feature_weights = np.float_power(selector.ranking_, -0.1)
                # feature_weights[np.where(feature_weights < np.float_power(85, -0.1))] = 0

            else:
                feature_weights = self.voting_weights

            # Choosing Top features based on new feature_weights
            if self.n_features_to_select == None:
                top_features = np.arange(self.n_features) # Choose all features
            else:
                top_features = np.argsort(feature_weights*-1)[:self.n_features_to_select]
            # Update distance values based on new feature importance
            for neighbor in neighbors:
                distances[neighbor] = sp.spatial.distance.minkowski(self.X_train[neighbor][top_features], X[query][top_features], self.p, feature_weights[top_features])
    
            # VOTING
            class_vote = {}
            for class_i in np.arange(self.n_classes):
                class_vote[self.classes[class_i]] = 0
            
            if self.voting_type == 'uniform':
                for neighbor in neighbors:
                    class_vote[self.y_train[neighbor]] += 1
                
            elif self.voting_type == 'inverse':
                for neighbor in neighbors:
                    class_vote[self.y_train[neighbor]] += 1/distances[neighbor]

            elif self.voting_type == 'minmax':
                max_dist = np.max(distances[neighbors])
                min_dist = np.min(distances[neighbors])
                for neighbor in neighbors:
                    if max_dist > 0 :
                        class_vote[self.y_train[neighbor]] += ((max_dist - distances[neighbor] + min_dist)/max_dist)
                    else:
                        class_vote[self.y_train[neighbor]] += 0
            else:
                raise ValueError("voting_type not supported. Choose from {'uniform', 'inverse', 'minmax'}")

            y_pred[query] = max(class_vote.items(), key=operator.itemgetter(1))[0]

            if self.proba:
                total_score = np.sum(list(class_vote.values()))
                class_probabilities = {key: value / total_score for key, value in class_vote.items()}
                probabilities.append(class_probabilities)

            if self.plot != None: # TODO improve

                if self.n_features_to_select == None or self.n_features_to_select > 10:
                    n_plot = 10
                else:
                    n_plot = self.n_features_to_select
                n_plot = min(n_plot, self.n_features)
                plot_features = np.argsort(feature_weights*-1)[:n_plot]
                plt.figure(figsize=(20,5))
                x = range(n_plot)
                with open("all_gene_names.txt", 'r') as file:
                    feature_names = np.asarray([a.strip() for a in file.readlines()])
                if self.voting_weights != 'snr':
                    class_mean = np.empty((self.n_classes, self.n_features))
                    X_class = []
                    for class_i in np.arange(self.n_classes):
                        X_class.append([np.zeros(self.n_features)])
                    for neighbor in neighbors:
                        X_class[np.where(self.classes == self.y_train[neighbor])[0][0]].append(self.X_train[neighbor])
                    for class_i in np.arange(self.n_classes):
                        X_class[class_i] = np.asarray(X_class[class_i])
                        class_mean[class_i] = np.mean(X_class[class_i], axis=0)
                class0mean = class_mean[0][plot_features]
                class1mean = class_mean[1][plot_features]
                plt.plot(x, class0mean, '--x', label = "Control (Class 0)")
                plt.plot(x, class1mean, '--x', label = "Subject (Class 1)")
                plt.plot(x, X[query][plot_features], '-o', label = 'Query')
                # plt.bar(x, feature_weights[SNRfeatures], color= 'gold', alpha=0.2)
                # plt.ylim(0, 2)
                plt.xlabel("Top SNR Features")
                plt.ylabel("Gene Expression")
                plt.title("Genetic Profiling")
                plt.xticks(x, feature_names[plot_features], rotation = 90)
                plt.legend()
                plt.savefig('WWKNN Results/'+str(self.plot)+"_"+str(query)+'.png', bbox_inches = 'tight', pad_inches = 0)
                # plt.close(fig)

        if self.proba:
            return y_pred, probabilities

        return y_pred