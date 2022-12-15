from data_handler import bagging_sampler
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator


    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        #bagging with random with duplicate
        self.estimators = []
        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            self.estimators.append(self.base_estimator.fit(X_sample, y_sample))
        
        


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        #predict with all the estimators and take majority vote
        y_pred = []
        for i in range(self.n_estimator):
            y_pred.append(self.estimators[i].predict(X))
        y_pred = np.array(y_pred)
        y_pred = np.sum(y_pred, axis=0)
        y_pred = np.where(y_pred > self.n_estimator/2, 1, 0)
        return y_pred
        
        

