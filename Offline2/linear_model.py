import matplotlib.pyplot as plt
import numpy as np
class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.learning_rate=params['learning_rate']
        self.iteration=params['iteration']
    def sigmoid(self,X,theta,bias):
        z=np.dot(X,theta)+bias
        return 1/(1+np.exp(-z))
    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        self.X=X
        self.y=y
        self.rows,self.cols=X.shape
        self.theta=np.zeros(self.cols)
        self.bias=0
        theta1=self.theta
        min_cost=99999999

        #gradient descent
        for i in range(self.iteration):
            y_pred=self.sigmoid(self.X,self.theta,self.bias)
            dw=(1/self.rows)*np.dot(self.X.T,(y_pred-self.y))
            db=(1/self.rows)*np.sum(y_pred-self.y)
            self.theta-=self.learning_rate*dw
            self.bias-=self.learning_rate*db
            #cost function
            cost=-(1/self.rows)*np.sum(self.y*np.log(y_pred)+(1-self.y)*np.log(1-y_pred))
            #taking track of minimum cost
            if cost>min_cost:
                min_cost=cost
                theta1=self.theta

        self.theta=theta1
        return self.theta,self.bias



    def predict(self, X,theta,bias):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        y_pred=self.sigmoid(X,theta,bias)
        y_pred=np.where(y_pred>0.5,1,0)
        return y_pred
