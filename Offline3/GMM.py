import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
class GMM:
    def __init__(self, k, max_iter=30):
        self.k = k
        self.max_iter = int(max_iter)

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape
        # Initialize mu, sigma, and phi
        #mu=k*m and select random row from X 
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [  X[row_index,:] for row_index in random_row ]
        #sigma=k*m*m m*m is an identity matrix
        #self.sigma= [ np.identity(self.m) for _ in range(self.k) ]
        self.sigma = [ np.cov(X.T) for _ in range(self.k) ]
        #phi=k*1
        self.phi = np.ones(self.k) / self.k
        
        

    def e_step(self, X):
        # E-Step: update weights and phi holding mu and sigma constant
        #weights=n*k
        self.weights = self.probability(X)
        #print(self.weights.sum(axis=1))
        self.phi = self.weights.mean(axis=0)
        #print(self.phi)
    
    def m_step(self, X):
        # M-Step: update mu and sigma holding phi and weights constant

        for c in range(self.k):
            #mu=k*m
            self.mu[c] = self.weights[:,c].dot(X) / self.weights[:,c].sum()
            #sigma=k*m*m
            self.sigma[c] = np.cov(X.T, aweights=(self.weights[:,c] / self.weights[:,c].sum()), bias=True)
            #add bias matrix with e-10 to avoid singular matrix
            self.sigma[c]+=np.identity(self.m)*1e-10
   

    def fit(self, X, plot_steps=False):
        self.initialize(X)
        if plot_steps:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            scatter = self.ax.scatter(X[:, 0], X[:, 1])

        # Iterate between E-Step and M-Step until convergence
        for _ in range(self.max_iter):
            self.e_step(X)
            #print(self.weights)
            self.m_step(X)
            if plot_steps:
                for mean, cov in zip(self.mu, self.sigma):
                    x, y = np.mgrid[min(X[:, 0]):max(X[:, 0]):.01, min(X[:, 1]):max(X[:, 1]):.01]
                    pos = np.empty(x.shape + (2,))
                    pos[:, :, 0] = x; pos[:, :, 1] = y
                    rv = multivariate_normal(mean, cov)
                    self.ax.contour(x, y, rv.pdf(pos))
                # Update the data in the scatter plot
                scatter.set_offsets(X)
                plt.pause(0.05)

            
    def probability(self, X):
        likelihood = np.zeros( (self.n, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i],allow_singular=True)
            likelihood[:,i] = distribution.pdf(X)
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def calculate_log_likelihood(self, X):
        #weights = self.probability(X)
        likelihood = np.zeros( (self.n, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i],allow_singular=True)
            
            likelihood[:,i] = distribution.pdf(X)
        log_likelihood = np.log(likelihood.dot(self.phi)+1e-5)
        return log_likelihood.sum()