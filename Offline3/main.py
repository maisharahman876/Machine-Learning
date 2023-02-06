import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GMM import GMM

if __name__ == '__main__':
    data = pd.read_csv('data2D(1).txt', sep=' ', header=None)
    X=data.values
    log_likelihoods = []
    for k in range(1, 11):
        gmm = GMM(k=k)
        gmm.fit(X)
        log_likelihoods.append([k,gmm.calculate_log_likelihood(X)])
    #plot a graph where x axis is log_likelihoods[i][0] and y axis is log_likelihoods[i][1] and save it as a png file
    plt.plot([log_likelihoods[i][0] for i in range(len(log_likelihoods))],[log_likelihoods[i][1] for i in range(len(log_likelihoods))])
    plt.xlabel('no. of clusters')
    plt.ylabel('log_likelihood')
    #plot scatter plot of log_likelihoods
    plt.scatter([log_likelihoods[i][0] for i in range(len(log_likelihoods))],[log_likelihoods[i][1] for i in range(len(log_likelihoods))])

    plt.savefig('log_likelihoods.png')
    plt.clf()
    #find the best k from log_likelihoods
    best_k=int(input("Enter the best k from the graph:"))

    
    gmm=GMM(k=best_k)
    gmm.fit(X,plot_steps=True)
    plt.savefig('best_k.png')
    