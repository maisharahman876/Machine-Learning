import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GMM import GMM
if __name__ == '__main__':
    data = pd.read_csv('data2D.txt.dat', sep=' ', header=None)
    X=data.values
    log_likelihoods = []
    for k in range(1, 11):
        gmm = GMM(k=k)
        gmm.fit(X)
        log_likelihoods.append([k,gmm.calculate_log_likelihood(X)])
    #plot a graph where x axis is log_likelihoods[i][0] and y axis is log_likelihoods[i][1] and save it as a png file
    plt.plot([log_likelihoods[i][0] for i in range(len(log_likelihoods))],[log_likelihoods[i][1] for i in range(len(log_likelihoods))])
    plt.savefig('log_likelihoods.png')
    
    gmm=GMM(k=3)
    gmm.fit(X,plot_steps=True)
    plt.savefig('gmm.gif', dpi=100, bbox_inches='tight')