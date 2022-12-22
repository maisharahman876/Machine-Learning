import numpy as np
import pandas as pd

def load_dataset(filename):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    # read data from csv
    df=pd.read_csv(filename)
    features=list(df.columns[:-1])
    X=df[features]
    y=df['isoriginal']

    # return 2D feature matrix and a vector of class
    return X,y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    train_size = int(len(X) * (1 - test_size))
    X_new=X
    X_new['Class']=y
    if shuffle:
        #join X and y
        X_new=X_new.sample(frac=1)
    features=list(X_new.columns[:-1])
    X_train=X_new[:train_size]
    y_train=X_train['Class'].to_numpy()
    X_train=X_train[features].to_numpy()
    X_test=X_new[train_size:]
    y_test=X_test['Class'].to_numpy()
    X_test=X_test[features].to_numpy()
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    size = X.shape[0]
    #numpy random sample array with duplicate
    index = np.random.randint(0, size, size)
    X_sample = X[index]
    y_sample = y[index]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    
    return X_sample, y_sample
