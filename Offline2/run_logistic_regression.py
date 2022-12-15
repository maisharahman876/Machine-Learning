"""
main code that you will run
"""

from linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset
from metrics import accuracy,precision_score, recall_score, f1_score


if __name__ == '__main__':
    print("========================================= Logistic Regression ======================================")
    print("Enter learning rate: ")
    learning_rate = float(input())
    print("Enter number of iterations: ")
    iteration = int(input())
    print("Do you want to shuffle the dataset? (y/n)")
    shuffle = input()
    if shuffle == 'y':
        shuffle = True
    else:
        shuffle = False
    print("Enter the proportion of the dataset to include in the test split: ")
    test_size = float(input())

    # data load
    X, y = load_dataset("data_banknote_authentication.csv")
    X.describe()
    y.value_counts()
    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, test_size, shuffle)

    # training
    params = dict(learning_rate=learning_rate,iteration=iteration)
    classifier = LogisticRegression(params)
    theta,bias=classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test,theta,bias)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
