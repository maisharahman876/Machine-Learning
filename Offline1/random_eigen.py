import numpy as np
import matplotlib.pyplot as plt
inf = 999999999
#taking matrix size as input
n = int(input("Enter the e dimension of the matrix: "))
#creating a random invertible matrix with integer entries with range -infinitely to infinitely
A=[[]]
while True:
    A = np.random.randint(-inf,inf,(n,n))
    if np.linalg.det(A) != 0:
        break
print("The matrix is:\n",A)

#finding eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("The eigenvalues are:\n",eigenvalues)
print("The eigenvectors are:\n",eigenvectors)

#Reconstruct A from eigenvalue and eigenvectors 
A_reconstructed = eigenvectors.dot(np.diag(eigenvalues)).dot(np.linalg.inv(eigenvectors))
print("The reconstructed matrix is:\n",A_reconstructed)

#checking if the reconstructed matrix is equal to the original matrix
if np.allclose(A, A_reconstructed):
    print("The reconstructed matrix is equal to the original matrix")
else:
    print("The reconstructed matrix is not equal to the original matrix")

