import numpy as np
import matplotlib.pyplot as plt
inf = 999999999

# taking matrix dimension n,m as input
n = int(input("Enter the row size of the matrix: "))
m = int(input("Enter the column size of the matrix: "))
# creating a matrix with integer entries with range -infinitely to infinitely
A = np.random.randint(-inf,inf,(n,m))
print("The matrix is:\n",A)

#Calculate the Moore-Penrose Pseudoinverse using NumPy’s builtin function
A_pinv = np.linalg.pinv(A)
print("The Moore-Penrose Pseudoinverse is:\n",A_pinv)

#singular value decomposition on non-square matrix
U, s, VT = np.linalg.svd(A)
#calculate pseudoinverse
D=np.zeros((m,n))
for i in range(s.size):
    D[i,i]=1/s[i]
A_pinv1 = VT.T.dot(D).dot(U.T)

#checking if the reconstructed matrix is equal to the original matrix
if np.allclose(A_pinv, A_pinv1):
    print("The reconstructed matrix is equal to the original matrix")
else:
    print("The reconstructed matrix is not equal to the original matrix")


