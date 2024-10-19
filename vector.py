import numpy as np
import pandas as pd

#initialize the vectors
v=pd.array([1,2,3,4,5,6,7,8,9,10])
print('v=',v)

# Access the elements
print('v[0] =', v[0])
print('v[1] =', v[1])
print('v[2] =', v[2])
print('v[3] =', v[3])
print('v[4] =', v[4])
print('v[5] =', v[5])
print('v[6] =', v[6])
print('v[7] =', v[7])
print('v[8] =', v[8])
print('v[9] =', v[9])

#multiplication of matrix
#Create a Matrix
A = np.array( [[ 1, 2**2, 3**2],[4**2, 5**2, 6**2],[7**2, 8**2,9**2]] )
print('A =\n', A, '\n') # View the whole matrix
print('A[2, :] =', A[2, :]) # View a row
print('A[:, 0] =', A[:, 0]) # View a column
print('A[1, 2] =', A[1, 2]) # View a single element

# Make a new matrix
B = np.array( [[ 1, 2, 3],[4, 5, 6],[7, 8,9]] )
print('B =\n', B, '\n') # View the whole matrix
print('Matrix Addition')
print('A+B =\n', A+B, '\n')
print('Matrix Subtraction')
print('A-B =\n', A-B, '\n')

print('Scalar multiplication')
print('3*A =\n', 3*A, '\n')
print('Transpose')
print('A^T =\n', A.T, '\n')
print('Matrix Multiplication')
print('AB =\n', A.dot(B), '\n')



