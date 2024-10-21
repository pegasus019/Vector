import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model, sklearn.datasets # sklearn is an important package for much of the ML we will be doing

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

# Create our arrays of training data
# Note, because we are using `intercept_fit=True` (the default) in our linear model, we do not have to include the column of 1's
# The model implicitly know we want to calculate theta_0
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([[2.259000208],[2.089513864],[2.691697786],[4.335616639],[5.990895127],[5.715164414],[6.785837863],[7.833729282],[9.136204457],[8.322833045]])

# Create linear regression object
obj = sklearn.linear_model.LinearRegression(fit_intercept=True)

# Train the model using the training sets
obj.fit(X, y)

# Can obtain out model coefficients from the model obj we have trained
print('theta_0:',obj.intercept_)
print('theta_1:',obj.coef_)

# Plot our results
plt.scatter(X, y,  color='black', label='y')
plt.plot(X, obj.predict(X), color='blue', label='Linear Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Look at our evaluation metrics - how well have we fit the data?
# Note: Usually we will be computing these on predictions based on our test data, but for this case we are testing how well our model fits the data we have.
# The mean squared error loss
print('Mean squared error loss: {:.4f}'.format(sklearn.metrics.mean_squared_error(y, obj.predict(X))))
# The R2 score: 1 is perfect prediction
print('R2 score: {:.4f}'.format(sklearn.metrics.r2_score(y, obj.predict(X))))

#Cost Functions

# Define a perfect y=x equation
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

# Generate a random value between +-5
thetas = (np.random.rand(1,1)-0.5)*10

# Predict our y values with this hypothesis: y_pred= X*theta
y_pred = X.dot(thetas)

# Plot outputs to see how well we fit the real data
plt.scatter(X, y,  color='black', label='y_true')
plt.plot(X, y_pred, color='blue', label='y_pred')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

print('Hypothesis: ', thetas) # show our guessed hypothesis
print('Loss: ', sklearn.metrics.mean_squared_error(y, y_pred)) # Mean Squared error we want minimised.

# Try running this cell multiple time to see how the predicted model and loss function change as we make differnt guesses for theta.

