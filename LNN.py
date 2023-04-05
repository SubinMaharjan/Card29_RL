import numpy as np
import json
# from sklearn.model_selection import train_test_split
# from sklearn import datasets
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from numpy.random import randn
# import random
# # from IPython.core.display import display,Image
# from string import Template
# # import IPython.display
# # import warnings

class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        hidden_layer = 64
        print(n_samples, n_features)
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for i in range(self.n_iters):
            if i%100 == 0: print(i)
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
 

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


data = np.load('datasets/data0.npz')
states = data["states"]
trumpInfo = data["trumpInfo"]
rewards = data["rewards"].flatten()
X_data = []
for i, state in enumerate(states):
    X = np.append(state.flatten(), trumpInfo[i])
    X_data.append(X)
X_data = np.array(X_data)
# print(trumpInfo[0])
# print(states[0])
# print(X_data[50])

i = np.random.choice(np.arange(X_data.shape[0]), size=X_data.shape[0], replace=False)
X_data = X_data[i]
rewards = rewards[i]
frac = int(len(X_data)*0.9)
X_train, X_test = X_data[:frac], X_data[frac:]
y_train, y_test = rewards[:frac], rewards[frac:] 


# X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


regressor = LinearRegression(learning_rate=0.05, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
    
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

y_pred_line = regressor.predict(X_train)
print(y_pred_line[:10])
print(rewards[:10])
mse = mean_squared_error(y_pred_line[:10], rewards[:10])
print("MSE:", mse)

np.savez('model/model0.npz', weights = regressor.weights, bias = regressor.bias)

# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
# plt.show()