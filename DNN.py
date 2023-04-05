import numpy as np

class DenseNeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = {}
        # Input layer to first hidden layer
        weights['W1'] = np.random.randn(self.input_size, self.neurons_per_layer[0])
        weights['b1'] = np.zeros((1, self.neurons_per_layer[0]))
        # Hidden layers
        for i in range(1, self.hidden_layers):
            weights['W' + str(i+1)] = np.random.randn(self.neurons_per_layer[i-1], self.neurons_per_layer[i])
            weights['b' + str(i+1)] = np.zeros((1, self.neurons_per_layer[i]))
        # Output layer
        weights['W' + str(self.hidden_layers+1)] = np.random.randn(self.neurons_per_layer[-1], self.output_size)
        weights['b' + str(self.hidden_layers+1)] = np.zeros((1, self.output_size))
        return weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def forward_propagation(self, X):
        self.A = {}
        self.Z = {}
        self.A[0] = X
        for i in range(1, self.hidden_layers+2):
            self.Z[i] = np.dot(self.A[i-1], self.weights['W'+str(i)]) + self.weights['b'+str(i)]
            # self.A[i] = self.sigmoid(self.Z[i])
            self.A[i] = self.relu(self.Z[i])
        return self.A[self.hidden_layers+1]

    def backward_propagation(self, X, y, learning_rate):
        m = X.shape[0]
        self.dW = {}
        self.db = {}
        self.dZ = {}
        self.dA = {}
        L = self.hidden_layers + 1
        self.dZ[L] = self.A[L] - y
        # self.dZ[L] = output - y
        self.dW[L] = (1/m) * np.dot(self.A[L-1].T, self.dZ[L])
        self.db[L] = (1/m) * np.sum(self.dZ[L], axis=0, keepdims=True)
        # print(self.weights['W'+str(L)].shape)
        for l in range(L-1, 0, -1):
            self.dA[l] = np.dot(self.dZ[l+1], self.weights['W'+str(l+1)].T)
            self.dZ[l] = self.dA[l] * self.relu_derivative(self.A[l])
            self.dW[l] = (1/m) * np.dot(self.A[l-1].T, self.dZ[l])
            self.db[l] = (1/m) * np.sum(self.dZ[l], axis=0, keepdims=True)
        # Update weights
        for i in range(1, self.hidden_layers+2):
            self.weights['W'+str(i)] -= learning_rate * self.dW[i]
            self.weights['b'+str(i)] -= learning_rate * self.db[i]
    
    def fit(self, X, y, learning_rate, num_iterations):
        for i in range(num_iterations):
            if i%100==0:print(i)
            output = self.forward_propagation(X)
            self.backward_propagation(X, y, learning_rate)
    
    def predict(self, X):
        return self.forward_propagation(X)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Initialize the neural network
nn = DenseNeuralNetwork(input_size=37, output_size=1, hidden_layers=2, neurons_per_layer=[64, 16])

# Generate some random data for training
# X = np.random.rand(1000, 2)
# y = np.random.rand(1000, 1)

data = np.load('datasets/data0.npz')
states = data["states"]
trumpInfo = data["trumpInfo"]
rewards = data["rewards"]
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


# Train the neural network
nn.fit(X_train, y_train, learning_rate=0.05, num_iterations=1000)
predictions = nn.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)


output = nn.predict(X_test[:10])
print(output)
print(y_test[:10])

#Save weights
print(nn.weights)
np.savez("model/model0.npz", parameter = nn.weights)