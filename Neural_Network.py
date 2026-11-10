# Digit recognition Neural Network 
# Based heavily on Kiana Jafari's great medium article (https://medium.com/@Kiana-Jafari/digit-recognition-building-a-deep-neural-network-from-scratch-132bd574564a)
# Corrected swish derivative (missing a factor in the activation function which made hidden layer 1's gradient the wrong size)
# Corrected adam bias 

# Task 01 - Imports and data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("mnist_train.csv") 

data = np.array(data) 
data = data.T # Transpose data so columns are samples and rows are features

label = data[0] 
Y = np.eye(10)[label].T
features = data[1:] 
features = features / 255.0 

# Training and validation split
validation_split = 0.2
np.random.seed(0)
m_total = features.shape[1]
perm = np.random.permutation(m_total)
n_val = int(m_total * validation_split)

val_idx = perm[:n_val]
train_idx = perm[n_val:]

X_val = features[:, val_idx]
Y_val = Y[:, val_idx]
X_train = features[:, train_idx]
Y_train = Y[:, train_idx]

# Task 02 - Creating mini-batches
# Shuffles the data and separates it into mini-batches
def random_mini_batches(X, Y, validation_split = 0.2, mini_batch_size = 32, seed = 0):
    np.random.seed(seed) 
    m = X.shape[1] 
    m_val = int(m * validation_split) 

    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Split into training and validation sets
    X_train = shuffled_X[:, m_val:]
    Y_train = shuffled_Y[:, m_val:]
    X_val = shuffled_X[:, :m_val]
    Y_val = shuffled_Y[:, :m_val]

    num_complete_minibatches_train = X_train.shape[1] // mini_batch_size
    mini_batches_train = []

    for k in range(num_complete_minibatches_train):
        mini_batch_X_train = X_train[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y_train = Y_train[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batches_train.append((mini_batch_X_train, mini_batch_Y_train))

    if X_train.shape[1] % mini_batch_size != 0:
        mini_batch_X_train = X_train[:, num_complete_minibatches_train * mini_batch_size :]
        mini_batch_Y_train = Y_train[:, num_complete_minibatches_train * mini_batch_size :]
        mini_batches_train.append((mini_batch_X_train, mini_batch_Y_train))
    return mini_batches_train, (X_val, Y_val)

# Task 03 - Activation functions
def Sigmoid(z):
    return 1 / (1 + (np.exp(-z)))

def Swish(z, gamma=1):
    return z * Sigmoid(z * gamma)

def Swish_derivative(z, gamma=1):
    s = Sigmoid(z * gamma)
    return s + gamma * z * s * (1 - s)

def Leaky_ReLu(z, alpha = 0.01):
    return np.maximum(z, z * alpha)

def Leaky_ReLu_derivative(z, alpha = 0.01):
    return np.where(z > 0, 1, alpha)

def Softmax(z):
    exp_z = np.exp(z - np.max(z, axis = 0, keepdims = True)) 
    return exp_z / np.sum(exp_z, axis = 0, keepdims = True)

# Task 04 - Initializing parameters (weights and biases)
def initialise_parameters(input_size, first_hidden_size, second_hidden_size, output_size):

  W1 = np.random.randn(first_hidden_size, input_size) * 0.01  
  b1 = np.zeros((first_hidden_size, 1))

  W2 = np.random.randn(second_hidden_size, first_hidden_size) * 0.01
  b2 = np.zeros((second_hidden_size, 1))

  W3 = np.random.randn(output_size, second_hidden_size) * 0.01
  b3 = np.zeros((output_size, 1))

  parameters = {"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2,
                "W3":W3,
                "b3":b3}

  return parameters

# Task 05 - Forward propagation
def forward_propagation(X, parameters):

    # Retrieve weights and biases from parameters dictionary
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    # Linear transformations and activation functions
    z1 = np.dot(W1, X) + b1
    a1 = Swish(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = Leaky_ReLu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = Softmax(z3)

    forward_cache = {"z1":z1,
                     "a1":a1,
                     "z2":z2,
                     "a2":a2,
                     "z3":z3,
                     "a3":a3}
    return a3, forward_cache

#  Task 06 - Regularized cost function
def compute_cost(a3, y, parameters, penalty, epsilon = 1e-8):

    W1 = parameters["W1"]
    W2 = parameters["W2"] 
    W3 = parameters["W3"]

    n = y.shape[1] 

    logprobs = np.multiply(-np.log(a3 + epsilon), y) 
    cost = np.sum(logprobs) / n

    regularized_cost = np.divide(penalty, np.multiply(2, n)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) 
    total_cost = cost + regularized_cost

    return total_cost

# Task 07 - Backward propagation
def backpropagation(X, y, parameters, forward_cache, penalty):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    z1 = forward_cache["z1"]
    a1 = forward_cache["a1"]
    z2 = forward_cache["z2"]
    a2 = forward_cache["a2"]
    a3 = forward_cache["a3"]

    m = X.shape[1] 

    # Gradients for output layer
    dZ3 = a3 - y
    dW3 = np.dot(dZ3, a2.T) / m + np.multiply (np.divide(penalty, m), W3)
    db3 = np.sum(dZ3, axis = 1, keepdims = True) / m

    # Gradients for hidden layer 2
    dZ2 = np.dot(W3.T, dZ3) * Leaky_ReLu_derivative(z2)
    dW2 = np.dot(dZ2, a1.T) / m + np.multiply(np.divide(penalty, m), W2)
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m

    # Gradients for hidden layer
    dZ1 = np.dot(W2.T, dZ2) * Swish_derivative(z1)
    dW1 = np.dot(dZ1, X.T) / m + np.multiply(np.divide(penalty, m), W1)
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    gradients = {"dW1":dW1,
                 "db1":db1,
                 "dW2":dW2,
                 "db2":db2,
                 "dW3":dW3,
                 "db3":db3}
    
    return gradients    

# Task 08 - Adam optimization
def initialise_adam(parameters):
    velocity = {}
    cache = {}

    # Initialise velocity and cache
    for key, value in parameters.items():
        velocity[key] = np.zeros_like(value)
        cache[key] = np.zeros_like(value)

    return velocity, cache

# Learning rate decay
def update_parameters(parameters, gradients, velocity, cache, learning_rate, decay_rate, current_iteration, t, momentum_coefficient = 0.9, rmsprop_coefficient = 0.999, epsilon = 1e-8):

    beta1 = momentum_coefficient
    beta2 = rmsprop_coefficient

    corrected_velocity = {}
    corrected_cache = {}

    learning_rate_decay = learning_rate / (1 + decay_rate * current_iteration)

    # Update velocity with momentum term, cache with rmsprop term
    for parameter in parameters:

        velocity[parameter] = beta1 * velocity[parameter] + (1 - beta1) * gradients['d' + parameter]
        corrected_velocity[parameter] = velocity[parameter] / (1 - beta1 ** t)

        cache[parameter] = beta2 * cache[parameter] + (1 - beta2) * (gradients['d' + parameter] ** 2)
        corrected_cache[parameter] = cache[parameter] / (1 - beta2 ** t)

        parameters[parameter] -= learning_rate_decay * corrected_velocity[parameter] / (np.sqrt(corrected_cache[parameter]) + epsilon)

    return parameters, velocity, cache

# Task 09 - Making predictions and model performance
def predict(x, parameters):
    a3, _ = forward_propagation(x, parameters)

    return np.argmax(a3, axis = 0) 

def accuracy_score(prediction, label):

    return np.mean(prediction == np.argmax(label, axis =0 )) 

# Task 10 - Evaluating the neural network
def mini_batch_gradient_descent (X_train, Y_train, X_val, Y_val, input_size, first_hidden_size, second_hidden_size, output_size, penalty, learning_rate, iterations, mini_batch_size = 32, decay_rate = 0.1):

    parameters = initialise_parameters(input_size, first_hidden_size, second_hidden_size, output_size)
    velocity, cache = initialise_adam(parameters)
    t = 0

    # Store cost of both sets
    train_cost_history = []
    val_cost_history = []

    for epoch in range(iterations):
        epoch_cost = 0
        seed = epoch

        mini_batches_train, _ = random_mini_batches(X_train, Y_train, validation_split = 0, mini_batch_size = mini_batch_size, seed = seed )

        for mini_batch_X, mini_batch_Y in mini_batches_train:

            a3, forward_cache = forward_propagation(mini_batch_X, parameters)

            cost = compute_cost(a3, mini_batch_Y, parameters, penalty)
            epoch_cost += cost

            gradients = backpropagation(mini_batch_X, mini_batch_Y, parameters, forward_cache, penalty)

            t += 1
            parameters, velocity, cache = update_parameters(parameters, gradients, velocity, cache, learning_rate, decay_rate, epoch, t)

        epoch_cost /= len(mini_batches_train)
        train_cost_history.append(epoch_cost)

        a3_val, _ = forward_propagation(X_val, parameters)
        val_epoch_cost = compute_cost(a3_val, Y_val, parameters, 0)
        val_cost_history.append(val_epoch_cost)


        if epoch % 10 == 0 or epoch == iterations - 1:
            val_acc = accuracy_score(predict(X_val, parameters), Y_val)
            print(f"Epoch {epoch}: Train Cost = {epoch_cost:.4}, Validation Cost = {val_epoch_cost:.4}, Validation Accuracy = {val_acc:.4}")

    return parameters, train_cost_history, val_cost_history

parameters, train_costs, val_costs = mini_batch_gradient_descent(
    X_train, Y_train, X_val, Y_val, 784, 128, 64, 10,
    penalty = 0.01, learning_rate = 0.001, iterations = 20)

# Driver
test = np.array(pd.read_csv("mnist_test.csv")).T
X_test = test[1:] / 255.0
Y_test = np.eye(10)[test[0]].T
print("Test accuracy: ", accuracy_score(predict(X_test, parameters), Y_test))

# Confusion matrix
predictions = predict(X_test, parameters)
true_labels = np.argmax(Y_test, axis = 0)

confusion = np.zeros((10, 10), dtype = int)
for true, pred in zip(true_labels, predictions):
    confusion[true, pred] += 1

print(confusion)
print(np.trace(confusion) / confusion.sum())

pairs = [(confusion[i, j], i, j) for i in range (10) for j in range (10) if i != j]
for count, i, j in sorted(pairs, reverse = True)[:5]:
    print(f"True {i} predicted as {j}: {count} times")

# Plotting 
errors = confusion.copy()
np.fill_diagonal(errors, 0) # Hide correct predictions

plt.imshow(errors, cmap = "Blues")
plt.colorbar()
plt.xlabel("Predicted digit")
plt.ylabel("True digit")
plt.xticks(range(10))
plt.yticks(range(10))
for i in range (10):
    for j in range (10):
        if errors[i, j] > 0:
            plt.text(j, i, errors[i, j], ha = "center", va = "center")
plt.savefig("confusion_matrix.png")
plt.show()

wrong = np.where(predictions != true_labels)[0]
plt.figure(figsize = (10, 4))
for k, idx in enumerate(wrong[:10]):
    plt.subplot(2, 5, k + 1)
    plt.imshow(X_test[:, idx].reshape(28, 28), cmap = "gray")
    plt.title(f"True {true_labels[idx]}, prediction {predictions[idx]} ")
    plt.axis("off")
plt.savefig("misclassified.png")
plt.show()