import numpy as np
import idx2numpy as i2n
from matplotlib import pyplot as plt
import random

#read dataset
data = i2n.convert_from_file('archive/train-images.idx3-ubyte').copy()

#read labels
labels = i2n.convert_from_file('archive/train-labels.idx1-ubyte').copy()

#get data dimensions
count, x, y = data.shape

#split data into dev and training subsets
#get dev indices
dev_size = 1000
dev_idx = random.sample(range(count), dev_size)

#get training indices
dev_idx_sorted = dev_idx
dev_idx_sorted.sort()
train_idx = []

train_idx += range(dev_idx_sorted[0])
for i in range(dev_size-1):
    train_idx += range(dev_idx_sorted[i]+1,dev_idx[i+1])
train_idx += range(dev_idx_sorted[dev_size-1]+1, count)

train_size = len(train_idx)

#dev data + labels
X_dev2D = data[dev_idx]
X_dev = np.zeros((dev_size,x*y))
for i in range(dev_size):
    X_dev[i] = X_dev2D[i].flatten()
    
Y_dev = labels[dev_idx]
X_dev /= 255.

#training data + labels
X_train2D = data[train_idx]
X_train = np.zeros((train_size,x*y))
for i in range(train_size):
    X_train[i] = X_train2D[i].flatten()
    
Y_train = labels[train_idx]

X_train /= 255.

output_size = 10


#initialize iteration 0 compute layer params
def init_params(layer_count, layer_size):
    W = []
    b = []
    W.append(np.random.rand(layer_size, x*y)-0.5)
    b.append(np.random.rand(layer_size, 1)-0.5)
    for i in range(layer_count-2):
        W.append(np.random.rand(layer_size, layer_size)-0.5)
        b.append(np.random.rand(layer_size, 1)-0.5)

    W.append(np.random.rand(output_size, layer_size)-0.5)
    b.append(np.random.rand(output_size, 1)-0.5)

    return W, b

#activation function
def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def forward_prop(W, b, X):
    Z = []
    Z.append(W[0].dot(X.T) + b[0])
    
    A = []
    for i in range(len(W)-1):
        A.append(ReLU(Z[i]))
        Z.append(W[i+1].dot(A[i]) + b[i+1])
    A.append(softmax(Z[len(Z)-1]))
    return Z, A

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z,A,W,b,X,Y):
    one_hot_Y = one_hot(Y)
    m = Y.size
    dZ = []
    dW = []
    dbt = []
    dZ.insert(0, (A[len(A)-1] - one_hot_Y))
    for i in range(1, len(A)):
        ir = len(A) - 1 - i
        dW.insert(0, 1/m * dZ[0].dot(A[ir].T))
        dbt.insert(0, 1/m * np.sum(dZ[0],1))
        dZ.insert(0, W[ir+1].T.dot(dZ[0]) * deriv_ReLU(Z[ir]))
    dW.insert(0, 1/m * dZ[0].dot(X))
    dbt.insert(0, 1/m * np.sum(dZ[0],1))

    db = []
    
    for i in range(len(dbt)):
        dbi = np.zeros((len(dbt[i]), 1))
        for j in range(len(dbt[i])):
            dbij = [dbt[i][j]]
            dbi[j] = dbt[i][j]
        db.append(dbi)
    
    return dW, db

def update_params(W, b, dW, db, alpha):
    for i in range(len(W)):
        W[i] -= alpha * dW[i]
        b[i] -= alpha * db[i]
    return W, b
    
def get_predictions(A):
    return np.argmax(A, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, layer_count, layer_size, iterations, alpha):
    W, b = init_params(layer_count, layer_size)
    acc = []
    for i in range(iterations):
        Z, A = forward_prop(W, b, X)
        dW, db = back_prop(Z, A, W, b, X, Y)
        W, b = update_params(W, b, dW, db, alpha)
        
        acc.append(get_accuracy(get_predictions(A[len(A)-1]), Y))
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", acc[-1])

    plt.figure()
    plt.title('Training Accuracy over time')
    plt.plot(acc)
    plt.show()
    return W, b


def predict(X, W, b):
    Z, A = forward_prop(W, b, X)
    predictions = get_predictions(A[len(A)-1])
    return predictions

def test(img, label, W, b):
    prediction = predict(img, W, b)

    plt.figure()
    plt.title(
        'Prediction: '  +
        str(prediction) +
        '\nLabel: '     +
        str(label)
        )
    img = img.reshape((28,28)) * 255
    plt.imshow(img)
    plt.show()
"""
for i in range(10):
    img = np.zeros((1,x*y))
    img[0] = X_dev[i]
    test(img, Y_dev[i], W, b)

plt.show()
"""
