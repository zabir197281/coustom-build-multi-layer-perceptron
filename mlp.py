
import numpy as np
from matplotlib import pyplot as plt
import pprint
np.random.seed(32)


"""-----------------------------Multi layer Persepton-------------------------"""
print()
print("""====================== Multi layer Persepton ============================""")
print()

def multi_layer_Persepton(X,Y):

    hidden_size=2
    input_size=3
    W1 = np.random.randn(hidden_size, input_size)
    b1 = np.ones((hidden_size, 1))
    W2 = np.random.randn(1, hidden_size)
    b2 = np.ones((1, 1))
    costs = []

    lr=0.1
    predict_mlp=0

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    """Sigmoid derivative used for backpropgation """
    def sigmoid_derivative(z):
        return sigmoid(z) * (1 - sigmoid(z))

    """Forward propagation"""
    def forward_propagation(X, W1, b1, W2, b2):
        Z1 = np.dot(W1, X.T) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        return Z2, A1, A2

    """Compute cost"""
    def compute_cost(A2, Y):
        m = Y.shape[0]
        cost = np.mean((0.5 * ((A2 - Y.T) ** 2)))
        return cost




    """Backward propagation"""
    def backward_propagation(X, Y, W2, A1, A2):
        m = X.shape[0]
        dZ2 = A2 - Y.T
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
        dW1 = 1/m * np.dot(dZ1, X)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2

    """Update parameters"""
    def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        return W1, b1, W2, b2
    

    for i in range(10001):
        Z2, A1, A2 = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, Y)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, W2, A1, A2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
        
        if (i%2000==0 and i != 0) or i ==1 :
            costs.append(cost)
            print(f"Cost after {i}  itaration = {cost}")

        if i==10000:
            print("\nGround Truth values: \n",Y.T)
            predict_mlp=np.round(A2)

    
    print("predicted values: \n",np.array(predict_mlp))  


"""AND Gate multi layer persepton"""
print()
print("----AND Gate for multi layer persepton----")
print()

X1=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
Y1=np.array([[0],[0],[0],[0],[0],[0],[0],[1]])
multi_layer_Persepton(X1,Y1)


"""OR Gate multi layer persepton"""
print()
print("----OR Gate for multi layer persepton----")
print()
X2=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
Y2=np.array([[0],[1],[1],[1],[1],[1],[1],[1]])
multi_layer_Persepton(X2,Y2)

"""((x1 ∧ ¬x2) ∨ (¬x1 ∧ x2)) ∧ x3  Gate multi layer persepton"""
print()
print("----((x1 ∧ ¬x2) ∨ (¬x1 ∧ x2)) ∧ x3  Gate multi layer persepton----")
print()
X3=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
Y3=np.array([[0],[0],[0],[1],[0],[1],[0],[0]])
multi_layer_Persepton(X3,Y3)