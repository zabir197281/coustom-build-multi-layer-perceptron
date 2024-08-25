
import numpy as np
from matplotlib import pyplot as plt
import pprint
np.random.seed(42)


"""-----------------------------Single layer Persepton-------------------------"""
print()
print("""====================== Single layer Persepton ===========================""")
print()

def single_layer_Persepton(X,Y):
    W=np.random.randn(3,1)
    bias=1

    lr=0.1
    predict=0
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    """Sigmoid derivative used for backpropgation """
    def sigmoid_derivative(x): 
        return sigmoid(x) * (1 - sigmoid(x))

    """Forward propagation"""
    def forward_propagation():
        z = np.dot(X,W)+bias
        z_activation_func = sigmoid(z)
        return  z_activation_func

    """Backword propagation"""
    def backword_prop( z_activation_func,Y):
        m=3 # Sgape of X
        MSE= z_activation_func- Y
        grad_w= 1/m*(np.dot((X*sigmoid_derivative(Y)).T ,MSE))
        dbias = 1 / m * np.sum(MSE)
        return grad_w,dbias

    """Updating the weights"""
    def Update_param(W,grad_w,dbias,bias):
        W= W-lr*grad_w
        bias=bias-lr*dbias
        return W,bias

    """Binary cross entropy cost function"""
    def BCECost(Y, z_activation_func): 
        bce_cost = -(np.sum(Y * np.log(z_activation_func) + (1 - Y) * np.log(1 - z_activation_func))) / len(Y)
        return bce_cost



    """Training  the data for 1000 epocs"""
        
    for i in range(10001):
        z_activation_func=forward_propagation()
        grad_w,dbias=backword_prop( z_activation_func,Y)
        W,bias=Update_param(W,grad_w,dbias,bias)
        bce_cost=BCECost(Y,z_activation_func)

        if (i%2000==0 and i != 0) or i ==1 :
            print(f"Cost after {i}  itaration = {bce_cost}")

        if i==10000:
            print("\nGround Truth values: \n",Y.T)
            predict=np.round(z_activation_func)
    print("predicted values: \n",np.array(predict).T)       



"""AND Function single layer persepton"""
print()
print("----AND Gate single layer persepton----")
print()
X1=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
Y1=np.array([[0],[0],[0],[0],[0],[0],[0],[1]])
single_layer_Persepton(X1,Y1)


"""OR Gate single layer persepton"""
print()
print("----OR Gate for single layer persepton----")
print()

X2=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
Y2=np.array([[0],[1],[1],[1],[1],[1],[1],[1]])
single_layer_Persepton(X2,Y2)

"""((x1 ∧ ¬x2) ∨ (¬x1 ∧ x2)) ∧ x3  Gate single layer persepton"""
print()
print("----((x1 ∧ ¬x2) ∨ (¬x1 ∧ x2)) ∧ x3  Gate single layer persepton----")
print()
X3=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
Y3=np.array([[0],[0],[0],[1],[0],[1],[0],[0]])
single_layer_Persepton(X3,Y3)

