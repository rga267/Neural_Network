#Feed Forward Nueral Network with 2 fully connected layers for classification


import numpy as np 
import time

#variables
n_hidden_layer = 10
n_input = 10

#outputs
n_output = 10
n_sample_size = 500


#hyperparameters
learning_rate = 0.01
momentum = 0.9

#seeding to ensure same random pool
np.random.seed(0)

#first layer activation function
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

#second layer activation function--used to make predictions more accurate/ stronger gradient--0 centered(-1,1)
def tanh_prime(x):
	return 1 - np.tanh(x)**2

#input, transpose, layer 1, layer 2, weight bias for layer 1, weight bias for layer 2
def train(x, t, V, W, bv, bw):

	#forward propagation--matrix multiply + biases
	A = np.dot(x, V) + bv
	Z = np.tanh(A)

	B = np.dot(Z, W) + bw
	Y = sigmoid(B)

	#backward propagation
	Ew = Y - t
	Ev = tanh_prime(A) * np.dot(W, Ew)


	#predict our loss
	dW = np.outer(Z, Ew)
	dV = np.outer(x, Ev)


	#cross entropy--cost function/loss function--could use mean squared error
	loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))

	return loss, (dV, dW, Ev, Ew)



def predict(x, V, W, bv, bw):
	A = np.dot(x, V) + bv
	B = np.dot(np.tanh(A), W) + bw
	return (sigmoid(B) > 0.5).astype(int)



#create layers
V = np.random.normal(scale=0.1, size=(n_input, n_hidden_layer))
W = np.random.normal(scale=0.1, size=(n_hidden_layer, n_output))

bv = np.zeros(n_hidden_layer)
bw = np.zeros(n_output)

params = [V, W, bv, bw]


#generate our data
X = np.random.binomial(1, 0.5, (n_sample_size, n_input))
T = X ^ 1

#Training Time
for epoch in range(100):
	err = []
	upd = [0]*len(params)

	t0 = time.process_time()

	#for each data point, update our weights 

	for i in range(X.shape[0]):
		loss, grad = train(X[i], T[i], *params)

		#update loss
		for j in range(len(params)):
			params[j] -= upd[j]

		for j in range(len(params)):
			upd[j] = learning_rate * grad[j] + momentum * upd[j]

		err.append(loss)


	print('Epoch: %d, Loss: %.8f, Time: %.4fs' %(epoch, np.mean(err), time.process_time()-t0))



#try to predict this
x = np.random.binomial(1, 0.5, n_input)
print('XOR prediction')
print(x)
print(predict(x, *params))







