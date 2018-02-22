import numpy as np
import matplotlib.pyplot as plt

n_examples, hidden_layer_dim = 100, 100
input_dim = 1000
X = np.random.randn(n_examples, input_dim) # 100 examples of 1000 points
n_layers = 20
layer_dim = [hidden_layer_dim] * n_layers # each one has 100 neurons

hs = [X]
hs_not_batchnormed = [X] # saves the results before we do batchnorm, because we need this in the backward pass.
zs = [X]
ws = []
gamma, beta = 1, 0
# the forward pass
for i in np.arange(n_layers):
	h = hs[-1] # get the input into this hidden layer
	W = np.random.normal(size = (layer_dim[i], h.shape[0])) * 0.01 # weight init: gaussian around 0
	z = np.dot(W, h)
	h_out = z * (z > 0)
	# save the not batchnormmed part for backprop
	hs_not_batchnormed.append(h_out)
	# apply batch normalization
	h_out = (h_out - np.mean(h_out, axis = 0)) / np.std(h_out, axis = 0)
	# scale and shift
	h_out = gamma * h_out + beta
	ws.append(W)
	zs.append(z)
	hs.append(h_out)



dLdh = 0.01 * np.random.randn(hidden_layer_dim, input_dim) # random incoming grad into our last layer
h_grads = [dLdh]
w_grads = []

# the backwards pass
for i in np.flip(np.arange(1, n_layers), axis = 0):
	# get the incoming gradient
	incoming_loss_grad = h_grads[-1]
	# backprop through the batchnorm layer
	#the y_i is the restult of batch norm, so h_out or hs[i]
	dldx_hat = incoming_loss_grad * gamma
	dldvar = np.sum(dldx_hat * (hs_not_batchnormed[i] - np.mean(hs_not_batchnormed[i], axis = 0)) * -.5 * np.power(np.var(hs_not_batchnormed[i], axis = 0), -1.5), axis = 0)
	dldmean = np.sum(dldx_hat * -1/np.std(hs_not_batchnormed[i], axis = 0), axis = 0) + dldvar * -2 * (hs_not_batchnormed[i] - np.mean(hs_not_batchnormed[i], axis = 0))/hs_not_batchnormed[i].shape[0]
	# the following is dL/hs_not_batchnormmed[i] (aka dL/dx_i) in the paper!
	dldx = dldx_hat * 1/np.std(hs_not_batchnormed[i], axis = 0) + dldvar * -2 * (hs_not_batchnormed[i] - np.mean(hs_not_batchnormed[i], axis = 0))/hs_not_batchnormed[i].shape[0] + dldmean/hs_not_batchnormed[i].shape[0]
	# although we don't need it for this demo, for completeness we also compute the derivatives with respect to gamma and beta. 
	dldgamma = incoming_loss_grad * hs[i]
	dldbeta = np.sum(incoming_loss_grad)
	# now incoming_loss_grad should be replaced by that backpropped result
	incoming_loss_grad = dldx
	# backprop through the relu
	print(incoming_loss_grad.shape)
	dLdz = incoming_loss_grad * (zs[i] > 0)
	# get the gradient dL/dh_{i-1}, this will be the incoming grad into the next layer
	h_grad = ws[i-1].T.dot(dLdz)
	# get the gradient of the weights of this layer (dL/dw)
	weight_grad = dLdz.dot(hs[i-1].T)
	h_grads.append(h_grad)
	w_grads.append(weight_grad)

# plot the resulting activatiosn
for i, activation in enumerate(hs):
	fig = plt.figure()
	num_bins = 50
	print('variance of linear units before relu is {}'.format(np.var(zs[i].ravel())))
	print('variance of activations is {}'.format(np.var(activation.ravel())))
	ratio = np.var(zs[i].ravel()) / np.var(activation.ravel())
	print('ratio is {}'.format(ratio))
	n, bins, patches = plt.hist(activation.ravel(), num_bins, normed=1, facecolor='green', alpha=0.5)
	plt.title('Activation {}, var: {}'.format(i, np.var(activation.ravel())))
	plt.xlabel('Activation Value')
	plt.ylabel('Number of Activations')
	# Tweak spacing to prevent clipping of ylabel
	plt.subplots_adjust(left=0.15)
	plt.savefig('activation-plots/act-{}.png'.format(i))	
	plt.ticklabel_format(axis='x',style='sci',scilimits=(1,4))


w_grads = list(reversed(w_grads))
for i, grad in enumerate(w_grads):
	fig = plt.figure()
	num_bins = 50
	n, bins, patches = plt.hist(grad.ravel(), num_bins, normed=1, facecolor='green', alpha=0.5)
	plt.title('Grads @ layer {}, var {}'.format(n_layers - i, np.var(grad.ravel())))
	plt.xlabel('Gradient Value')
	plt.ylabel('Number of Gradients')
	# Tweak spacing to prevent clipping of ylabel
	plt.subplots_adjust(left=0.15)
	plt.savefig('gradient-plots/grad-{}.png'.format(i))	
	plt.ticklabel_format(axis='x',style='sci',scilimits=(1,4))	

# first_hidden_activation = hs[1]
# fig = plt.figure()
# num_bins = 50
# # the histogram of the data
# n, bins, patches = plt.hist(first_hidden_activation.ravel(), num_bins, normed=1, facecolor='green', alpha=0.5)
# plt.xlabel('Activation Value')
# plt.ylabel('Number of Activations')

# # Tweak spacing to prevent clipping of ylabel
# plt.subplots_adjust(left=0.15)
# plt.savefig('test-model-1.png')
plt.show()

