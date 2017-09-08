#!usr/bin/python3
import tensorflow as tf
'''
 input > weight > hidden layer1(activation fn) > weights> hidden layer2
 (activation fn) > weights > output layer
 compare output to indt=ended output> cost fn
 optimiser > minimise cost(AdamOptimiser ... SGD, ADaGrad)
 back propagation 
 feed forward + backprop = epoch
'''
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

# 10 classes, 0-9
'''
0 = 0, 1 = 1, 2 = 2,....
one_hot gives 0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
'''
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
# there are three layers(1 + 2 hidden)
n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neuralNetworkModel(data):
	# (inputData * weights) + biases
	hidden1layer = {'weights' : tf.Variable(tf.random_normal([784, n_nodes_hl1])), 
					'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
	# creates a tensor of weights 
	hidden2layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
					'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden3layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
					'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}

	outputlayer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 
					'biases' : tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden1layer['weights']), hidden1layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden2layer['weights']), hidden2layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden3layer['weights']), hidden3layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, outputlayer['weights']) + outputlayer['biases']
	return output

def trainNeuralNetwork(x):
	prediction = neuralNetworkModel(x)
	cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	hmEpochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hmEpochs):
			epochLoss = 0
			for _ in range(int(mnist.train.num_examples / batch_size)):
				epochX, epochY = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x : epochX, y : epochY})
				epochLoss += c
			print('Epoch', epoch, 'completed out of ', hmEpochs, 'loss:', epochLoss)
		
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy: ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

trainNeuralNetwork(x)