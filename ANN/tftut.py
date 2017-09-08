import tensorflow as tf
#constants to be multiplied
x1 = tf.constant(5)
x2 = tf.constant(6)
# multiply x1 nad x2 using multiply() function of tensorflow
result = tf.multiply(x1, x2)
print(result)

# To run session and graph
"""
sess = tf.Session()
print(sess.run(result))
sess.close()
"""
# OR
# do the following
with tf.Session() as sess:
	output = sess.run(result)
	print(output)