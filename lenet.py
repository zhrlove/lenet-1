"""
@author: xiao-data
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
mnist = input_data.read_data_sets("./data/", one_hot=True)
batch_size = 125
learning_rate = 1e-3
display_step = 10
num_steps = 50000
dropout = 0.7

model_path = "./model/model.ckpt"

X = tf.placeholder(tf.float32, [None, 32*32])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
#     return tf.nn.relu(x)
    return tf.maximum(0.1*x,x)  #leaky relu

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def fc(x, W, b):
    x = tf.add(tf.matmul(x, W) , b)
    return tf.maximum(0.1*x,x)
#     return tf.nn.relu(x)
#     return tf.nn.tanh(x)

def ext2d(arr,dim):
    arr = np.array(arr)
    arr = np.reshape(arr, [-1, 28, 28])

    shp = arr.shape
    tmps = []
    i = 0
    for a in arr:
        i += 1
        s1 = arr.shape[1]
        s2 = arr.shape[2]
        tmp = np.zeros([s1+dim*2,s2+dim*2])
        tmp[dim:-dim,dim:-dim] = a
        tmp = np.reshape(tmp, [32*32])
        tmps.append(tmp)
    return np.array(tmps)

def lenet(X, weights, biases, dropout):
    X = tf.reshape(X, [-1, 32, 32, 1])
    conv1 = conv2d(X, weights['conv1'], biases['conv1'])
    pool2 = maxpool2d(conv1)
    conv3 = conv2d(pool2, weights['conv3'], biases['conv3'])
    pool4 = maxpool2d(conv3)
    conv5 = conv2d(pool4, weights['conv5'], biases['conv5'])
    conv5 = tf.contrib.layers.flatten(conv5)
    fc6 = fc(conv5, weights['fc6'],biases['fc6'])
    fc7 = fc(fc6, weights['fc7'],biases['fc7'])
    fc7 = tf.nn.dropout(fc7, dropout)
    return fc7

weights = {
    'conv1' : tf.Variable(tf.random_normal([5, 5, 1, 6])),
    'conv3' : tf.Variable(tf.random_normal([5, 5, 6, 16])), 
    'conv5' : tf.Variable(tf.random_normal([5, 5, 16, 120])),
    'fc6' : tf.Variable(tf.random_normal([120, 84])),
    'fc7' : tf.Variable(tf.random_normal([84, 10]))
}
biases = {
    'conv1' : tf.Variable(tf.random_normal([6])),
    'conv3' : tf.Variable(tf.random_normal([16])),
    'conv5' : tf.Variable(tf.random_normal([120])),
    'fc6' : tf.Variable(tf.random_normal([84])),
    'fc7' : tf.Variable(tf.random_normal([10]))
}

logits = lenet(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
restore_from_model = False
if len(sys.argv) == 2:
    restore_from_model = ('frommodel' == sys.argv[1])
    if restore_from_model:
        num_steps = 10000
        learning_rate = 1e-6
else:
    if len(sys.argv) >2:
        print('use: python3 lenet.py | python3 lenet.py frommodel')

with tf.Session() as sess:
    sess.run(init)
    if restore_from_model :
        print('restored from model')
        saver.restore(sess, model_path)
    for step in range(1, num_steps+1):
        if step == 10000:
            learning_rate = 1e-4
        if step == 20000:
            learning_rate = 1e-5
        if step == 40000:
            learning_rate = 1e-6
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = ext2d(batch_x, 2)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0 or step == 1:
            pre,loss, acc = sess.run([prediction,loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    XX = mnist.test.images[:1000]
    XX = ext2d(XX, 2)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: XX , 
                                      Y: mnist.test.labels[:1000],keep_prob: 1.0}))
    save_path = saver.save(sess, model_path)
