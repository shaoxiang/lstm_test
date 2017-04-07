
# Thanks to Zhao Yu for converting the .ipynb notebook to
# this simplified Python script that I edited a little.

# Note that the dataset must be already downloaded for this script to work, do:
#     $ cd data/
#     $ python download_dataset.py

import tensorflow as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd


checkpoint_dir = ''

x_test = pd.read_csv('x_test_norm_new_6m.csv')
x_train = pd.read_csv('x_train_norm_new.csv')
y_test = pd.read_csv('y_test_6m.csv')
y_train = pd.read_csv('y_train_6m.csv')
numeric_cols = ['open','high','low','close','volume','turnover','time_id']
x_test = x_test[numeric_cols]
x_train = x_train[numeric_cols]

show_acc = 10
save_model = 20000
input_day_length = 2

result_list = []
result_file = open("result.txt","w")


# Load "X" (the neural network's training and testing inputs)
def load_X(x_input):
    X_signals = []
    for i in range((len(x_input)/240)-input_day_length+1):
        X_signals.append(x_input.iloc[240*i:240*(i+input_day_length)].values[:, :])

    return np.array(X_signals)

# Load "X" (the neural network's training and testing inputs)
#def load_X(x_input):
    #X_signals = []
    #for i in range(len(x_input)/240):
        #X_signals.append(x_input.iloc[0+240*i:240+240*i].values[:, :])

    #return np.array(X_signals)


# Load "y" (the neural network's training and testing outputs)
def load_y(y_input):
    # y_input = y_input.iloc[(input_day_length-1):]
    #for i in range(input_day_length-1):
        #y_input = y_input.drop(0)
    # Substract 1 to each output class for friendly 0-based indexing
    return (y_input.values[:])[(input_day_length-1):]
    #return y_input.values[:] - 1
    #return np.array(((y_input.values[:]-1).tolist())[(input_day_length-1):])

# Load "y" (the neural network's training and testing outputs)
#def load_y(y_input):

    #return np.array(y_input.values[:]-1)


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 196 training series
        self.test_data_count = len(X_test)  # 42 testing series
        self.n_steps = len(X_train[0])  # 240 time_steps per series

        # Trainging
        self.learning_rate = 0.01   #0.0025
        self.lambda_loss_amount = 0.001  # 0.0015
        self.training_epochs = 20200#1000000
        self.batch_size = 49

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  #7 # Features count is of 9: three 3D sensors features over time
        self.n_hidden = 400  # nb of neurons inside the neural network 32
        self.n_classes = 3  # Final output classes  6 -> 1
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(feature_mat, config):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # Exchange dim 1 and dim 0
    feature_mat = tf.transpose(feature_mat, [1, 0, 2])
    # New feature_mat's shape: [time_steps, batch_size, n_inputs]

    # Temporarily crush the feature_mat's dimensions
    feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
    # New feature_mat's shape: [time_steps*batch_size, n_inputs]

    # Linear activation, reshaping inputs to the LSTM's number of hidden:
    hidden = tf.nn.relu(tf.matmul(
        feature_mat, config.W['hidden']
    ) + config.biases['hidden'])
    # New feature_mat (hidden) shape: [time_steps*batch_size, n_hidden]

    # Split the series because the rnn cell needs time_steps features, each of shape:
    hidden = tf.split(hidden,config.n_steps, 0)  #tensorflow 1.0.1
    # New hidden's shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_hidden]

    # Define LSTM cell of first hidden layer:
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    #lstm_cell = rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    # Stack two LSTM layers, both layers has the same shape
    #lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    lsmt_layers =tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)
    #lsmt_layers =rnn_cell.MultiRNNCell([lstm_cell] * 2)
    # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
    #outputs, _ = tf.nn.rnn(lsmt_layers, hidden, dtype=tf.float32)\
    outputs, _ = tf.contrib.rnn.static_rnn(lsmt_layers, hidden, dtype=tf.float32)
    # outputs' shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_classes]

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


def one_hot(label):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    label_num = len(label)
    new_label = label.reshape(label_num)  # shape : [sample_num]
    # because max is 5, and we will create 6 columns
    n_values = np.max(new_label) + 1
    return np.eye(n_values)[np.array(new_label, dtype=np.int32)]


if __name__ == "__main__":

    #-----------------------------
    # step1: load and prepare data
    #-----------------------------
    # Those are separate normalised input features for the neural network

    # Output classes to learn how to classify
    LABELS = [
        "fall",
        "rise"
    ]

    X_train = load_X(x_train)
    X_test = load_X(x_test)

    Y_train = one_hot(load_y(y_train))
    Y_test = one_hot(load_y(y_test))
    #Y_train = load_y(y_train)
    #Y_test =  load_y(y_test)
    #-----------------------------------
    # step2: define parameters for model
    #-----------------------------------
    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    #print(X_test.shape, y_test.shape,np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    #------------------------------------------------------
    # step3: Let's get serious and build the neural network
    #------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits= pred_Y, labels=Y)) + l2
    #cost = tf.reduce_mean(
        #tf.nn.sigmoid_cross_entropy_with_logits(logits= pred_Y, labels=Y)) + l2

    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    #optimizer = tf.train.AdadeltaOptimizer(
        #learning_rate=config.learning_rate).minimize(accuracy)

    #optimizer = tf.train.MomentumOptimizer(
        #learning_rate=config.learning_rate,momentum = 0.9).minimize(cost)

    #optimizer = tf.train.GrandientDescentOptimizer(
        #learning_rate=config.learning_rate).minimize(cost)

    #optimizer = tf.train.RMSPropOptimizer(
        #learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    #--------------------------------------------
    # step4: Hooray, now train the neural network
    #--------------------------------------------
    # Note that log_device_placement can be turned ON but will cause console spam.
    sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    tf.global_variables_initializer().run() #  tf.global_variables_initializer

    best_accuracy = 0.0

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #print ckpt
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, './'+ckpt.model_checkpoint_path)
        print "use saved model"
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: Y_train[start:end]})

        if ((i+1)%show_acc == 0):
            # Test completely at every epoch: calculate accuracy
            pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
                                                X: X_test, Y: Y_test})
            print("traing iter: {},".format(i)+\
                " test accuracy : {},".format(accuracy_out)+\
                " loss : {}".format(loss_out))
            best_accuracy = max(best_accuracy, accuracy_out)

            result_list.append([i,accuracy_out,loss_out])

	if ((i+1)%save_model == 0):
            saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i + 1)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")
    result_file.write(str(result_list))
    result_file.close()
    #------------------------------------------------------------------
    # step5: Training is good, but having visual insight is even better
    #------------------------------------------------------------------
    # The code is in the .ipynb

    #------------------------------------------------------------------
    # step6: And finally, the multi-class confusion matrix and metrics!
    #------------------------------------------------------------------
    # The code is in the .ipynb
