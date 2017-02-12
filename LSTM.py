import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import math


# This function generates an array with [t x_t]
def generateTimeSeries(nSample, exponential=True, sine=True, random=True, timeStamp=False):
    t = np.arange(0, nSample, 1)
    trend1 = np.exp(2 / (nSample) * t) * exponential
    trend2 = np.sin(t / 80) * sine
    randomComp1 = np.random.randn(nSample) * np.sin(t * np.random.rand(nSample) / 2) * random
    data = np.array(trend1 + trend2 + randomComp1)

    if (timeStamp):
        data = np.array([t, data])

    return np.array(data).T


# This function generates an array [[x(t), ... x(t+timesteps)], [x(t +timesteps+1)]
# It is a sliding window of the timeSteps. Data should be only the timeseries without the timestamp.
def generateData(data, timeSteps=2, trainPct=0.6):
    print(len(data))
    nSample = len(data)
    nTrain = int(nSample * trainPct)
    nTest = nSample - nTrain

    nbSequence = nSample - timeSteps - 1

    sequenceData = []
    sequenceLabel = []
    for i in range(0, nbSequence):
        sequenceData.append(data[i:i + timeSteps])
        sequenceLabel.append(data[i+1:i + timeSteps+1])
        #sequenceLabel.append(data[i + timeSteps:i + timeSteps + 1])

    sequenceData = np.array(sequenceData)
    sequenceLabel = np.array(sequenceLabel)

    trainData = sequenceData[:nTrain, :]
    trainLabel = sequenceLabel[:nTrain]
    testData = sequenceData[nTrain + 1:nSample, :]
    testLabel = sequenceLabel[nTrain + 1:nSample]

    return trainData, trainLabel, testData, testLabel


# Producing the data in batches
def batchData(trainData, trainLabel, batchSize, timeSteps):
    nbSample = len(trainLabel)
    batchInput = []
    batchLabel = []

    # trainData = np.array(trainData)
    # trainLabel = np.array(trainLabel)

    for i in range(0, batchSize):
        index = np.random.choice(range(0, nbSample), 1)
        batchInput.append(trainData[index, :][0])
        batchLabel.append(trainLabel[index,:][0])

    batchInput =np.reshape(np.array(batchInput), (batchSize, timeSteps, 1))
    batchLabel =np.reshape(np.array(batchLabel), (batchSize*timeSteps, 1))

    return batchInput, batchLabel


# Input needs to reshape to  shape [batch, timestep, dimension]
def lstm(trainData, paramsLSTM, paramsMLP, batchSize):
    nbHidden = paramsLSTM["nbHidden"]
    nbLayers = paramsLSTM["nbLayers"]
    timeSteps = paramsLSTM["timeSteps"]
    dropout = paramsLSTM["dropout"]

    # Setting up memory cell layers
    cell = tf.nn.rnn_cell.LSTMCell(nbHidden, forget_bias=1.0)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
    multcell = tf.nn.rnn_cell.MultiRNNCell([cell] * nbLayers)
    initState = cell.zero_state(batchSize, tf.float32)

    # unrolling the network depends on the input size. if the input size if of time t0-tn
    # then the BPTT will be truncated at after the network is unrolled for t0-tn
    output, lastState = tf.nn.dynamic_rnn(multcell, trainData, dtype=tf.float32)
    output = tf.reshape(output, [-1, nbHidden])

    W1 = paramsMLP["W1"]
    W2 = paramsMLP["W2"]
    W3 = paramsMLP["W3"]
    B1 = paramsMLP["B1"]
    B2 = paramsMLP["B2"]
    B3 = paramsMLP["B3"]

    # MLP Layer
    # h1 = tf.nn.relu(tf.matmul(output, W1) + B1)
    # h1 = tf.nn.dropout(h1, dropout)
    #
    # h2 = tf.nn.relu(tf.matmul(h1, W2) + B2)
    # h2 = tf.nn.dropout(h2, dropout)

    prediction = tf.matmul(output, W1) + B1

    return prediction

TIMESTEPS = 200
EPOCH = 100
BATCHSIZE = 100



# Defining Placeholders
xpredict = tf.placeholder(tf.float32, [None, TIMESTEPS, 1], name='input_placeholder')
x = tf.placeholder(tf.float32, [None, TIMESTEPS, 1], name='input_placeholder')
y = tf.placeholder(tf.float32, [BATCHSIZE*TIMESTEPS, 1], name='output_placeholder')
dropout = tf.placeholder(tf.float32)


# Setting up the variable:
paramsLSTM = {
    "nbHidden": 500,
    "nbLayers": 8,
    "timeSteps": TIMESTEPS,
    "dropout": 1.0
}

paramsMLP = {
    "W1": tf.get_variable("weights1", (paramsLSTM["nbHidden"], 1), initializer=tf.random_normal_initializer(0, 0.01)),
    "W2": tf.get_variable("weights2", (100, 50), initializer=tf.random_normal_initializer(0, 0.01)),
    "W3": tf.get_variable("weights3", (50, 1), initializer=tf.random_normal_initializer(0, 0.01)),
    "B1": tf.get_variable("bias1", (100), initializer=tf.constant_initializer(0.5)),
    "B2": tf.get_variable("bias2", (50), initializer=tf.constant_initializer(0.5)),
    "B3": tf.get_variable("bias3", (1), initializer=tf.constant_initializer(0.5)),
}


data = generateTimeSeries(5000, exponential=True, sine=True, random=False, timeStamp=False)
trainData, trainLabel, testData, testLabel = generateData(data, timeSteps=TIMESTEPS, trainPct=0.6)

trainDataPrediction = np.reshape(trainData,(3000,TIMESTEPS,1))


ymodel = lstm(x, paramsLSTM, paramsMLP, BATCHSIZE)
loss = tf.reduce_sum((ymodel - y)**2)
opt = tf.train.AdamOptimizer().minimize(loss)



with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    for i in range(0, EPOCH):
        batchInput, batchLabel = batchData(trainData, trainLabel, BATCHSIZE, TIMESTEPS)
        _, Loss = sess.run([opt, loss], feed_dict={x: batchInput, y: batchLabel})
        print(Loss)

    prediction = sess.run(ymodel, feed_dict={x: trainDataPrediction})


p=[]
for i in range(0,3000):
    p.append(prediction[i*TIMESTEPS])


# Data Visualization
plt.plot(data, color = 'b')
plt.plot(p, color = 'r')
# plt.xlabel('Time')
# plt.ylabel('y')
plt.show()
