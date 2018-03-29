import os

import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle


FTRAIN = '/Users/Tiantian/Documents/CNN/face/training.csv'
FTEST = '/Users/Tiantian/Documents/CNN/face/test.csv'

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))


#####################################
#####################################
#####################################
# Part I: build models (8)
#####################################
#####################################
#####################################



#####################################
# First model: a single hidden layer
#####################################
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )

net1.fit(X, y)
# epoch    trn loss    val loss    trn/val  dur
# -------  ----------  ----------  ---------  -----
# ...
# 398     0.00255     0.00352    0.72423  0.60s
# 399     0.00245     0.00330    0.74375  0.50s
# 400     0.00233     0.00316    0.73810  0.51s
# how good is a validation loss of 0.00316?
# root MSE
np.sqrt(0.00316) * 48
# 2.6982661099305978
# compare to Kaggle leaderboard as low as 1.28236, not good

# test out
train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
plt.plot(train_loss, linewidth=1, label="train")
plt.plot(valid_loss, linewidth=1, label="valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3, 1e-2)
plt.yscale("log")
plt.show()
# The plot can provide an indication of useful things about
# the training of the model, such as:

# It’s speed of convergence over epochs (slope).
# Whether the model may have already converged (plateau of the line).
# Whether the mode may be over-learning the training data (inflection for validation line).
# And more.

# If the model is overfitting the graph will show
# great performance on the training data and
# poor performance on the test data.

# see how predictions work: pick samples from test set
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X, _ = load(test=True)
y_pred = net1.predict(X)

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

plt.show()

#####################################
# Second model: convolutions
#####################################
def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=1000,
    verbose=1,
    )

X, y = load2d()  # load 2-d data
net2.fit(X, y)

#   epoch    trn loss    val loss    trn/val  dur
# -------  ----------  ----------  ---------  ------
#       1     0.02881     0.01000    2.88144  60.68s
#       2     0.01008     0.00856    1.17735  55.41s
#       3     0.00847     0.00792    1.06939  59.59s
#       4     0.00788     0.00739    1.06735  52.88s
#       5     0.00743     0.00698    1.06448  55.41s
#       6     0.00707     0.00664    1.06408  61.09s
# Training for 1000 epochs will take a while.
# pickle the trained model so that we can load it back in the future:
import cPickle as pickle
with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)


np.sqrt(0.001566) * 48
# 1.8994904579913006

# compare single hidden layer model with CNN model
sample1 = load(test=True)[0][6:7]
sample2 = load2d(test=True)[0][6:7]
y_pred1 = net1.predict(sample1)[0]
y_pred2 = net2.predict(sample2)[0]

fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(sample1[0], y_pred1, ax)
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(sample1[0], y_pred2, ax)
plt.show()

# plot the learning curves of the first and the second network
train_loss1 = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss1 = np.array([i["valid_loss"] for i in net1.train_history_])
train_loss2 = np.array([i["train_loss"] for i in net2.train_history_])
valid_loss2 = np.array([i["valid_loss"] for i in net2.train_history_])
plt.plot(train_loss1, 'b', linewidth=1, linestyle = 'solid', label="net1 train")
plt.plot(valid_loss1, 'g', linewidth=1, linestyle = 'solid', label="net1 valid")
plt.plot(train_loss2, 'b', linewidth=1, linestyle = 'dashed', label="net2 train")
plt.plot(valid_loss2, 'g', linewidth=1, linestyle = 'dashed', label="net2 valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3, 1e-2)
plt.yscale("log")
plt.show()

#####################################
# Third model: data augmentation (apply transformation, add noise, etc.)
#####################################
# X, y = load2d()
X_flipped = X[:, :, :, ::-1]
# X_flipped[0,0,0,-1] = X[0,0,0,0]

# plot two images:
fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(X[1], y[1], ax)
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(X_flipped[1], y[1], ax)
plt.show()

flip_indices = [
    (0, 2), (1, 3),
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (22, 24), (23, 25),
    ]
df = read_csv(os.path.expanduser(FTRAIN))
for i, j in flip_indices:
    print("# {} -> {}".format(df.columns[i], df.columns[j]))

from nolearn.lasagne import BatchIterator
class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

net3 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    max_epochs=3000,
    verbose=1,
    )

net3.fit(X, y)
with open('net3.pickle', 'wb') as f:
    pickle.dump(net3, f, -1)

#   epoch    trn loss    val loss    trn/val  dur
# -------  ----------  ----------  ---------  ------
#       1     0.12692     0.07376    1.72061  61.55s
#       2     0.03026     0.00940    3.21836  60.36s
#       3     0.00786     0.00640    1.22843  55.50s
#       4     0.00632     0.00574    1.10107  58.04s
#       5     0.00592     0.00551    1.07363  58.75s
#       6     0.00574     0.00535    1.07305  57.28s

# compare single hidden layer model with CNN model
#sample1 = load(test=True)[0][6:7]
sample2 = load2d(test=True)[0][6:7]
y_pred2 = net2.predict(sample2)[0]
y_pred3 = net3.predict(sample2)[0]

fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(sample2[0], y_pred2, ax)
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(sample2[0], y_pred3, ax)
plt.show()

# plot the learning curves of the second and the third network
train_loss2 = np.array([i["train_loss"] for i in net2.train_history_])
valid_loss2 = np.array([i["valid_loss"] for i in net2.train_history_])
train_loss3 = np.array([i["train_loss"] for i in net3.train_history_])
valid_loss3 = np.array([i["valid_loss"] for i in net3.train_history_])
plt.plot(train_loss2, 'b', linewidth=1, linestyle = 'dashed', label="net2 train")
plt.plot(valid_loss2, 'g', linewidth=1, linestyle = 'dashed', label="net2 valid")
plt.plot(train_loss3, 'b', linewidth=1, linestyle = 'solid', label="net3 train")
plt.plot(valid_loss3, 'g', linewidth=1, linestyle = 'solid', label="net3 valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3, 1e-2)
plt.yscale("log")
plt.show()

#####################################
# 4th model: Change learning rate and momentum
#####################################
import theano

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

net4 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    # batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=1000,
    verbose=1,
    )
net4.fit(X, y)
with open('net4.pickle', 'wb') as f:
    pickle.dump(net4, f, -1)

#   epoch    trn loss    val loss    trn/val  dur
# -------  ----------  ----------  ---------  ------
#       1     0.05702     0.01099    5.18779  62.59s
#       2     0.00892     0.00699    1.27504  58.20s
#       3     0.00671     0.00587    1.14366  60.84s
#       4     0.00585     0.00528    1.10760  58.74s
#       5     0.00536     0.00492    1.09022  56.68s
#       6     0.00507     0.00469    1.07959  53.93s
# 'training is happening much faster now!'  ???????????
#####################################
# 5th model: change learning rate and momentum with data augmentation
#####################################
net5 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=1000,
    verbose=1,
    )
net5.fit(X, y)
with open('net5.pickle', 'wb') as f:
    pickle.dump(net5, f, -1)

#   epoch    trn loss    val loss    trn/val  dur
# -------  ----------  ----------  ---------  ------
#       1     0.05578     0.01123    4.96544  61.23s
#       2     0.00859     0.00685    1.25430  59.30s
#       3     0.00657     0.00571    1.15083  56.20s
#       4     0.00573     0.00511    1.12182  53.82s
#       5     0.00523     0.00478    1.09614  54.47s
#       6     0.00495     0.00453    1.09108  62.12s

# plot the learning curves of the 4th and the 5th network
train_loss4 = np.array([i["train_loss"] for i in net4.train_history_])
valid_loss4 = np.array([i["valid_loss"] for i in net4.train_history_])
train_loss5 = np.array([i["train_loss"] for i in net5.train_history_])
valid_loss5 = np.array([i["valid_loss"] for i in net5.train_history_])
plt.plot(train_loss4, 'b', linewidth=1, linestyle = 'dashed', label="net4 train")
plt.plot(valid_loss4, 'g', linewidth=1, linestyle = 'dashed', label="net4 valid")
plt.plot(train_loss5, 'b', linewidth=1, linestyle = 'solid', label="net5 train")
plt.plot(valid_loss5, 'g', linewidth=1, linestyle = 'solid', label="net5 valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3, 1e-2)
plt.yscale("log")
plt.show()


#####################################
# 6th model: dropout regularization with changeing learning rate
#            and momentum and data augmentation
#####################################
net6 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,  # !
    hidden4_num_units=500,
    dropout4_p=0.5,  # !
    hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=3000,
    verbose=1,
    )

import sys
sys.setrecursionlimit(10000)

#X, y = load2d()
net6.fit(X, y)
with open('net6.pickle', 'wb') as f:
    pickle.dump(net6, f, -1)

#   epoch    trn loss    val loss    trn/val  dur
# -------  ----------  ----------  ---------  ------
#       1     0.07271     0.04276    1.70035  66.79s
#       2     0.01710     0.02690    0.63570  67.37s
#       3     0.01125     0.01918    0.58659  69.50s
#       4     0.00872     0.01512    0.57675  70.08s
#       5     0.00783     0.01257    0.62294  71.57s
#       6     0.00731     0.01148    0.63679  67.07s
#       7     0.00717     0.01058    0.67779  72.41s

from sklearn.metrics import mean_squared_error
print mean_squared_error(net6.predict(X), y)
# prints something like 0.0010073791


#####################################
# 7th model: make net larger with dropout regularization,
#            and changeing learning rate and momentum
#            and data augmentation
#####################################
net7 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,  # !
    hidden4_num_units=1000,  ### !
    dropout4_p=0.5,  # !
    hidden5_num_units=1000,  ### !
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=3000,
    verbose=1,
    )

#X, y = load2d()
net7.fit(X, y)
with open('net7.pickle', 'wb') as f:
    pickle.dump(net7, f, -1)



#####################################
# 8th model: increase epochs
#            and make net larger with dropout regularization,
#            and changeing learning rate and momentum
#            and data augmentation
#####################################

net8 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,  # !
    hidden4_num_units=1000,  ### !
    dropout4_p=0.5,  # !
    hidden5_num_units=1000,  ### !
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=10000,   ###### !
    verbose=1,
    )

#X, y = load2d()
net8.fit(X, y)
with open('net8.pickle', 'wb') as f:
    pickle.dump(net8, f, -1)


# compare the nets we trained so for and their respective train and validation errors:
#  Name  |   Description         |  Epochs  |  Train loss  |  Valid loss
# -------|-----------------------|----------|--------------|--------------
#  net1  |  single hidden        |     400  |    0.002244  |    0.003255
#  net2  |  convolutions         |    1000  |    0.001079  |    0.001566
#  net3  |  augmentation         |    3000  |    0.000678  |    0.001288
#  net4  |  mom + lr adj         |    1000  |    0.000496  |    0.001387
#  net5  |  net4 + augment       |    2000  |    0.000373  |    0.001184
#  net6  |  net5 + dropout       |    3000  |    0.001306  |    0.001121
#  net7  |  net6 + larger net    |    3000  |    0.001195  |    0.001087
#  net8  |  net7 + epochs        |   10000  |    0.000760  |    0.000787




#####################################
#####################################
#####################################
# Part II: build specialists (6)
#####################################
#####################################
#####################################

#####################################
# 9th model: apply early stopping
#####################################
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

net9 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,  # !
    hidden4_num_units=1000,  ### !
    dropout4_p=0.5,  # !
    hidden5_num_units=1000,  ### !
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),    ########## !
        ],
    max_epochs=10000,   ###### !
    verbose=1,
    )


# define 6 specialists
SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
    ]


from collections import OrderedDict
from sklearn.base import clone

# fit the specialist models in a new function
def fit_specialists():
    specialists = OrderedDict()

    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        X, y = load2d(cols=cols)

        model = clone(net9)
        model.output_num_units = y.shape[1]
        model.batch_iterator_train.flip_indices = setting['flip_indices']
        # set number of epochs relative to number of training examples:
        model.max_epochs = int(1e7 / y.shape[0])
        if 'kwargs' in setting:
            # an option 'kwargs' in the settings list may be used to
            # set any other parameter of the net:
            vars(model).update(setting['kwargs'])

        print("Training model for columns {} for {} epochs".format(
            cols, model.max_epochs))
        model.fit(X, y)
        specialists[cols] = model

    with open('net-specialists.pickle', 'wb') as f:
        # we persist a dictionary with all models:
        pickle.dump(specialists, f, -1)

fit_specialists()
# Training model for columns ('left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y') for 4672 epochs

#   epoch    trn loss    val loss    trn/val  dur
# -------  ----------  ----------  ---------  ------
#       1     0.10662     0.08216    1.29769  98.43s
#   epoch    trn loss    val loss    trn/val  dur
# -------  ----------  ----------  ---------  ------
#       1     0.10662     0.08216    1.29769  98.43s
#       2     0.07222     0.07548    0.95677  93.20s
#       2     0.07222     0.07548    0.95677  93.20s
#       3     0.06919     0.07284    0.94986  91.87s
#       3     0.06919     0.07284    0.94986  91.87s
#       4     0.06799     0.06754    1.00665  76.23s
#       4     0.06799     0.06754    1.00665  76.23s
#       5     0.06745     0.06915    0.97545  98.62s
#       5     0.06745     0.06915    0.97545  98.62s
#       6     0.06717     0.06819    0.98508  86.82s
#       6     0.06717     0.06819    0.98508  86.82s
# Training model for columns ('nose_tip_x', 'nose_tip_y') for 4672 epochs
# Training model for columns ('mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x', 'mouth_center_top_lip_y') for 4672 epochs
# ......
# Neural Network with 16561502 learnable parameters


#####################################
# Supervised pre-training
#####################################
def fit_specialists(fname_pretrain=None):
    if fname_pretrain:  # !
        with open(fname_pretrain, 'rb') as f:  # !
            net_pretrain = pickle.load(f)  # !
    else:  # !
        net_pretrain = None  # !

    specialists = OrderedDict()

    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        X, y = load2d(cols=cols)

        model = clone(net9)
        model.output_num_units = y.shape[1]
        model.batch_iterator_train.flip_indices = setting['flip_indices']
        model.max_epochs = int(4e6 / y.shape[0])
        if 'kwargs' in setting:
            # an option 'kwargs' in the settings list may be used to
            # set any other parameter of the net:
            vars(model).update(setting['kwargs'])

        if net_pretrain is not None:  # !
            # if a pretrain model was given, use it to initialize the
            # weights of our new specialist model:
            model.load_params_from(net_pretrain)  # !

        print("Training model for columns {} for {} epochs".format(
            cols, model.max_epochs))
        model.fit(X, y)
        specialists[cols] = model

    with open('net-specialists.pickle', 'wb') as f:
        # this time we're persisting a dictionary with all models:
        pickle.dump(specialists, f, -1)



#####################################
# Predict on Test data and output to CSV
#####################################
def predict(fname_specialists = 'net-specialists.pickle'):
    with open(fname_specialists, 'rb') as f:
        specialists = pickle.load(f)

    X = load2d(test = True)[0]
    y_pred = np.empty((X.shape[0], 0))

    for model in specialists.values():
        y_pred1 = model.predict(X)
        y_pred = np.hstack([y_pred, y_pred1])

    columns = ()
    for cols in specialists.keys():
        columns += cols

    y_pred2 = y_pred * 48 + 48
    y_pred2 = y_pred2.clip(0, 96)
    df = DataFrame(y_pred2, columns = columns)

    lookup_table = read_csv(os.path.expanduser(FLOOKUP))
    values = []

    for index, row in lookup_table.iterrows():
        values.append((row['RowID'], df.ix[row.ImageId - 1][row.FeatureName]))
    now_str = datetime.now().isoformat().replace(':', '-')
    submission = DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index = False)
    print('Wrote {}'.format(filename))


#####################################
# Plot specialists loss-epoch curves
#####################################
def rebin(a, newshape):
    assert len(a.shape) == len(newshape)
    slices = [slice(0, old, float(old)/new) for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')
    return a[tuple(indices)]

def plot_learning_curves(fname_specialists = 'net-specialists.pickle'):
    with open(fname_specialists, 'rd') as f:
        models = pickle.load(f)

    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(1,1,1)
    ax.set_color_cycle(['c','c','m','m','y','y','k','k','g','g','b','b'])

    train_losses = []
    valid_losses = []

    for model_numer, (cg, model) in enumerate(models.items(), 1):
        train_loss = np.array([i['train_loss'] for i in model.train_history_])
        valid_loss = np.array([i['valid_loss'] for i in model.train_history_])
        train_loss = np.sqrt(train_loss) * 48
        valid_loss = np.sqrt(valid_loss) * 48

        train_loss = rebin(train_loss, (100, ))
        valid_loss = rebin(valid_loss, (100, ))

        ax.plot(train_loss, linestyle = 'dashed', linewidth = 2, alpha = 0.6)
        ax.plot(valid_loss, label='{} ({})'.format(cg[0], len(cg)), linewidth=2)
        ax.set_xticks([])

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    weights = np.array([m.output_num_units for m in models.values()], dtype = float)
    weights /= weights.sum()
    mean_valid_loss = (np.vstack(valid_losses) * weights.reshape(-1, 1)).sum(axis = 0)
    ax.plot(mean_valid_loss, color = 'r', label = 'mean', linewidth = 3, alpha = 0.8)
    ax.legend()
    ax.set_ylim((1.0, 4.0))
    ax.grid()
    plt.ylabel('RMSE')
    plt.show()
