# file kfkd.py
import os

import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = '/Users/Tiantian/Documents/CNN/face/training.csv'
FTEST = '/Users/Tiantian/Documents/CNN/face/test.csv'

def load(test=False, cols=None):
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep = ' '))

    if cols:
        df = df[list(cols) + ['Image']]

    print(df.count())  # print(df.info())
    df = df.dropna()   # 2140 rows left

    X = np.vstack(df['Image'].values) / 255
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values  # 2140 rows x 30 columns
        # np.max(y)
        #     95.808983121499992
        # np.min(y)
        #     3.8262430562800001
        y = (y - 48) / 48   # scale target coordinates to [-1, 1]
    else:
        y = None
    return X, y

X, y = load()
print('X.shape == {}; X.min == {:.3f}; X.max == {:.3f}'.format(
    X.shape, X.min(), X.max()))
print('y.shape == {}; y.min == {:.3f}; y.max == {:.3f}'.format(
    y.shape, y.min(), y.max()))



# add to kfkd.py
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],
    input_shape = (None, 9216),
    hidden_num_units = 100,
    output_nonlinearity = None,
    output_num_units = 30,

    update = nesterov_momentum,
    update_learning_rate = 0.01,
    update_momentum = 0.9,

    regression = True,
    max_epochs = 400,
    verbose = 1,
)
net1.fit(X, y)
#   epoch    trn loss    val loss    trn/val  dur
# -------  ----------  ----------  ---------  -----
#       1     0.14927     0.05203    2.86871  0.46s
#     ...         ...         ...        ...    ...
#     399     0.00255     0.00318    0.80101  0.48s
#     400     0.00255     0.00318    0.80177  0.47s

RMSE = np.sqrt(0.00318) * 48
print('The loss (RMSE) on the validation set is: %f, epoch = %d' % (RMSE, net1.max_epochs))

# plot loss-epoch curves
train_loss = np.array([i['train_loss'] for i in net1.train_history_])
valid_loss = np.array([i['valid_loss'] for i in net1.train_history_])
plt.plot(train_loss, linewidth = 2, c = 'b', label = 'train')
plt.plot(valid_loss, linewidth = 2, c = 'g', label = 'valid')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-3, 1e-2)
plt.yscale('log')
plt.show()

# test on Test data
X, _ = load(test = True) # X.shpae = (1783, 9216)
y_pred = net1.predict(X)

def plot_sample(x, y, axis):
    img = x.reshape(96, 96) # image: 2140 * 9216
    axis.imshow(img, cmap = 'gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, color='r', s=10, marker='x', alpha=1)

fig = plt.figure(figsize = (6,6))
fig.subplots_adjust(
    left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i+1)
    plot_sample(X[i], y_pred[i], ax)

plt.show()




# LeNet5 CNN
def load2d(test = False, cols = None):
    X, y = load(test = test)
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
    input_shape = (None, 1, 96, 96),
    conv1_num_filters = 32, conv1_filter_size = (3,3), pool1_pool_size = (2,2),
    conv2_num_filters = 64, conv2_filter_size = (2,2), pool2_pool_size = (2,2),
    conv3_num_filters = 128, conv3_filter_size = (2,2), pool3_pool_size = (2,2),
    hidden4_num_units = 500,
    hidden5_num_units = 500,
    output_nonlinearity = None,
    output_num_units = 30,

    update = nesterov_momentum,
    update_learning_rate = 0.01,
    update_momentum = 0.9,

    regression = True,
    max_epochs = 1000,
    verbose = 1,
)

X, y = load2d()
net2.fit(X, y)

import cPickle as pickle
with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)

# RMSE = np.sqrt() * 48

# compare loss-epoch curve
train_loss1 = np.array([i['train_loss'] for i in net1.train_history_])
valid_loss1 = np.array([i['valid_loss'] for i in net1.train_history_])
train_loss2 = np.array([i['train_loss'] for i in net2.train_history_])
valid_loss2 = np.array([i['valid_loss'] for i in net2.train_history_])

plt.plot(train_loss1, linewidth = 2, linestyle = 'solid', c = 'b', label = 'net1 train')
plt.plot(valid_loss1, linewidth = 2, linestyle = 'solid', c = 'g', label = 'net1 valid')
plt.plot(train_loss2, linewidth = 2, linestyle = 'dashed', c = 'b', label = 'net2 train')
plt.plot(valid_loss2, linewidth = 2, linestyle = 'dashed', c = 'g', label = 'net2 valid')

plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-3, 1e-2)
plt.yscale('log')
plt.show()

# see how it predict one problematic sample in the Test set, #7 obs
sample1 = load(test = True)[0][[6]]
sample2 = load2d(test = True)[0][[6]]
y_pred1 = net1.predict(sample1)[0]
y_pred2 = net2.predict(sample2)[0]

fig = plt.figure(figsize = (6,3))
ax = fig.add_subplot(1,2,1)
plot_sample(sample1[0], y_pred1, ax)
ax = fig.add_subplot(1,2,2)
plot_sample(sample1[0], y_pred2, ax)
plt.show()


# data augmentation
X_flipped = X[:,:,:, ::-1]
fig = plt.figure(figsize = (6,3))
ax = fig.add_subplot(1,2,1)
plot_sample(X[1], y[1], ax)
ax = fig.add_subplot(1,2,2)
plot_sample(X_flipped[1], y[1], ax)
plt.show()

# list(df.columns.values)
flip_indices = [
    (0, 2), (1, 3),
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (22, 24), (23, 25),
]

df = read_csv(os.path.expanduser(FTRAIN))
for i, j in flip_indices:
    print('# {} -> {}'.format(df.columns[i], df.columns[j]))

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
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs/2, replace = False)
        Xb[indices] = Xb[indices, :,:, ::-1]

        if yb is not None:
            yb[indices, ::2] = yb[indices, ::2] * -1
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = yb[indices, b], yb[indices, a]
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
    input_shape = (None, 1, 96, 96),
    conv1_num_filters = 32, conv1_filter_size = (3,3), pool1_pool_size = (2,2),
    conv2_num_filters = 64, conv2_filter_size = (2,2), pool2_pool_size = (2,2),
    conv3_num_filters = 128, conv3_filter_size = (2,2), pool3_pool_size = (2,2),
    hidden4_num_units = 500,
    hidden5_num_units = 500,
    output_nonlinearity = None,
    output_num_units = 30,

    update = nesterov_momentum,
    update_learning_rate = 0.01,
    update_momentum = 0.9,

    regression = True,
    batch_iterator_train = FlipBatchIterator(batch_size = 128),
    max_epochs = 3000,
    verbose = 1,
)

#X, y = load2d()
net3.fit(X, y)
with open('net3.pickle', 'wb') as f:
    pickle.dump(net3, f, -1)

# RMSE = np.sqrt() * 48

# compare loss-epoch curve
train_loss2 = np.array([i['train_loss'] for i in net2.train_history_])
valid_loss2 = np.array([i['valid_loss'] for i in net2.train_history_])
train_loss3 = np.array([i['train_loss'] for i in net3.train_history_])
valid_loss3 = np.array([i['valid_loss'] for i in net3.train_history_])

plt.plot(train_loss2, linewidth = 2, linestyle = 'dashed', c = 'b', label = 'net2 train')
plt.plot(valid_loss2, linewidth = 2, linestyle = 'dashed', c = 'g', label = 'net2 valid')
plt.plot(train_loss3, linewidth = 2, linestyle = 'solid', c = 'b', label = 'net3 train')
plt.plot(valid_loss3, linewidth = 2, linestyle = 'solid', c = 'g', label = 'net3 valid')

plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-3, 1e-2)
plt.yscale('log')
plt.show()

# see how it predict one problematic sample in the Test set, #7 obs
#sample1 = load(test = True)[0][[6]]
#sample2 = load2d(test = True)[0][[6]]
y_pred2 = net2.predict(sample2)[0]
y_pred3 = net3.predict(sample2)[0]

fig = plt.figure(figsize = (6,3))
ax = fig.add_subplot(1,2,1)
plot_sample(sample1[0], y_pred2, ax)
ax = fig.add_subplot(1,2,2)
plot_sample(sample1[0], y_pred3, ax)
plt.show()


# change learning rate and momentum
import theano

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start = 0.03, stop = 0.001):
        self.name = name
        self.start , self.stop = start, stop
        self.ls = None
    def __call__(self, nn, train_history_):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history_[-1]['epoch']
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
    input_shape = (None, 1, 96, 96),
    conv1_num_filters = 32, conv1_filter_size = (3,3), pool1_pool_size = (2,2),
    conv2_num_filters = 64, conv2_filter_size = (2,2), pool2_pool_size = (2,2),
    conv3_num_filters = 128, conv3_filter_size = (2,2), pool3_pool_size = (2,2),
    hidden4_num_units = 500,
    hidden5_num_units = 500,
    output_nonlinearity = None,
    output_num_units = 30,

    update = nesterov_momentum,
    update_learning_rate = theano.shared(float32(0.03)),
    update_momentum = theano.shared(float32(0.9)),

    regression = True,
    #batch_iterator_train = FlipBatchIterator(batch_size = 128),
    on_epoch_finished = [
        AdjustVariable('update_learning_rate', start = 0.03, stop = 0.0001),
        AdjustVariable('update_momentum', start = 0.9, stop = 0.999),
        ],
    max_epochs = 3000,
    verbose = 1,
)

#X, y = load2d()
net4.fit(X, y)
with open('net4.pickle', 'wb') as f:
    pickle.dump(net4, f, -1)

# RMSE = np.sqrt() * 48

# see how it predict one problematic sample in the Test set, #7 obs
#sample1 = load(test = True)[0][[6]]
#sample2 = load2d(test = True)[0][[6]]
y_pred3 = net4.predict(sample2)[0]
y_pred4 = net4.predict(sample2)[0]

fig = plt.figure(figsize = (6,3))
ax = fig.add_subplot(1,2,1)
plot_sample(sample1[0], y_pred3, ax)
ax = fig.add_subplot(1,2,2)
plot_sample(sample1[0], y_pred4, ax)
plt.show()

# add flipped data
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
    input_shape = (None, 1, 96, 96),
    conv1_num_filters = 32, conv1_filter_size = (3,3), pool1_pool_size = (2,2),
    conv2_num_filters = 64, conv2_filter_size = (2,2), pool2_pool_size = (2,2),
    conv3_num_filters = 128, conv3_filter_size = (2,2), pool3_pool_size = (2,2),
    hidden4_num_units = 500,
    hidden5_num_units = 500,
    output_nonlinearity = None,
    output_num_units = 30,

    update = nesterov_momentum,
    update_learning_rate = theano.shared(float32(0.03)),
    update_momentum = theano.shared(float32(0.9)),

    regression = True,
    batch_iterator_train = FlipBatchIterator(batch_size = 128),
    on_epoch_finished = [
        AdjustVariable('update_learning_rate', start = 0.03, stop = 0.0001),
        AdjustVariable('update_momentum', start = 0.9, stop = 0.999),
        ],
    max_epochs = 3000,
    verbose = 1,
)

#X, y = load2d()
net5.fit(X, y)
with open('net5.pickle', 'wb') as f:
    pickle.dump(net5, f, -1)

# RMSE = np.sqrt() * 48

# see how it predict one problematic sample in the Test set, #7 obs
#sample1 = load(test = True)[0][[6]]
#sample2 = load2d(test = True)[0][[6]]
y_pred4 = net4.predict(sample2)[0]
y_pred5 = net5.predict(sample2)[0]

fig = plt.figure(figsize = (6,3))
ax = fig.add_subplot(1,2,1)
plot_sample(sample1[0], y_pred4, ax)
ax = fig.add_subplot(1,2,2)
plot_sample(sample1[0], y_pred5, ax)
plt.show()
# compare loss-epoch curve
train_loss4 = np.array([i['train_loss'] for i in net4.train_history_])
valid_loss4 = np.array([i['valid_loss'] for i in net4.train_history_])
train_loss5 = np.array([i['train_loss'] for i in net5.train_history_])
valid_loss5 = np.array([i['valid_loss'] for i in net5.train_history_])

plt.plot(train_loss4, linewidth = 2, linestyle = 'dashed', c = 'b', label = 'net4 train')
plt.plot(valid_loss4, linewidth = 2, linestyle = 'dashed', c = 'g', label = 'net4 valid')
plt.plot(train_loss5, linewidth = 2, linestyle = 'solid', c = 'b', label = 'net5 train')
plt.plot(valid_loss5, linewidth = 2, linestyle = 'solid', c = 'g', label = 'net5 valid')

plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-3, 1e-2)
plt.yscale('log')
plt.show()

# add dropout
net6 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],
    input_shape = (None, 1, 96, 96),
    conv1_num_filters = 32, conv1_filter_size = (3,3), pool1_pool_size = (2,2), dropout1_p = 0.1,
    conv2_num_filters = 64, conv2_filter_size = (2,2), pool2_pool_size = (2,2), dropout2_p = 0.2,
    conv3_num_filters = 128, conv3_filter_size = (2,2), pool3_pool_size = (2,2), dropout3_p = 0.3,
    hidden4_num_units = 500,
    dropout4_p = 0.5,
    hidden5_num_units = 500,
    output_nonlinearity = None,
    output_num_units = 30,

    update = nesterov_momentum,
    update_learning_rate = theano.shared(float32(0.03)),
    update_momentum = theano.shared(float32(0.9)),

    regression = True,
    batch_iterator_train = FlipBatchIterator(batch_size = 128),
    on_epoch_finished = [
        AdjustVariable('update_learning_rate', start = 0.03, stop = 0.0001),
        AdjustVariable('update_momentum', start = 0.9, stop = 0.999),
        ],
    max_epochs = 3000,
    verbose = 1,
)

import sys
sys.setrecursionlimit(10000)

#X, y = load2d()
net6.fit(X, y)
with open('net6.pickle', 'wb') as f:
    pickle.dump(net6, f, -1)

# RMSE = np.sqrt() * 48
from sklearn.metrics import mean_squared_error
print(mean_squared_error(net6.predict(X), y))

# see how it predict one problematic sample in the Test set, #7 obs
#sample1 = load(test = True)[0][[6]]
#sample2 = load2d(test = True)[0][[6]]
y_pred5 = net5.predict(sample2)[0]
y_pred6 = net6.predict(sample2)[0]

fig = plt.figure(figsize = (6,3))
ax = fig.add_subplot(1,2,1)
plot_sample(sample1[0], y_pred5, ax)
ax = fig.add_subplot(1,2,2)
plot_sample(sample1[0], y_pred6, ax)
plt.show()

# add hidden units and more epochs!
net7 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],
    input_shape = (None, 1, 96, 96),
    conv1_num_filters = 32, conv1_filter_size = (3,3), pool1_pool_size = (2,2), dropout1_p = 0.1,
    conv2_num_filters = 64, conv2_filter_size = (2,2), pool2_pool_size = (2,2), dropout2_p = 0.2,
    conv3_num_filters = 128, conv3_filter_size = (2,2), pool3_pool_size = (2,2), dropout3_p = 0.3,
    hidden4_num_units = 1000,
    dropout4_p = 0.5,
    hidden5_num_units = 1000,
    output_nonlinearity = None,
    output_num_units = 30,

    update = nesterov_momentum,
    update_learning_rate = theano.shared(float32(0.03)),
    update_momentum = theano.shared(float32(0.9)),

    regression = True,
    batch_iterator_train = FlipBatchIterator(batch_size = 128),
    on_epoch_finished = [
        AdjustVariable('update_learning_rate', start = 0.03, stop = 0.0001),
        AdjustVariable('update_momentum', start = 0.9, stop = 0.999),
        ],
    max_epochs = 10000,
    verbose = 1,
)

#X, y = load2d()
net7.fit(X, y)
with open('net7.pickle', 'wb') as f:
    pickle.dump(net7, f, -1)
# compare loss-epoch curve
train_loss6 = np.array([i['train_loss'] for i in net6.train_history_])
valid_loss6 = np.array([i['valid_loss'] for i in net6.train_history_])
train_loss7 = np.array([i['train_loss'] for i in net7.train_history_])
valid_loss7 = np.array([i['valid_loss'] for i in net7.train_history_])

plt.plot(train_loss6, linewidth = 2, linestyle = 'dashed', c = 'b', label = 'net6 train')
plt.plot(valid_loss6, linewidth = 2, linestyle = 'dashed', c = 'g', label = 'net6 valid')
plt.plot(train_loss7, linewidth = 2, linestyle = 'solid', c = 'b', label = 'net7 train')
plt.plot(valid_loss7, linewidth = 2, linestyle = 'solid', c = 'g', label = 'net7 valid')

plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(1e-3, 1e-2)
plt.yscale('log')
plt.show()


# early stopping
class EarlyStopping(object):
    def __init__(self, patience = 100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
    def __call__(self, nn, train_history_):
        current_valid = train_history_[-1]['valid_loss']
        current_epoch = train_history_[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print('Early stopping.')
            print('Best valid loss is {:.6f} at epoch{}.'.format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

net8 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape = (None, 1, 96, 96),
    conv1_num_filters = 32, conv1_filter_size = (3,3), pool1_pool_size = (2,2), dropout1_p = 0.1,
    conv2_num_filters = 64, conv2_filter_size = (2,2), pool2_pool_size = (2,2), dropout2_p = 0.2,
    conv3_num_filters = 128, conv3_filter_size = (2,2), pool3_pool_size = (2,2), dropout3_p = 0.3,
    hidden4_num_units = 1000,
    dropout4_p = 0.5,
    hidden5_num_units = 1000,
    output_nonlinearity = None,
    output_num_units = 30,

    update = nesterov_momentum,
    update_learning_rate = theano.shared(float32(0.03)),
    update_momentum = theano.shared(float32(0.9)),

    regression = True,
    batch_iterator_train = FlipBatchIterator(batch_size = 128),
    on_epoch_finished = [
        AdjustVariable('update_learning_rate', start = 0.03, stop = 0.0001),
        AdjustVariable('update_momentum', start = 0.9, stop = 0.999),
        EarlyStopping(patience=200),
        ],
    max_epochs = 10000,
    verbose = 1,
)
net8.fit(X, y)
with open('net8.pickle', 'wb') as f:
    pickle.dump(net8, f, -1)



# fit specialists
from collections import OrderedDict
from sklearn.base import clone

SPECIALIST_SETTINGS = [
    dict(
        columns = (
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
        ),
        flip_indices = ((0, 2), (1, 3)),
    ),
    dict(
        columns = (
            'nose_tip_x', 'nose_tip_y',
        ),
        flip_indices = (),
    ),
    dict(
        columns = (
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        ),
        flip_indices = ((0, 2), (1, 3)),
    ),
    dict(
        columns = (
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
        ),
        flip_indices = (),
    ),
    dict(
        columns = (
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        ),
        flip_indices = ((0, 2), (1, 3), (4, 6), (5, 7)),
    ),
    dict(
        columns = (
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        ),
        flip_indices = ((0, 2), (1, 3), (4, 6), (5, 7)),
    ),
]

def fit_specialists(fname_pretrain = None):
    if fname_pretrain:
        with open(fname_pretrain, 'rb') as f:
            net_pretrain = pickle.load(f)
    else:
        net_pretrain = None
    specialists = OrderedDict()
    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        X, y = load2d(cols = cols)
        model = clone(net8)
        model.output_num_units = y.shape[1]  # 30
        model.batch_iterator_train.flip_indices = setting['flip_indices']
        model.max_epochs = int(4e6/y.shape[0])
        if 'kwargs' in setting:
            vars(model).update(setting['kwargs'])
        if net_pretrain is not None:
            model.load_params_from(net_pretrain)
        print('Training model for columns {} for {} epochs'.format(cols, model.max_epochs))
        model.fit(X, y)
        specialists[cols] = model
    with open('net-specialists.pickle', 'wb') as f:
        pickle.dump(specialists, f, -1)

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
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
            ))
    now_str = datetime.now().isoformat().replace(':', '-')
    submission = DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)
    print("Wrote {}".format(filename))

def rebin( a, newshape ):
    from numpy import mgrid
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]

def plot_learning_curves(fname_specialists='net-specialists.pickle'):
    with open(fname_specialists, 'rb') as f:
        models = pickle.load(f)

    fig = pyplot.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_color_cycle(
        ['c', 'c', 'm', 'm', 'y', 'y', 'k', 'k', 'g', 'g', 'b', 'b'])

    valid_losses = []
    train_losses = []

    for model_number, (cg, model) in enumerate(models.items(), 1):
        valid_loss = np.array([i['valid_loss'] for i in model.train_history_])
        train_loss = np.array([i['train_loss'] for i in model.train_history_])
        valid_loss = np.sqrt(valid_loss) * 48
        train_loss = np.sqrt(train_loss) * 48

        valid_loss = rebin(valid_loss, (100,))
        train_loss = rebin(train_loss, (100,))

        valid_losses.append(valid_loss)
        train_losses.append(train_loss)
        ax.plot(valid_loss,
                label='{} ({})'.format(cg[0], len(cg)), linewidth=3)
        ax.plot(train_loss,
                linestyle='--', linewidth=3, alpha=0.6)
        ax.set_xticks([])

    weights = np.array([m.output_num_units for m in models.values()],
                       dtype=float)
    weights /= weights.sum()
    mean_valid_loss = (
        np.vstack(valid_losses) * weights.reshape(-1, 1)).sum(axis=0)
    ax.plot(mean_valid_loss, color='r', label='mean', linewidth=4, alpha=0.8)

    ax.legend()
    ax.set_ylim((1.0, 4.0))
    ax.grid()
    plt.ylabel("RMSE")
    plt.show()
