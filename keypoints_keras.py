import os

import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = '/Users/tian/Documents/CNN/face/training.csv'
FTEST = '/Users/tian/Documents/CNN/face/test.csv'

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


def load2d(test=False, cols=None):
    X, y = load(test, cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', color = 'b', s=10)

X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
# >>> X.shape
# (2140, 9216)
model = Sequential()
model.add(Dense(100, input_dim = 9216))
model.add(Activation('relu'))
model.add(Dense(30))

sgd = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'mean_squared_error')
fit = model.fit(X, y, nb_epoch = 400, validation_split = 0.2)
train_history = fit.history

plt.figure()
plt.plot(train_history['loss'], 'b', linewidth = 3, label = 'train')
plt.plot(train_history['val_loss'], 'g', linewidth = 3, label = 'valid')
plt.grid()
plt.legend()
plt.ylim(1e-3, 1e-2)
plt.xlim(0, 400)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.show()

X_test, _ = load(test = True)
y_pred = model.predict(X_test)

fig = plt.figure(figsize = (6,6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(16):
    axis = fig.add_subplot(4, 4, i + 1, xticks = [], yticks = [])
    plot_sample(X_test[i], y_pred[i], axis)

plt.show()

# save the model
json_string = model.to_json()
open('model1_architecture.json', 'w').write(json_string)
model.save_weights('model1_weights.h5')
# load the saved model
from keras.models import model_from_json
model = model_from_json(open('model1_architecture.json').read())
model.load_weights('model1_weights.h5')



from keras.layers import Convolution2D, MaxPooling2D, Flatten
X, y = load2d()
model2 = Sequential()

model2.add(Convolution2D(32, 3, 3, input_shape=(96, 96, 2140)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Convolution2D(64, 2, 2))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Convolution2D(128, 2, 2))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Flatten())
model2.add(Dense(500))
model2.add(Activation('relu'))
model2.add(Dense(500))
model2.add(Activation('relu'))
model2.add(Dense(30))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model2.compile(loss='mean_squared_error', optimizer=sgd)
fit2 = model2.fit(X, y, nb_epoch=1000, validation_split=0.2)
