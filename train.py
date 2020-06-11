from keras.optimizers import Adam
from keras.optimizers import SGD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from model import FCNet

BATCH_SIZE = 64
EPOCHS = 200
DIM = 7

args = {"model": "./fc.model", "output": "./result.png"}

dataset = np.loadtxt('./data.csv', delimiter=',')

X = dataset[:60000, :-1]
Y = dataset[:60000, -1]

# k-fold cross validation
k = 5 # 1~5
shift = 10000 * k
X = np.vstack([X[-shift:], X[:-shift]])
Y = np.concatenate([Y[-shift:], Y[:-shift]])

# min-max scaling
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

# train set and validation set
trainX = X[:50000, :]
validX = X[50000:, :]
trainY = Y[:50000]
validY= Y[50000:]
 
print("Compiling model...")
# opt = SGD(lr=0.01, momentum=0.5, nesterov=True)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model = FCNet.build(DIM)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
print(model.summary())
print("Training model...")
H = model.fit(trainX, trainY, validation_data=(validX, validY), batch_size=BATCH_SIZE, 
	epochs=EPOCHS)

# find min_loss and max_acc
i = 0
min_loss = 10
min_loss_epoch = 0
for item in H.history["val_loss"]:
	i += 1
	if item < min_loss:
		min_loss = item
		min_loss_epoch = i
i = 0
max_acc = 0
max_acc_epoch = 0
for  item in H.history["val_acc"]:
	i += 1
	if item > max_acc:
		max_acc = item
		max_acc_epoch = i

print("\nmin val_loss: %f (epoch %d)" % (min_loss, min_loss_epoch))
print("max val_acc: %f (epoch %d)" % (max_acc, max_acc_epoch))

# save model
model.save(args["model"])

# plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("# Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
