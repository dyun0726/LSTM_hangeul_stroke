import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split

# 그래픽 카드 설정
os.environ["CUDA_VISIBLE_DEVICES"]='0' 

selectCharacter = "ㄱ"
dic = {"ㄱ":"a","ㄴ":"b","ㄷ":"c","ㄹ":"d","ㅁ":"e","ㅂ":"f","ㅅ":"g","ㅇ":"h","ㅈ":"i","ㅊ":"j","ㅋ":"k","ㅌ":"l","ㅍ":"m","ㅎ":"n","ㅏ":"o","ㅓ":"p","ㅗ":"q","ㅜ":"r","ㅡ":"s","ㅣ":"t","ㅑ":"u","ㅕ":"v","ㅛ":"w","ㅠ":"x","ㅐ":"y","ㅒ":"z","ㅔ":"a1","ㅖ":"b1"}

filedataname = "./data/" + dic[selectCharacter]+ "_data.npy"
filelabelname = "./data/" + dic[selectCharacter]+ "_label.npy"
modelname = './model/'+ dic[selectCharacter] + '.h5'
graphname = './model_graph/'+ dic[selectCharacter]+ '_train_hist.png'

x = np.load(filedataname)
y = np.load(filelabelname)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y, random_state=34)

print(x_train.shape)
print(y_train.shape)

# ind = np.where(y_train == 1)
# print(ind)
# print(x_train[5])

batch = 64
epoch = 50

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32, activation = 'relu', input_shape = (x_train.shape[1], x_train.shape[2])))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 100, mode = 'auto')
hist = model.fit(x_train, y_train, batch_size = batch, epochs = epoch, verbose = 1, validation_data=(x_valid, y_valid))

model.evaluate(x_valid, y_valid, verbose = 1)

model.save(modelname)

# 학습 과정 그림 그리기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.savefig(graphname)