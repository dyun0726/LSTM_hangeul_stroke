import tensorflow as tf
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split

# 그래픽 카드 설정
os.environ["CUDA_VISIBLE_DEVICES"]='0' 

consonant = ["A","B","C","D","E","F","G","H","I"] # 모양에 따른 자음 한-획순 데이터
vowel = ["J", "K"]
stroke_kind = {"A":"a1, d1, e1, k1", "B":"b1, c2, d3, l3", "C": "c1, d2, e3, f3, f4, k2, l1, l2, m1, m4, n2", "D":"e1, f1, f2, m2, m3", "E":"g1", "F":"g2, i2, j3", "G":"h1, n3", "H":"i1, j2", "I":"j1, n1",
                "J": "o2, p1, q2, r1, s1, u2, u3, v1, v2, w3, x1", "K" : "o1, p2, q1, r2, t1, u1, v3, w1, w2, x2, x3"}
stroke_differ_except = {"A":["A", "H"], "B":["B"], "C":["C","I"], "D":["D", "I", "E", "F"], "E": ["E"], "F": ["F", "I"], "G": ["G"], "H": ["H", "A"], "I": ["C", "D", "F", "I"], "J": ["J", "C", "I"], "K": ["D", "I", "K"]}

selectStroke = "D"

# 정순, 역순 데이터 로드
right_data_name = "./stroke_data/" + selectStroke + "_right_data.npy"
reverse_data_name = "./stroke_data/" + selectStroke+ "_reverse_data.npy"
modelname = './stroke_model/'+ selectStroke + '.h5'
graphname = './stroke_model_graph/'+ selectStroke+ '_train_hist.png'
right_data = np.load(right_data_name)
reverse_data = np.load(reverse_data_name)

# 다른 모양의 획 데이터 로드 (비슷한거 제외)
wrong_data = np.empty((0, 50, 2), float)
print(wrong_data)
if selectStroke in consonant:
    differ_except = stroke_differ_except[selectStroke]
    for ch in consonant:
        if ch not in differ_except:
            print("잘못된 데이터에 추가할 형태: ",ch)
            wrong_right_data = np.load("./stroke_data/" + ch + "_right_data.npy")
            wrong_reverse_data = np.load("./stroke_data/" + ch + "_reverse_data.npy")
            wrong_data = np.append(wrong_data, wrong_right_data, axis=0)
            wrong_data = np.append(wrong_data, wrong_reverse_data, axis=0)
elif selectStroke in vowel:
    differ_except = stroke_differ_except[selectStroke]
    for ch in vowel + consonant:
        if ch not in differ_except:
            print("잘못된 데이터에 추가할 형태: ",ch)
            wrong_right_data = np.load("./stroke_data/" + ch + "_right_data.npy")
            wrong_reverse_data = np.load("./stroke_data/" + ch + "_reverse_data.npy")
            wrong_data = np.append(wrong_data, wrong_right_data, axis=0)
            wrong_data = np.append(wrong_data, wrong_reverse_data, axis=0)


print("맞는 형태의 획순 데이터 개수: ", right_data.shape)
print("맞지만 역순 형태의 획순 데이터 개수: ", reverse_data.shape)
print("다른 모양의 획순 데이터 개수: ", wrong_data.shape)

# labels numpy 준비          
labels = np.ones(len(right_data))
labels = np.append(labels, np.zeros(int(len(right_data) / 2) * 2))

# 데이터를 1/3 씩 이루게 하기
reverse_data_sampled = random.sample(reverse_data.tolist(), int(len(right_data) / 2))
wrong_data_sampled = random.sample(wrong_data.tolist(), int(len(right_data) / 2))
fullData = right_data.tolist() + reverse_data_sampled + wrong_data_sampled 
fullData = np.array(fullData)

print(fullData.shape)
print(labels.shape)


x_train, x_valid, y_train, y_valid = train_test_split(fullData, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=34)

print(x_train.shape)
print(y_train.shape)

# ind = np.where(y_train == 1)
# print(x_train[ind])

batch = 32
epoch = 50

print("학습 시킬 모델: ", selectStroke)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32, activation = 'relu', input_shape = (x_train.shape[1], x_train.shape[2])))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto')
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