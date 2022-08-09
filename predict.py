import tensorflow as tf
import os
import numpy as np

selectCharacter = "ㄱ"
differCharacter = "ㅜ"
dic = {"ㄱ":"a","ㄴ":"b","ㄷ":"c","ㄹ":"d","ㅁ":"e","ㅂ":"f","ㅅ":"g","ㅇ":"h","ㅈ":"i","ㅊ":"j","ㅋ":"k","ㅌ":"l","ㅍ":"m","ㅎ":"n","ㅏ":"o","ㅓ":"p","ㅗ":"q","ㅜ":"r","ㅡ":"s","ㅣ":"t","ㅑ":"u","ㅕ":"v","ㅛ":"w","ㅠ":"x","ㅐ":"y","ㅒ":"z","ㅔ":"a1","ㅖ":"b1"}

# 1 = 정, 2 = 역, 3 = 다른 
select = 2

rightdatafilename = './data/'+ dic[selectCharacter] + '_right.txt'
reversedatafilename = './data/'+ dic[selectCharacter] + '_reverse.txt'
differdatafilename = './data/'+ dic[differCharacter] + '_right.txt'

modelname = './model/'+ dic[selectCharacter] + '.h5'

# 정방향 역방향 선택
if select == 1:
    file = open(rightdatafilename, "r")
elif select == 2:
    file = open(reversedatafilename, "r")
elif select == 3:
    file = open(differdatafilename, "r")

strings = file.readlines()
file.close()


test_data = []
test_labels = []

for i in range(len(strings)):
    string = strings[i].strip('\n')
    if i % 3 == 0:
        if selectCharacter == string:
            test_labels.append(1)
        else:
            test_labels.append(0)
    elif i % 3 == 1:
        dataStr = string.replace('[', "").replace(']', '').split(", ")
        dataX = []
        dataY = []
        for j in range(len(dataStr)):
            if j % 2 == 0:
                posX = float(dataStr[j])
                dataX.append(posX)
            else:
                posY = float(dataStr[j])
                dataY.append(posY)

        numpyX = np.array(dataX)
        numpyY = np.array(dataY)

        dataXY = [[dataX[j], dataY[j]] for j in range(len(dataX))]
        test_data.append(dataXY)

test_data = np.array(test_data)
test_labels = np.array(test_labels)


print('테스트 데이터 모양: ', test_data.shape)
print('테스트 데이터 라벨 모양: ', test_labels.shape)

os.environ["CUDA_VISIBLE_DEVICES"]='0' 

model = tf.keras.models.load_model(modelname)


for i in range(len(test_labels)):
    prediction = model.predict(test_data[i].reshape(1, 50, 2))
    print('실제: ', test_labels[i], ', 예측: ', prediction)
print("검사 모델: ", selectCharacter)
if select == 1:
    print("선택한 글씨: ", selectCharacter, "정방향")
elif select == 2:
    print("선택한 글씨: ", selectCharacter, "역방향")
elif select == 3:
    print("선택한 글씨: ", differCharacter, "정방향")

    





    
# print(test_data)
# print(test_labels)

# arr = strings.split('\n')
# print(arr)