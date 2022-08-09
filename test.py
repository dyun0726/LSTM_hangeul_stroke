import pymongo
import numpy as np
from scipy import interpolate
import os
# TensorFlow 라이브러리
from ast import And
from traceback import print_tb
# Keras (딥러닝 라이브러리)
import tensorflow as tf

# 딥러닝 서버 주소
serverUrl = 'mongodb://192.168.0.27'
# 딥러닝 서버 포트
severPort = 27017
# 몽고DB 연결
connection = pymongo.MongoClient(serverUrl, severPort)
# 몽고DB database
db = connection.takeNoteFullData
# 몽고DB collection, 현재는 한 collection 밖에 없음.
collections = db.schemas
subCollections = db.schemasubs
# 몽고DB collection, 결과정보를 저장.
resultCollections = db.result

# print(collections.count())

doc = collections.find({"phoneme":{"$nin": ["가", "노"]}}).sort("phoneme", 1)
subDoc = subCollections.find({"wordType":{"$nin": ["word"]}}).sort("phoneme", 1)

print(len(list(doc)))
print(len(list(subDoc)))


targetCharacters = ["ㄱ","ㄴ","ㄷ","ㄹ","ㅁ","ㅂ","ㅅ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ","ㅏ","ㅓ","ㅗ","ㅜ","ㅡ","ㅣ","ㅑ","ㅕ","ㅛ","ㅠ","ㅐ","ㅒ","ㅔ","ㅖ"] 
consonant = ["ㄱ","ㄴ","ㄷ","ㄹ","ㅁ","ㅂ","ㅅ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
vowel = ["ㅏ","ㅓ","ㅗ","ㅜ","ㅡ","ㅣ","ㅑ","ㅕ","ㅛ","ㅠ","ㅐ","ㅒ","ㅔ","ㅖ"]    
dic = {"ㄱ":"a","ㄴ":"b","ㄷ":"c","ㄹ":"d","ㅁ":"e","ㅂ":"f","ㅅ":"g","ㅇ":"h","ㅈ":"i","ㅊ":"j","ㅋ":"k","ㅌ":"l","ㅍ":"m","ㅎ":"n","ㅏ":"o","ㅓ":"p","ㅗ":"q","ㅜ":"r","ㅡ":"s","ㅣ":"t","ㅑ":"u","ㅕ":"v","ㅛ":"w","ㅠ":"x","ㅐ":"y","ㅒ":"z","ㅔ":"a1","ㅖ":"b1"}
CharacterSize = {"ㄱ":100,"ㄴ":80,"ㄷ":110,"ㄹ":150,"ㅁ":150,"ㅂ":150,"ㅅ":80,"ㅇ":110,"ㅈ":110,"ㅊ":130,"ㅋ":80,"ㅌ":90,"ㅍ":100,"ㅎ":110,"ㅏ":80,"ㅓ":100,"ㅗ":100,"ㅜ":100,"ㅡ":80,"ㅣ":80,"ㅑ":100,"ㅕ":100,"ㅛ":100,"ㅠ":100,"ㅐ":100,"ㅒ":100,"ㅔ":100,"ㅖ":100}

for selectCharacter in targetCharacters:
    # 선택된 문자 뭔지 표시
    print("selectCharacter:(",selectCharacter,")")
    inputDataList = []
    wordType = ""

    # 선택된 문자가 자음인지 모음인지 체크
    if selectCharacter in consonant:
        print("consonant")
        wordType = "consonant"
    else:
        print("vowel")
        wordType = "vowel"

    doc = collections.find({"phoneme":{"$nin": ["가", "노"]}}).sort("phoneme", 1)
    subDoc = subCollections.find({"wordType":{"$nin": ["word"]}}).sort("phoneme", 1)

    for record in doc:
        # record가 무슨 글자인지
        phonemeData = record["phoneme"]

        # record가 선택한 글자가 맞나?
        answerNum = 3
        if phonemeData == selectCharacter:
            answerNum = 1
        else:
            answerNum = 0
        print(answerNum)


        recordDataList = []
        xdata = []
        ydata = []
        zdata = []

        # recordDataList에 문자 데이터 삽입
        recordDataList.append(list(phonemeData))

        # 좌표 데이터 처리
        # 데이터 획으로 나누고
        dataStr = record["data"].split('/')
        for i in range(len(dataStr)): # 획마다 또 처리
            xList = []
            yList = []
            zList = []

            # 획 데이터 , 로 나누고 float으로 형처리해서 다시 list로 만들기
            data = dataStr[i].split(',')
            recordList = [float(j) for j in data]

            # x, y좌표로 분류 (짝수, 홀수)
            for k in range(len(recordList)):
                if k % 2 == 0:
                    xList.append(recordList[k])
                    # 상호님 z축은 획 + 같은 획에서의 순서 ex) 1 + 50/2 + 0.01
                    # z축 의미가 크게 있을까?
                    weightCnt = float(i) + (k/2 * 0.01)
                    zList.append(weightCnt)
                else:
                    yList.append(recordList[k])
            
            xdata.append(xList)
            ydata.append(yList)
            zdata.append(zList)
        recordDataList.append(xdata)
        recordDataList.append(ydata)
        recordDataList.append(zdata)
        recordDataList.append(answerNum)
        inputDataList.append(recordDataList)

    print(recordDataList)
    print(len(inputDataList))

    # train, test 데이터 로드?
    for record in subDoc:
        phonemeData = record["phoneme"]

        # record가 선택한 글자가 맞나?
        answerNum = 3
        if phonemeData == selectCharacter:
            answerNum = 1
        else:
            answerNum = 0
        # print(answerNum)

        recordDataList = []
        xdata = []
        ydata = []
        zdata = []

        recordDataList.append(list(phonemeData))
        dataStr = record["data"].split('/')
        for i in range(len(dataStr)):
            xList = []
            yList = []
            zList = []

            # 획 데이터 , 로 나누고 float으로 형처리해서 다시 list로 만들기
            data = dataStr[i].split(',')
            recordList = [float(j) for j in data]

            # x, y좌표로 분류 (짝수, 홀수)
            for k in range(len(recordList)):
                if k % 2 == 0:
                    xList.append(recordList[k])
                    # 상호님 z축은 획 + 같은 획에서의 순서 ex) 1 + 50/2 + 0.01
                    # z축 의미가 크게 있을까?
                    weightCnt = float(i) + (k/2 * 0.01)
                    zList.append(weightCnt)
                else:
                    yList.append(recordList[k])
            xdata.append(xList)
            ydata.append(yList)
            zdata.append(zList)
        recordDataList.append(xdata)
        recordDataList.append(ydata)
        recordDataList.append(zdata)
        recordDataList.append(answerNum)
        inputDataList.append(recordDataList)
    print("inputDataList: ", len(inputDataList))

    wrongDataList = []
    #오답 케이스 추가  
    for i in range(len(inputDataList)): 
        # 정답 글자만 역순과 랜덤 획순 오답 추가
        if inputDataList[i][4] == 1 : 
            wrongData = []
            # 역 정렬
            wrongData.append(inputDataList[i][0])
            xReversData = []
            yReversData = []
            for k in range(len(inputDataList[i][1]) -1,-1,-1): 
                xReversData.append(list(reversed(inputDataList[i][1][k])))
                yReversData.append(list(reversed(inputDataList[i][2][k])))
            wrongData.append(xReversData)
            wrongData.append(yReversData)
            wrongData.append(inputDataList[i][3])
            wrongData.append(0)
            wrongDataList.append(wrongData)

    print("wrongDataList:",len(wrongDataList))
    # print(wrongDataList[0])

    # 학습 데이터 정리
    trainDataXY = []
    trainLabel = []
    testDataXY = []
    testLabel = []

    # 정답 오답 합침
    fullDataList = inputDataList + wrongDataList
    print("fullDataList: ", len(fullDataList))

    # 데이터 길이 100
    maxLength = 100

    # test 데이터 수
    testCnt = ""
    print("testCnt: ", testCnt)
    rTestCnt = 11

    for k in range(len(fullDataList)):
        xfullData = []
        yfullData = []
        zfullData = []
        resizeXDataList = []
        resizeYDataList = []
        resizeZDataList = []
        for l in range(len(fullDataList[k][1])):
            xfullData = xfullData + fullDataList[k][1][l]
            yfullData = yfullData + fullDataList[k][2][l]
            zfullData = zfullData + fullDataList[k][3][l]

        x = np.array(xfullData)
        y = np.array(yfullData)
        z = np.array(zfullData)
        t = np.array(range(x.shape[0]))

        tnew = np.linspace(t.min(), t.max(),int(maxLength/2))
        fx_linear = interpolate.interp1d(t, x, kind='linear')
        x_new_linear = fx_linear(tnew)
        fy_linear = interpolate.interp1d(t, y, kind='linear')
        y_new_linear = fy_linear(tnew)
        fy_linear = interpolate.interp1d(t, z, kind='linear')
        z_new_linear = fy_linear(tnew)

        # np -> list 로 변환
        xdata = x_new_linear.tolist()
        ydata = y_new_linear.tolist()
        zdata = z_new_linear.tolist()
        #print(t)

        # ?
        # for l in range(int(maxLength/2)):
        #     resizeXDataList.append(xdata[l])
        #     resizeYDataList.append(ydata[l])
        #     resizeZDataList.append(zdata[l])
        resizeXDataList = xdata
        resizeYDataList = ydata
        resizeZDataList = zdata
        
        dataXY =  [[float(resizeXDataList[l]), float(resizeYDataList[l])] for l in range(len(resizeXDataList))]
        
        if testCnt != fullDataList[k][0][0] :
            testCnt = fullDataList[k][0][0]
            rTestCnt = 0
            
        if testCnt == fullDataList[k][0][0] and rTestCnt < 11 :
            testDataXY.append(dataXY)
            testLabel.append(fullDataList[k][0][0])
            rTestCnt = rTestCnt + 1 
        elif testCnt != fullDataList[k][0][0] and rTestCnt > 10 :
            trainDataXY.append(dataXY)
            trainLabel.append(fullDataList[k][4])
            testCnt = fullDataList[k][0][0]
            rTestCnt = 0
        else :
            trainDataXY.append(dataXY)
            trainLabel.append(fullDataList[k][4])

    # 학습을 위해 array로 변환
    x = np.array(trainDataXY)
    y = np.array(trainLabel)

    #print(x)
    #print(y)

    # 학습을 위해 1차원 추가
    x = x.reshape((x.shape[0], x.shape[1], 2))

    print("Shapes of input X : ", x.shape) # (데이터 개수, 배열안에 데이터 개수, )
    print("Shapes of label Y : ", y.shape) # (데이터 개수, 1)

        
    os.environ["CUDA_VISIBLE_DEVICES"]='0' 

    # 학습 파라미터 설정
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32, activation = 'relu', input_shape = (x.shape[1], 2)))
    #model.add(Dense(7, activation = 'softmax'))
    model.add(tf.keras.layers.Dense(7))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer = 'adam', loss = 'mse')
    
    # loss값을 모니터해서 과적합이 생기면 100번 더 돌고 끊음
    # mode = auto면 loss 최저값이 100번정도 반복되면 정지, acc면 최고값이 100번정도 반복되면 정지
    # mode = min, mode = max
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 100, mode = 'auto')
    model.fit(x, y, epochs = 250, batch_size = 16384, verbose = 2, callbacks = [early_stopping])

    # 학습된 모델을 json으로 저장
    model_json = model.to_json()
    modelJson = dic[selectCharacter] + "_models.json"
    modelWeights = dic[selectCharacter] + "_weights.h5"

    with open(modelJson, "w") as json_file : 
        json_file.write(model_json)
        
    # 학습된 모델의 weight를 저장
    model.save_weights(modelWeights)
    
    # 테스트 데이터 확인
    for i in range(len(testDataXY)): 
        # 테스트를 위해 array로 변환
        x_test = np.array(testDataXY[i])
        # 테스트를 위해 1차원 추가
        x_test = x_test.reshape((1, len(x_test), 2))
        # 예측
        yhat = model.predict(x_test)
        print('Correction: ', testLabel[i], ', Prediction:  ', yhat)



    break



#interpolate 실험
# from scipy import interpolate
# import numpy

# arr_len = 7

# t = [0,1,2]

# print(len(t)/arr_len)
# x = [1,2.345,4]

# x_t = interpolate.interp1d(t, x, kind='linear')

# tnew = numpy.arange(min(t), max(t), max(t)/arr_len)
# xnew = x_t(tnew)

# print(xnew)