import pymongo
import random
import numpy as np
from scipy import interpolate

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

selectCharacter = "ㄱ"

print("selectCharacter:(", selectCharacter, ")")

doc = collections.find({"phoneme":{"$nin": ["가", "노"]}}).sort("phoneme", 1)
subDoc = subCollections.find({"wordType":{"$nin": ["word"]}}).sort("phoneme", 1)


answerCnt = 0
recordDataList = []
for record in doc:
    recordPhoneme = record["phoneme"]
    # answernum: selectCharacter랑 같은 문자의 데이터 인지
    if selectCharacter == recordPhoneme:
        answernum = 1
        answerCnt += 1
    else:
        answernum = 0
    
    recordData = []
    recordData.append(recordPhoneme)

    # x, y, z(stroke) data 담을 list 초기화
    xList = []
    yList = []
    zList = []
    dataStr = record["data"].split('/')
    for i in range(len(dataStr)):
        data = dataStr[i].split(',')

        dataFloat = [float(j) for j in data]

        for j in range(len(dataFloat)):
            if j % 2 == 0:
                xList.append(dataFloat[j])
                zList.append(float(i))
            else:
                yList.append(dataFloat[j])
    
    recordData.append(xList)
    recordData.append(yList)
    recordData.append(zList)
    recordData.append(answernum)
    recordDataList.append(recordData)

for record in subDoc:
    recordPhoneme = record["phoneme"]
     # answernum: selectCharacter랑 같은 문자의 데이터 인지
    if selectCharacter == recordPhoneme:
        answernum = 1
        answerCnt += 1
    else:
        answernum = 0
    
    recordData = []
    recordData.append(recordPhoneme)

    # x, y, z(stroke) data 담을 list 초기화
    xList = []
    yList = []
    zList = []
    dataStr = record["data"].split('/')
    for i in range(len(dataStr)):
        data = dataStr[i].split(',')

        dataFloat = [float(j) for j in data]

        for j in range(len(dataFloat)):
            if j % 2 == 0:
                xList.append(dataFloat[j])
                zList.append(float(i))
            else:
                yList.append(dataFloat[j])
    
    recordData.append(xList)
    recordData.append(yList)
    recordData.append(zList)
    recordData.append(answernum)
    recordDataList.append(recordData)
print("총 데이터 개수: ", len(recordDataList))
print("해당 글자 데이터 개수: ", answerCnt)

# 역순 데이터 추가
reverseDataList = []
for i in recordDataList:
    if i[4] == 1:
        tmp = i
        reverseData = []
        reverseData.append(i[0])
        reverseData.append(list(reversed(i[1])))
        reverseData.append(list(reversed(i[2])))

        # 획순 데이터 수정 (앞뒤 바뀌니까)
        maxnum = max(i[3])
        arr = []
        for j in i[3]:
            arr.append(maxnum - j)
    
        reverseData.append(list(reversed(arr)))
        reverseData.append(0)
        reverseDataList.append(reverseData)

# print(reverseData)
print("역순 데이터 개수: ", len(reverseDataList))

# 학습 데이터 고르기
# 정답 1/3, 역순 1/3, 다른 글자 1/3
rightData = []
wrongData = []
for i in recordDataList:
    if i[4] == 1:
        rightData.append(i)
    else:
        wrongData.append(i)

wrongSampleData = random.sample(wrongData, len(rightData))

fullData = rightData + wrongSampleData + reverseDataList

print("학습 데이터 개수",len(fullData))

# 데이터 길이 수정 
dataLen = 50
dataPos = []
dataLabel = []
for i in fullData:

    x = np.array(i[1])
    y = np.array(i[2])
    t = np.array(range(x.shape[0]))

    tnew = np.linspace(t.min(), t.max(), dataLen)

    fx_linear = interpolate.interp1d(t, x, kind='linear')
    x_new_linear = fx_linear(tnew)
    fy_linear = interpolate.interp1d(t, y, kind='linear')
    y_new_linear = fy_linear(tnew)

    resizeXY = []
    for j in range(dataLen):
        resizeXY.append([float(x_new_linear[j]), float(y_new_linear[j])])

    dataPos.append(resizeXY)
    dataLabel.append(int(i[4]))
dataPos = np.array(dataPos)
dataLabel = np.array(dataLabel)

# (데이터 수, 50, 2)
print("데이터 모양 : ", dataPos.shape)
print("라벨 모양: ", dataLabel.shape)

dic = {"ㄱ":"a","ㄴ":"b","ㄷ":"c","ㄹ":"d","ㅁ":"e","ㅂ":"f","ㅅ":"g","ㅇ":"h","ㅈ":"i","ㅊ":"j","ㅋ":"k","ㅌ":"l","ㅍ":"m","ㅎ":"n","ㅏ":"o","ㅓ":"p","ㅗ":"q","ㅜ":"r","ㅡ":"s","ㅣ":"t","ㅑ":"u","ㅕ":"v","ㅛ":"w","ㅠ":"x","ㅐ":"y","ㅒ":"z","ㅔ":"a1","ㅖ":"b1"}

filedataname = "./data/" + dic[selectCharacter]+ "_data"
filelabelname = "./data/" + dic[selectCharacter]+ "_label"
print(filedataname)
print(filelabelname)


np.save(filedataname, dataPos)
np.save(filelabelname, dataLabel)