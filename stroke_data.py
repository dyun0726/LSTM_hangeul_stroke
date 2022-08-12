import pymongo
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

targetCharacters = ["ㄱ","ㄴ","ㄷ","ㄹ","ㅁ","ㅂ","ㅅ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ","ㅏ","ㅓ","ㅗ","ㅜ","ㅡ","ㅣ","ㅑ","ㅕ","ㅛ","ㅠ","ㅐ","ㅒ","ㅔ","ㅖ"] 
dic = {"ㄱ":"a","ㄴ":"b","ㄷ":"c","ㄹ":"d","ㅁ":"e","ㅂ":"f","ㅅ":"g","ㅇ":"h","ㅈ":"i","ㅊ":"j","ㅋ":"k","ㅌ":"l","ㅍ":"m","ㅎ":"n","ㅏ":"o","ㅓ":"p","ㅗ":"q","ㅜ":"r","ㅡ":"s","ㅣ":"t","ㅑ":"u","ㅕ":"v","ㅛ":"w","ㅠ":"x","ㅐ":"y","ㅒ":"z","ㅔ":"a1","ㅖ":"b1"}
dic_stroke = {"ㄱ":1,"ㄴ":1,"ㄷ":2,"ㄹ":3,"ㅁ":3,"ㅂ":4,"ㅅ":2,"ㅇ":1,"ㅈ":2,"ㅊ":3,"ㅋ":2,"ㅌ":3,"ㅍ":4,"ㅎ":3,"ㅏ":2,"ㅓ":2,"ㅗ":2,"ㅜ":2,"ㅡ":1,"ㅣ":1,"ㅑ":3,"ㅕ":3,"ㅛ":3,"ㅠ":3,"ㅐ":3,"ㅒ":4,"ㅔ":3,"ㅖ":4}
stroke_kind = {"A":"a1, d1, e1, k1", "B":"b1, c2, d3, l3", "C": "c1, d2, e3, f3, f4, k2, l1, l2, m1, m4, n2", "D":"e1, f1, f2, m2, m3", "E":"g1", "F":"g2, i2, j3", "G":"h1, n3", "H":"i1, j2", "I":"j1, n1", 
                "J": "o2, p1, q2, r1, s1, u2, u3, v1, v2, w3, x1", "K" : "o1, p2, q1, r2, t1, u1, v3, w1, w2, x2, x3"}

# 한번 하면 할 필요 없음 (주석처리)
# doc = collections.find({"phoneme":{"$nin": ["가", "노"]}}).sort("phoneme", 1)
# subDoc = subCollections.find({"wordType":{"$nin": ["word"]}}).sort("phoneme", 1)
# # 이상한 데이터 있는지 체크
# for record in doc:
#     recordPhoneme = record["phoneme"]
#     # 필요한 글자가 아니면 넘기기
    
#     # record 데이터 획마다 나르기
#     dataStr = record["data"].split('/')
    
#     # record의 획수가 이상하면 넘기기
#     if (dic_stroke[recordPhoneme] != len(dataStr)):
#         print("no")
# for record in subDoc:
#     recordPhoneme = record["phoneme"]
#     # 필요한 글자가 아니면 넘기기
    
#     # record 데이터 획마다 나르기
#     dataStr = record["data"].split('/')
    
#     # record의 획수가 이상하면 넘기기
#     if (dic_stroke[recordPhoneme] != len(dataStr)):
#         print("no")

# for 문 돌리니까 len가 안되네;;
sk = "K"
# 기본 획 분류
str_kind_list = stroke_kind[sk].split(", ")
need_char_list = []
need_stroke_list = []
for s in str_kind_list:
    need_char_list.append(s[0])
    need_stroke_list.append(int(s[1]))

print(need_char_list)
print(need_stroke_list)


doc = collections.find({"phoneme":{"$nin": ["가", "노"]}}).sort("phoneme", 1)
subDoc = subCollections.find({"wordType":{"$nin": ["word"]}}).sort("phoneme", 1)

stroke_data_list = []
for record in doc:
    recordPhoneme = record["phoneme"]
    # 필요한 글자가 아니면 넘기기
    if (dic[recordPhoneme] not in need_char_list):
        continue
    
    # record 데이터 획마다 나르기
    dataStr = record["data"].split('/')
    
    
    for i in range(len(need_char_list)):
        if (need_char_list[i] == dic[recordPhoneme]):

            stroke_str_data = dataStr[need_stroke_list[i]-1]
            stroke_data = stroke_str_data.split(',')
            stroke_num_data = [float(j) for j in stroke_data]

            xList = []
            yList = []
            for j in range(len(stroke_num_data)):
                if j % 2 == 0:
                    xList.append(stroke_num_data[j])
                else:
                    yList.append(stroke_num_data[j])
            # 데이터 배열 크기가 너무 작음 (이상한 데이터?)
            if len(xList) < 3:
                print(xList)
                print(yList)
                print(recordPhoneme)
            else:
                stroke_data_list.append([xList, yList]) 

for record in subDoc:
    recordPhoneme = record["phoneme"]
    # 필요한 글자가 아니면 넘기기
    if (dic[recordPhoneme] not in need_char_list):
        continue
    
    # record 데이터 획마다 나르기
    dataStr = record["data"].split('/')
    
    
    for i in range(len(need_char_list)):
        if (need_char_list[i] == dic[recordPhoneme]):

            stroke_str_data = dataStr[need_stroke_list[i]-1]
            stroke_data = stroke_str_data.split(',')
            stroke_num_data = [float(j) for j in stroke_data]

            xList = []
            yList = []
            for j in range(len(stroke_num_data)):
                if j % 2 == 0:
                    xList.append(stroke_num_data[j])
                else:
                    yList.append(stroke_num_data[j])
            # 데이터 배열 크기가 너무 작음 (이상한 데이터?)
            if len(xList) < 3:
                print(xList)
                print(yList)
                print(recordPhoneme, need_stroke_list[i])
            else:
                stroke_data_list.append([xList, yList]) 
 

print("해당 stroke 데이터 개수: ",len(stroke_data_list))

# 데이터 길이 수정 
dataLen = 50
resize_data_list = []
for i in stroke_data_list:

    
    
    # 획 가운데로 설정, 비율 조정
    x = np.array(i[0])
    y = i[1]

    maxX, minX, maxY, minY = max(x), min(x), max(y), min(y)
    cenX, cenY = (maxX + minX) / 2, (maxY + minY)/2
    lenX, lenY = (maxX - minX), maxY-minY
    len = lenX if lenX > lenY else lenY

    x = (np.array(i[0])  - cenX) / len
    y = (np.array(i[1]) - cenY) / len
    t = np.array(range(x.shape[0]))

    # if x.shape[0] < 4:
    #     print(x)
    #     print(y)

    # 배열 길이 50으로 맞추기
    tnew = np.linspace(t.min(), t.max(), dataLen)

    fx_linear = interpolate.interp1d(t, x, kind='linear')
    resize_x = fx_linear(tnew)
    fy_linear = interpolate.interp1d(t, y, kind='linear')
    resize_y = fy_linear(tnew)

    resizeXY = []
    for j in range(dataLen):
        resizeXY.append([resize_x[j], resize_y[j]])

    resize_data_list.append(resizeXY)
    
resize_data_list = np.array(resize_data_list)
# print(resize_data_list[0])
print("해당 stroke 데이터 모양:" , resize_data_list.shape)

filedataname = "./stroke_data/" + sk + "_right_data"
np.save(filedataname, resize_data_list)

    # 역순 데이터 생성
reverse_data_list = []
for i in resize_data_list:
    reverse_data_list.append(i[::-1])
reverse_data_list = np.array(reverse_data_list)

# print(reverseData)
print("역순 데이터 모양: ", reverse_data_list.shape)
reversedataname = "./stroke_data/" + sk + "_reverse_data"
np.save(reversedataname, reverse_data_list)


        



        
            


    
    
