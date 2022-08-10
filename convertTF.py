import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'

selectCharacter = "ㄷ"
dic = {"ㄱ":"a","ㄴ":"b","ㄷ":"c","ㄹ":"d","ㅁ":"e","ㅂ":"f","ㅅ":"g","ㅇ":"h","ㅈ":"i","ㅊ":"j","ㅋ":"k","ㅌ":"l","ㅍ":"m","ㅎ":"n","ㅏ":"o","ㅓ":"p","ㅗ":"q","ㅜ":"r","ㅡ":"s","ㅣ":"t","ㅑ":"u","ㅕ":"v","ㅛ":"w","ㅠ":"x","ㅐ":"y","ㅒ":"z","ㅔ":"a1","ㅖ":"b1"}

modelname = './well_trained/'+ dic[selectCharacter] + '.h5'
modelTFname = './tflite/'+ dic[selectCharacter] + '.tflite'

model = tf.keras.models.load_model(modelname)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_float_model = converter.convert()

# Show model size in KBs.
float_model_size = len(tflite_float_model) / 1024
print('Float model size = %dKBs.' % float_model_size)


# Lite 모델 양자화로 크기 줄이는 과정
# Re-convert the model to TF Lite using quantization.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
# Show model size in KBs.
quantized_model_size = len(tflite_quantized_model) / 1024
print('Quantized model size = %dKBs,' % quantized_model_size)
print('which is about %d%% of the float model size.'\
      % (quantized_model_size * 100 / float_model_size))

# TFlite 모델 저장
# Save the quantized model to file to the Downloads directory

f = open(modelTFname, "wb")
f.write(tflite_float_model)
f.close()
