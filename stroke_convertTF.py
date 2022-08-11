import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'

selectCharacter = "ㄷ"
consonant = ["A","B","C","D","E","F","G","H","I"] 

for ch in consonant:
    modelname = './stroke_well_trained/'+ ch + '.h5'
    modelTFname = './stroke_tflite/stroke_'+ ch.lower() + '.tflite'

    model = tf.keras.models.load_model(modelname)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_float_model = converter.convert()

    # Show model size in KBs.
    float_model_size = len(tflite_float_model) / 1024
    print('Float model size = %dKBs.' % float_model_size)

    # TFlite 모델 저장
    # Save the quantized model to file to the Downloads directory

    f = open(modelTFname, "wb")
    f.write(tflite_float_model)
    f.close()