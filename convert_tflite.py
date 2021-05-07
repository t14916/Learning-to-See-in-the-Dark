import tf2onnx
import tflite2onnx

tflite_path = 'checkpoint/SonyTFLite/tf2_sony_int8.tflite'
onnx_path = 'checkpoint/int8_sony'
tflite2onnx.convert(tflite_path, onnx_path)
"""DON'T USE THIS!! USE THE FOLLOWING COMMAND:
python -m tf2onnx.convert --opset 13 --tflite 
checkpoint/SonyTFLite/tf2_sony_float32_pad.tflite --output checkpoint/float32_sony_pad.onnx,
where after --tflite argument place your tflite model, and after --output put the filepath to the output
onnx model

"""

