import onnx
from onnx import shape_inference
path = "./fcos_effb0.onnx" #the path of your onnx model
new = "./fcos_effb0_with_shape.onnx"
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), new)
