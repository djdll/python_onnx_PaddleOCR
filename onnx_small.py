import os
import sys
import cv2
import glob
import onnxruntime
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import onnx
import onnxsim #import simplify
model = onnx.load("onnx_model/rec_large.onnx")
model_sim,flag = onnxsim.simplify(model)
if not flag:
    print("failed")
onnx.save(model_sim,"onnx_model/rec_sim.onnx")
# import onnx_tool
# modelpath = 'onnx_model/rec_large.onnx'
# onnx_tool.model_profile(modelpath,dynamic_shapes={'x':np.zeros((1,3,48,320))})
#Total                      11,276,898,321.0  100%        1,673,623.0  100% 

# resolution        time                MACs                Memory              Params
# 640*640        251.32ms               11.54G               1.04G                1.67M
# 320*320        64.96ms                2.88G                0.27G                1.67M