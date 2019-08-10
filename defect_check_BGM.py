#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:23:10 2019

@author: sudhakar
"""

from numpy import loadtxt
from keras.models import load_model
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
 
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required =True, help="PAth to image")
args = vars(ap.parse_args())
# load model
model = load_model('weights.defective_parts_BGM_2.hdf5')
# summarize model.
#model.summary()

image = cv2.imread(args["image"])
image = cv2.resize(image,(224,224))
image = np.array(image)
image = image.reshape(1,224,224,3)
print(model.predict(image))
if(model.predict(image)>0.5):
    print("defect")
else:
    print("no defect")
