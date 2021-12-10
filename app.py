import math
import argparse
import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import urllib.request
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
from PIL import Image

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = ""

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        image = request.files["image"]
        image.save(os.path.join(app.config["IMAGE_UPLOADS"],image.filename))
        file=os.path.join(app.config["IMAGE_UPLOADS"],image.filename)

    frame=cv.imread(file)
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["LShoulder", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["LShoulder", "LHip"], ["RShoulder", "RHip"], ["LHip", "RHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
               ["LHip", "LKnee"], ["RHip", "RKnee"], ["RKnee", "RAnkle"] ]
    print='fewfew'
    inWidth = 368
    inHeight = 368

    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False))
    out = net.forward()
    out = out[:, :19, :, :] 

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > 0.2 else None)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    
    
    N_T_S=math.ceil(float('%.2f'%(0.08*(max(((points[1][0]-points[2][0])**2+(points[1][1]-points[2][1])**2)**0.5,
          ((points[1][0]-points[5][0])**2+(points[1][1]-points[5][1])**2)**0.5)))))   
    S_T_S=math.ceil(float('%.2f'%(0.1*(((points[2][0]-points[5][0])**2+(points[5][1]-points[2][1])**2)**0.5))))
    S_T_WR=math.ceil(float('%.2f'%((0.065*(max(((points[5][0]-points[6][0])**2+(points[5][1]-points[6][1])**2)**0.5,
       ((points[3][0]-points[2][0])**2+(points[3][1]-points[2][1])**2)**0.5)+
          max(((points[6][0]-points[7][0])**2+(points[6][1]-points[7][1])**2)**0.5, 
            ((points[3][0]-points[4][0])**2+(points[3][1]-points[4][1])**2)**0.5))))))
    S_T_WA=math.ceil(float('%.2f'%(0.054*(max(((points[8][0]-points[2][0])**2+(points[8][1]-points[2][1])**2)**0.5,
          ((points[11][0]-points[5][0])**2+(points[11][1]-points[5][1])**2)**0.5)))))
    W=math.ceil(float('%.2f'%(0.135*(((points[8][0]-points[11][0])**2+(points[8][1]-points[11][1])**2)**0.5))))
    W_T_F=math.ceil(float('%.2f'%(0.095*((max(((points[11][0]-points[12][0])**2+(points[11][1]-points[12][1])**2)**0.5,
       ((points[8][0]-points[9][0])**2+(points[8][1]-points[9][1])**2)**0.5)+
          max(((points[12][0]-points[13][0])**2+(points[12][1]-points[13][1])**2)**0.5, 
            ((points[10][0]-points[9][0])**2+(points[9][1]-points[10][1])**2)**0.5))))))
    H=math.ceil(float('%.2f'%(0.135*1.15*((points[8][0]-points[11][0])**2+(points[8][1]-points[11][1])**2)**0.5)))
    S=math.ceil(float('%.2f'%(0.09*0.29*(max(((points[11][0]-points[12][0])**2+(points[11][1]-points[12][1])**2)**0.5,
       ((points[8][0]-points[9][0])**2+(points[8][1]-points[9][1])**2)**0.5)+
          max(((points[12][0]-points[13][0])**2+(points[12][1]-points[13][1])**2)**0.5, 
            ((points[10][0]-points[9][0])**2+(points[9][1]-points[10][1])**2)**0.5)))))
    C=math.ceil(float('%.2f'%(0.06*1.1*(max(((points[5][0]-points[6][0])**2+(points[5][1]-points[6][1])**2)**0.5,
       ((points[3][0]-points[2][0])**2+(points[3][1]-points[2][1])**2)**0.5)+
          max(((points[6][0]-points[7][0])**2+(points[6][1]-points[7][1])**2)**0.5, 
            ((points[3][0]-points[4][0])**2+(points[3][1]-points[4][1])**2)**0.5)))))
    
    return render_template('result.html', v1=N_T_S, v2=S_T_S, v3=S_T_WR, v4=S_T_WA, v5=W, v6=W_T_F, v7=H, v8=S, v9=C)

if __name__ == "__main__":
  app.run()