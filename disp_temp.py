# coding: utf-8
import picamera
import picamera.array
import cv2
import pigpio

import Seeed_AMG8833
import os
import time

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import io
from PIL import Image

xsv = 5		#X軸サーボのPort番号
ysv = 4		#y軸サーボのPort番号
span = 300	#サーボのセンターからの可動範囲duty値
xct = 1550	#X軸サーボのセンターduty値
yct = 1490	#X軸サーボのセンターduty値
dly = 0.01  #サーボ駆動時のウェイト時間
stp = 10		#サーボ駆動時のdutyステップ値
xsize = 240	#RGB 水平サイズ
ysize = 240	#RGB 垂直サイズ

#サーボの駆動範囲
xmin = xct - span
xmax = xct + span
ymin = yct - span
ymax = yct + span

#グローバル変数
xpos = xct
ypos = yct
xpos0 = xpos
ypos0 = ypos

sv = pigpio.pi()
sensor = Seeed_AMG8833.AMG8833()

def move(svn,in0,in1,step):
	if in1 > in0:
		for duty in range(in0,in1,step):
			sv.set_servo_pulsewidth(svn,duty)
			time.sleep(dly)
	if in1 < in0:
		for duty in range(in0,in1,-step):
			sv.set_servo_pulsewidth(svn,duty)
			time.sleep(dly)


def tharmal_plot(pix):
    temp = np.array([pix[num*8:num*8+8] for num in range(8)])
    temp_array = np.rot90(temp, 2)

    # plot
    plt.figure(figsize=(4, 4), dpi=50)
    sns.heatmap(temp_array, vmax=38, vmin=26, cmap='jet', cbar=False)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # display
    img_pil = Image.open(buf)
    img_array = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGBA2BGR)
    return img_array

#カメラをセンターに移動
sv.set_servo_pulsewidth(xsv, xpos)
sv.set_servo_pulsewidth(ysv, ypos)


cascade_file = "./haarcascade_frontalface_default.xml"
with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (xsize, ysize)
        camera.vflip = True
        camera.hflip = True
        while True:
            # stream.arrayにRGBの順で映像データを格納
            camera.capture(stream, 'bgr', use_video_port=True)
            # サーモグラフィーからのヒートマップデータを格納
            pixels = sensor.read_temp()
            tharmal_img = tharmal_plot(pixels)

            # グレースケールに変換
            gray = cv2.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

            # カスケードファイルを利用して顔の位置を見つける
            cascade = cv2.CascadeClassifier(cascade_file)
            face_list = cascade.detectMultiScale(gray, minSize=(100, 100))

            if len(face_list):
                for (x, y, w, h) in face_list:
                    print("face_position:",x, y, w, h)
                    color = (0, 0, 255)
                    pen_w = 5
                    cv2.rectangle(stream.array, (x, y), (x+w, y+h), color, thickness = pen_w)
                # カメラ移動
                xdf = (x + w/2) - xsize/2
                ydf = (y + h/2) - ysize/2
                xpos = int(xpos0 - xdf*0.2)
                ypos = int(ypos0 + ydf*0.2)
                if xpos > xmax:
                    xpos = xmax
                if xpos < xmin:
                    xpos = xmin
                if ypos > ymax:
                    ypos = ymax
                if ypos < ymin:
                    ypos = ymin
                move(xsv,xpos0,xpos,stp)
                move(ysv,ypos0,ypos,stp)
                xpos0 = xpos
                ypos0 = ypos

            # system.arrayをウィンドウに表示
            cv2.imshow('frame', stream.array)
            cv2.imshow('tharmao', tharmal_img)

            # "q"でウィンドウを閉じる
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # streamをリセット
            stream.seek(0)
            stream.truncate()
        cv2.destroyAllWindows()
        sv.stop()
