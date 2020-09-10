import sys
import face_recognition
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import glob

face_locations = []

# 入力
video_capture = cv2.VideoCapture(0)

def camera():
  while True:
    # ビデオの単一フレームを取得
    _, frame = video_capture.read()
    
    # 顔の位置情報検索
    face_locations = face_recognition.face_locations(frame)

    # 位置情報の表示
    for (top, right, bottom, left) in face_locations:
      # 枠を描画
      cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 3)

    # 結果をビデオに表示
    cv2.imshow('Video', frame)

    # Esckey で終了
    if cv2.waitKey(1) == 27:
      break

camera()

# 資源の開放
video_capture.release()
cv2.destroyAllWindows()