import sys
import face_recognition
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import glob

# 入力
video_capture = cv2.VideoCapture(0)

def camera():
  while True:
    # ビデオの単一フレームを取得
    _, frame = video_capture.read()
    # 結果をビデオに表示
    cv2.imshow('Video', frame)

    # Esckey で終了
    if cv2.waitKey(1) == 27:
      break

camera()

# 資源の開放
video_capture.release()
cv2.destroyAllWindows()