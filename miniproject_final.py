from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from model.u2net import U2NET, U2NETP
from notebook_utils import load_image
import openvino as ov
import torch
import os

# 데이터 및 이미지 로드를 위한 매개변수
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'  # 얼굴 감지 모델 경로
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'  # 감정 감지 모델 경로
background_folder = 'background'    # 배경 이미지 폴더 경로

# 감정 감지 모델 로드
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
face_detection = cv2.CascadeClassifier(detection_model_path)

# U-2-Net 모델 로드
u2net_model = U2NETP
model_path = "model/u2net_lite/u2net_lite.pth"
net = u2net_model(3, 1)
net.eval()
net.load_state_dict(torch.load(model_path, map_location="cpu"))

# OpenVINO 모델 로드
model_ir = ov.convert_model(net, example_input=torch.zeros((1, 3, 512, 512)), input=([1, 3, 512, 512]))
core = ov.Core()
device = 'AUTO'
compiled_model_ir = core.compile_model(model=model_ir, device_name=device)
input_layer_ir = compiled_model_ir.input(0)
output_layer_ir = compiled_model_ir.output(0)

# 비디오 캡처
camera = cv2.VideoCapture(0)

# 배경 이미지 로드
backgrounds = {}
for emotion in EMOTIONS:
    background_file = os.path.join(background_folder, f"{emotion}.jpg")
    backgrounds[emotion] = cv2.imread(background_file)

while True:
    frame = camera.read()[1]
    frame = imutils.resize(frame, width=800)  # 프레임 너비 조정
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 감지 수행
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        
        # 얼굴 ROI 추출
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # 감정 분류 수행
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        
        # 배경 제거 수행
        input_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
        input_scale = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
        resized_image = cv2.resize(src=frame, dsize=(512, 512))
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
        input_image = (input_image - input_mean) / input_scale
        result = compiled_model_ir([input_image])[output_layer_ir]
        resized_result = np.rint(cv2.resize(src=np.squeeze(result), dsize=(frame.shape[1], frame.shape[0]))).astype(np.uint8)
        bg_removed_result = frame.copy()
        bg_removed_result[resized_result == 0] = 255
        
        # 감정에 따라 배경 변경
        bg_image = backgrounds[label]
        bg_image = cv2.resize(bg_image, (bg_removed_result.shape[1], bg_removed_result.shape[0]))
        output_frame = np.where(bg_removed_result == 255, bg_image, bg_removed_result)
        
        # 프레임에 감정 레이블 표시
        cv2.putText(output_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(output_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        
        cv2.imshow('Emotion Recognition', output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
