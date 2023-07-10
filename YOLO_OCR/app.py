import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from torchvision.models import detection
from ai.ai_model import load_yolov5_model
from helper.general_utils import filter_text

import torch
from torchvision import models
from flask import Flask, render_template, request, redirect, Response
from ai.ocr_model import easyocr_model_load
from ai.ai_model import detection
from helper.params import Parameters
params = Parameters()


app = Flask(__name__, static_folder='static')


from io import BytesIO


def gen():
    """The function takes in a video stream from the webcam, runs it through the model, and returns the
    output of the model as a video stream
    """
    model, labels = load_yolov5_model()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if success == True:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            results.print()
            img = np.squeeze(results.render())
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            break
        frame = cv2.imencode(".jpg", img_BGR)[1].tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/image", methods=["GET", "POST"])
def image():
    model, labels = load_yolov5_model()
    text = ""
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = cv2.imread("./test/"+file.filename)
        if img is None:
            print("Failed to read the image:", file.filename)
            return
        print(file)
        # img = Image.open(io.BytesIO(img_bytes))
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text_reader = easyocr_model_load()
        # Detecting the text from the image.
        detected, _, detected_plate = detection(frame_rgb, model, labels)
        # Reading the text from the image.
        resulteasyocr = text_reader.readtext(
            detected
        ) 
        
        print(detected_plate)
        
        text = filter_text(params.rect_size, resulteasyocr, params.region_threshold)
        # cv2.imshow("detected", detected)
        # cv2.imshow("detected_plate", detected_plate)
        
        cv2.imwrite("./static/detected_image.jpg", detected)
        
    return render_template("image.html", image_file="detected_image.jpg", text=text)

def detect_objects(video):
    model, labels = load_yolov5_model()

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            # Thực hiện phát hiện đối tượng trên frame
            detected_frame, label, detected_plate = detection(frame, model, labels)

            # Hiển thị frame với các bounding box và nhãn
            cv2.imshow("Object Detection", detected_frame)

            # Chuyển đổi frame thành dạng byte để trả về
            frame_bytes = cv2.imencode('.jpg', detected_frame)[1].tobytes()

            # Trả về frame dưới dạng video stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break

    video.release()

@app.route("/video", methods=["GET", "POST"])
def video():
    if request.method == "POST":
        # Kiểm tra xem người dùng đã chọn file video hay chưa
        if 'video' not in request.files:
            return render_template("video.html")
        
        video_file = request.files['video']
        
        # Kiểm tra xem người dùng đã chọn file hay không
        if video_file.filename == '':
            return render_template("video.html")
        
        # Lưu video vào thư mục tạm thời hoặc xử lý trực tiếp từ dữ liệu trong bộ nhớ
        video_path = f"./static/{video_file.filename}"
        video_file.save(video_path)
        
        # Đọc video từ file
        video = cv2.VideoCapture(video_path)
        
        # Kiểm tra xem video đã đọc thành công hay không
        if not video.isOpened():
            return render_template("video.html")
        
        # Trả về response chứa video stream
        return Response(detect_objects(video),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return render_template("video.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
