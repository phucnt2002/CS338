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


# model = torch.hub.load(
#     "ultralytics/yolov5", "custom", path="model/best.pt", force_reload=True
# )

# model.eval()
# model.conf = 0.5
# model.iou = 0.45

from io import BytesIO


def gen():
    """
    The function takes in a video stream from the webcam, runs it through the model, and returns the
    output of the model as a video stream
    """
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


@app.route("/video")
def video():
    """
    It returns a response object that contains a generator function that yields a sequence of images
    :return: A response object with the gen() function as the body.
    """
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/", methods=["GET", "POST"])
def predict():
    model, labels = load_yolov5_model()
    
    """
    The function takes in an image, runs it through the model, and then saves the output image to a
    static folder
    :return: The image is being returned.
    """
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
        text_reader = easyocr_model_load()
        # Detecting the text from the image.
        detected, _ = detection(img, model, labels)
        # Reading the text from the image.
        resulteasyocr = text_reader.readtext(
            detected
        ) 
        
        print(text_reader)
        text = filter_text(params.rect_size, resulteasyocr, params.region_threshold)
        print(text)
        cv2.imwrite("./static/detected_image.jpg", detected)
        
    return render_template("index.html", image_file="detected_image.jpg", text=text)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
