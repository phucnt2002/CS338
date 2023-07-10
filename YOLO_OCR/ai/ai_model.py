import numpy as np
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, xyxy2xywh
import torch.backends.cudnn as cudnn
import os

from cv2 import VideoCapture
from utils.params import Parameters

params = Parameters()


def load_yolov5_model():
    """
    It loads the model and returns the model and the names of the classes.
    :return: model, names
    """
    print(os.getcwd())
    model_file = "./YOLO_OCR/model/best1.pt"
    model = attempt_load(model_file, map_location=params.device)
    print("device", params.device)
    stride = int(model.stride.max())  # model stride
    names = (
        model.module.names if hasattr(model, "module") else model.names
    )  # get class names

    return model, names


def detection(frame, model, names):
    detected_plate = ""
    detected_plate_rescaled=""
    """
    It takes an image, runs it through the model, and returns the image with bounding boxes drawn around
    the detected objects
    
    :param frame: The frame of video or webcam feed on which we're running inference
    :param model: The model to use for detection
    :param names: a list of class names
    :return: the image with the bounding boxes and the label of the detected object.
    """
    out = frame.copy()

    frame = cv2.resize(
        frame,
        (params.pred_shape[1], params.pred_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    frame = np.transpose(frame, (2, 1, 0))

    cudnn.benchmark = True  # set True to speed up constant image size inference

    if params.device.type != "cpu":
        model(
            torch.zeros(1, 3, params.imgsz, params.imgsz)
            .to(params.device)
            .type_as(next(model.parameters()))
        )  # run once

    frame = torch.from_numpy(frame).to(params.device)
    frame = frame.float()
    frame /= 255.0
    if frame.ndimension() == 3:
        frame = frame.unsqueeze(0)

    frame = torch.transpose(frame, 2, 3)

    pred = model(frame, augment=False)[0]
    pred = non_max_suppression(pred, params.conf_thres, max_det=params.max_det)

    label = ""
    
    # detections per image
    for i, det in enumerate(pred):

        img_shape = frame.shape[2:]
        out_shape = out.shape

        s_ = f"{i}: "
        s_ += "%gx%g " % img_shape  # print string

        if len(det):

            gain = min(
                img_shape[0] / out_shape[0], img_shape[1] / out_shape[1]
            )  # gain  = old / new

            coords = det[:, :4]

            pad = (
                (img_shape[1] - out_shape[1] * gain) / 2,
                (img_shape[0] - out_shape[0] * gain) / 2,
            )  # wh padding

            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
            coords[:, :4] /= gain

            coords[:, 0].clamp_(0, out_shape[1])  # x1
            coords[:, 1].clamp_(0, out_shape[0])  # y1
            coords[:, 2].clamp_(0, out_shape[1])  # x2
            coords[:, 3].clamp_(0, out_shape[0])  # y2

            det[:, :4] = coords.round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s_ += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det):

                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())

                confidence_score = conf
                class_index = cls
                object_name = names[int(cls)]

                detected_plate = (
                    frame[:, :, y1:y2, x1:x2].squeeze().permute(1, 2, 0).cpu().numpy()
                )
                
                if detected_plate.size > 0: 
                    cv2.imshow("Crooped Plate ", detected_plate)
                # detected_plate_bgr = cv2.cvtColor(detected_plate, cv2.COLOR_RGB2BGR)
                # Rescale the pixel values to the range 0-255
                detected_plate_rescaled = (detected_plate * 255).astype(np.uint8)


                # Save the image as JPG
                output_path = "./static/output.jpg"  # Replace with the desired output path
                cv2.imwrite(output_path, detected_plate_rescaled)

                # rect_size= (detected_plate.shape[0]*detected_plate.shape[1])
                c = int(cls)  # integer class
                label = names[c] if params.hide_conf else f"{names[c]} {conf:.2f}"

                tl = params.rect_thickness

                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(
                    out, c1, c2, params.color, thickness=tl, lineType=cv2.LINE_AA
                )

                if label:
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[
                        0
                    ]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(out, c1, c2, params.color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(
                        out,
                        label,
                        (c1[0], c1[1] - 2),
                        0,
                        tl / 3,
                        [225, 255, 255],
                        thickness=tf,
                        lineType=cv2.LINE_AA,
                    )
    return out, label, detected_plate_rescaled
