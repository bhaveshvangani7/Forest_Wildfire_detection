from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

coco_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
forzen_model = "frozen_inference_graph.pb"
Labels = "coco.names"
model = cv2.dnn_DetectionModel(forzen_model, coco_file)
classlabel = []

with open(Labels, 'rt') as fpt:
    classlabel = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

def detect_objects(frame):
    Classindex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    print(Classindex)

    if len(Classindex) != 0:
        for Classind, conf, boxes in zip(Classindex.flatten(), confidence.flatten(), bbox):
            if 0 < Classind <= len(classlabel):
                cv2.rectangle(frame, boxes, (255, 0, 0), 4)
                cv2.putText(frame, classlabel[Classind - 1], (boxes[0] + 10, boxes[1] + 40),
                            font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
            else:
                print(f"Invalid Classind: {Classind}")

    return frame


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/main", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file:
                video_path = "uploaded_video.mp4"
                file.save(video_path)

                # Set the desired frame size
                desired_width, desired_height = 50, 50

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise IOError("Cannot open video file!")

                # Set the frame size before reading frames
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_with_objects = detect_objects(frame)

                    cv2.imshow("Object Detection", frame_with_objects)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                cap.release()
                cv2.destroyAllWindows()

            return redirect(url_for("index"))

        elif "live" in request.form:
            # Open the live camera for object detection
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            # Set the desired frame size
            desired_width, desired_height = 550, 550

            # Set the frame size before reading frames
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_with_objects = detect_objects(frame)

                cv2.imshow("Object Detection", frame_with_objects)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

            return redirect(url_for("index"))

    return render_template("index.html")

if __name__ == "__main__":
    app.run()
