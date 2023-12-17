"""Main script for the Flask web application."""
from datetime import datetime
import base64
import os

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

recognizer = cv2.face.LBPHFaceRecognizer_create()
TRAINER_PATH = "trainer.yml"
if os.path.exists(TRAINER_PATH):
    recognizer.read(TRAINER_PATH)
else:
    print(f"Cannot find file {TRAINER_PATH}")

CASCADE_WEIGHTS_PATH = "haarcascade_frontalface_default.xml"
if os.path.exists(CASCADE_WEIGHTS_PATH):
    faceCascade = cv2.CascadeClassifier(CASCADE_WEIGHTS_PATH)
else:
    print(f"Cannot find file {CASCADE_WEIGHTS_PATH}")

font = cv2.FONT_HERSHEY_SIMPLEX

unique_names = set()  # Use a set to track unique names
data = {"Name": [], "Time": [], "Id": []}
id_to_names = {
    0: "None",
    1: "Arwa",
    2: "Mariam",
    3: "Toqa",
}
ATTENDANCE_SHEET_PATH = "attendance.xlsx"


@app.route("/")
def index():
    """Index endpoint for the Flask web application."""
    return render_template("index.html")


def _recognize_faces(img: np.ndarray) -> np.ndarray:
    """Recognize faces in the given image and return the image with names.

    Args:
        img: Image to recognize faces in.

    Returns:
        Image with names of recognized faces written on top.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        face_id, distance = recognizer.predict(gray_img[y : y + h, x : x + w])
        if distance >= 100:
            continue
        name = id_to_names[face_id]

        if name in unique_names:
            continue

        unique_names.add(name)

        confidence = round(100 - distance)
        data["Name"].append(name)
        data["Time"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        data["Id"].append(face_id)

        cv2.putText(
            img=img,
            text=f"{name} ({confidence}%)",
            org=(x + 5, y - 5),
            fontFace=font,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
        )

    return img


@app.route("/upload_image", methods=["POST"])
def upload_image():
    """Upload image endpoint for the Flask web application."""
    image_data = request.json.get("image", "")

    if not image_data:
        return jsonify({"status": "error", "message": "No image data received"})

    image_data = base64.b64decode(image_data.split(",")[1])
    img_array = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img = _recognize_faces(img)

    recognized_names = [id_to_names[id] for id in data["Id"]]

    df = pd.DataFrame(data)
    df.to_excel(ATTENDANCE_SHEET_PATH, index=False)

    return jsonify({"status": "success", "names": recognized_names})


if __name__ == "__main__":
    app.run(debug=True)
