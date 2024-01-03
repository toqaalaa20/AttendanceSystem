"""Main script for the Flask web application."""
from datetime import datetime
import base64
import os
import face_recognition

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

unique_names = set()  # Used to track unique names
known_face_names = []
known_face_encodings = []
data = {"Name": [], "Time": [], "Id": []}
name_to_id = {}
FACES_DIRECTORY = "faces"


def load_faces():
    """Load the faces from the faces directory."""
    for face in os.listdir(FACES_DIRECTORY):
        face_path = os.path.join(FACES_DIRECTORY, face)
        face_image = face_recognition.load_image_file(face_path)
        face_encoding = face_recognition.face_encodings(face_image)[0]
        known_face_encodings.append(face_encoding)
        # name of the file is <face_name>_<id>_<photo_id>.jpg
        face_name = face.split(".")[0].split("_")[0]
        face_id = int(face.split(".")[0].split("_")[1])
        name_to_id[face_name] = face_id
        known_face_names.append(face_name)


print("[INFO] Loading faces...")
load_faces()
print(f"[INFO] Collected {len(known_face_encodings)} faces encoding.")


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
    print("[INFO] Loading image...")
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    print("[INFO] Comparing faces...")
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    for face_name in face_names:
        if face_name in unique_names or face_name == "Unknown":
            continue

        unique_names.add(face_name)

        data["Name"].append(face_name)
        data["Time"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        data["Id"].append(name_to_id[face_name])

    return img


@app.route("/upload_image", methods=["POST"])
def upload_image():
    """Upload image endpoint for the Flask web application."""
    image_data = request.data.decode("utf-8")
    if not image_data:
        return jsonify({"status": "error", "message": "No image data received"})

    image_data = base64.b64decode(image_data.split(",")[1])
    img_array = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img = _recognize_faces(img)

    recognized_faces = [[name_to_id[name], name] for name in data["Name"]]

    # df = pd.DataFrame(data)
    # df.to_excel(ATTENDANCE_SHEET_PATH, index=False)

    return jsonify({"status": "success", "id_names": recognized_faces})


if __name__ == "__main__":
    # load faces after the app is initialized
    app.run()
