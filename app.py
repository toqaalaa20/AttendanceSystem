from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import base64
import io
import os

app = Flask(__name__)

recognizer = cv2.face.LBPHFaceRecognizer_create()
file_path = 'trainer\\trainer.yml'
if os.path.exists(file_path):
    recognizer.read(file_path)
else:
    print("Error reading the file 1")

cascadePath = 'haarcascade_frontalface_default.xml'
if os.path.exists(cascadePath):
    faceCascade = cv2.CascadeClassifier(cascadePath)
else:
    print("Error reading the file 2")

font = cv2.FONT_HERSHEY_SIMPLEX

# Initiate some global variable
data = {'Name': [], 'Time': [], 'Id': []}
id_names = {0: 'None', 1: 'Arwa', 2: 'Mariam', 3: 'Toqa'}  # Replace with your actual IDs and names

@app.route('/')
def index():
    return render_template('index.html')

def recognize_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    unique_names = set()  # Use a set to track unique names

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 100:
            name = id_names[id]

            # Check if the name is already in the set
            if name not in unique_names:
                unique_names.add(name)

                confidence = round(100 - confidence)
                data['Name'].append(name)
                data['Time'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                data['Id'].append(id)  # Store the recognized face ID

                # Draw the recognized name and confidence on the frame
                cv2.putText(img, f"{name} ({confidence}%)", (x + 5, y - 5), font, 1, (255, 255, 255), 2)

    return img

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    image_data = request.json.get('image', '')

    if image_data:
        # Decode base64 image data
        image_data = base64.b64decode(image_data.split(',')[1])
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img = recognize_faces(img)

        recognized_names = [id_names[id] for id in data['Id']]

        df = pd.DataFrame(data)
        df.to_excel('attendance.xlsx', index=False)

        # Clear the data dictionary after processing
        data['Name'].clear()
        data['Time'].clear()
        data['Id'].clear()

        return jsonify({'status': 'success', 'names': recognized_names})

    return jsonify({'status': 'error', 'message': 'No image data received'})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    image_data = request.json.get('image', '')

    if image_data:
        image_data = base64.b64decode(image_data.split(',')[1])
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img = recognize_faces(img)

        recognized_names = [id_names[id] for id in data['Id']]

        df = pd.DataFrame(data)
        df.to_excel('attendance.xlsx', index=False)

        # Clear the data dictionary after processing
        data['Name'].clear()
        data['Time'].clear()
        data['Id'].clear()

        return jsonify({'status': 'success', 'names': recognized_names})

    return jsonify({'status': 'error', 'message': 'No image data received'})

if __name__ == '__main__':
    app.run(debug=True)
