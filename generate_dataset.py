"""Script to generate faces dataset"""
import os
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image

MAX_FACES_PER_VIDEO = 131
DATASET_PATH = "dataset"
TRAINER_PATH = "trainer.yml"


def generate_frames(video_path: str, face_id: int):
    """Generate frames from the given video path and save them in the dataset folder."""
    print("Seperating frames from video: " + video_path)
    cam = cv2.VideoCapture(video_path)
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to capture video.")
            break

        if face_detector.empty():
            print("Error: Unable to load the Haar Cascade XML file.")
            exit()

        faces = face_detector.detectMultiScale(img, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(
                filename=f"{DATASET_PATH}/User."
                + str(face_id)
                + "."
                + str(count)
                + ".jpg",
                img=img[y : y + h, x : x + w],
            )
        if count >= MAX_FACES_PER_VIDEO:
            break


def generate_dataset():
    """Generate dataset from the videos"""
    videos_path = "Videos"
    person_id = 1
    for video in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video)
        generate_frames(video_path, person_id)
        person_id += 1


def get_images_and_lables(detector) -> Tuple[List, List]:
    """Get images and labels from the dataset folder.

    Args:
        detector: Haar cascade detector.

    Returns:
        Tuple of face samples and their corresponding ids.
    """
    image_paths = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        img_pil = Image.open(image_path).convert("L")  # grayscale
        img_numpy = np.array(img_pil, "uint8")
        image_name = os.path.split(image_path)[-1]

        # Images has the format: "User.<id>.<count>.jpg"
        id = int(image_name.split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for x, y, w, h in faces:
            face_samples.append(img_numpy[y : y + h, x : x + w])
            ids.append(id)
    return face_samples, ids


def main():
    """Main entry point."""
    generate_dataset()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = get_images_and_lables(detector)
    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_PATH)
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


if __name__ == "__main__":
    main()
