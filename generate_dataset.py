"""Script to generate faces dataset"""
import os
import cv2

MAX_FACES_PER_VIDEO = 1
DATASET_PATH = "dataset"
VIDEOS_PATH = "videos"


def generate_frames(video_path: str):
    """Generate frames from the given video path and save them in the dataset folder."""
    print("Seperating frames from video: " + video_path)

    # The name of the videos has the form <name>_<id>.mp4
    name = os.path.split(video_path)[-1].split("_")[0]
    id = int(os.path.split(video_path)[-1].split("_")[1].split(".")[0])

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

        faces = face_detector.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(5, 5),
        )
        for x, y, w, h in faces:
            cv2.imwrite(
                filename=f"{DATASET_PATH}/"
                + str(name)
                + "_"
                + str(id)
                + "_"
                + str(count)
                + ".jpg",
                img=img[y : y + h, x : x + w],
            )
        count += 1
        if count >= MAX_FACES_PER_VIDEO:
            break


def generate_dataset():
    """Generate dataset from the videos"""
    for video in os.listdir(VIDEOS_PATH):
        video_path = os.path.join(VIDEOS_PATH, video)
        generate_frames(video_path)


def main():
    """Main entry point."""
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    generate_dataset()


if __name__ == "__main__":
    main()
