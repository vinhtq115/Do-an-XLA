import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_path = 'images'

def train():
    images = [os.path.join(image_path, f) for f in os.listdir(image_path)]


def extract_videos_to_images(file: str, output: str, studentid: str):
    """
    Accept a video file and return all the frames in grayscale
    :param file: Path to video file
    :param output: Output image folder
    :param studentid: Student ID for naming file
    """
    video = cv2.VideoCapture(file)
    success, image = video.read()
    count = 0
    while success:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.imwrite(os.path.join(output, studentid + "_%d.jpg" % count), gray[y: y+h, x: x+w])
            success, image = video.read()
            count += 1
            if count == 30:
                break
