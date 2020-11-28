import cv2
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def extract_videos_to_images(file: str, output: str, studentid: str, face_detector):
    """
    Accept a video file and return 30 frames with face in grayscale
    :param file: Path to video file
    :param output: Output image folder
    :param studentid: Student ID for naming file
    :param face_detector: CV2 face detector
    :return True if success (30 images with face), else false
    """
    video = cv2.VideoCapture(file)
    success, image = video.read()
    count = 0
    while success:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(128, 128))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.imwrite(os.path.join(output, studentid + "_%d.jpg" % count), gray[y: y + h, x: x + w])
            count += 1
            if count == 30:
                return True
        success, image = video.read()

    return False


def train_image(path: str, recognizer, face_detector):
    """
    Train image
    :param path: Path to images
    :param recognizer: LBPHFaceRecognizer
    :param face_detector: CascadeClassifier
    :return: Number of trained faces and trained ids
    """
    # Get all images
    image_paths = [os.path.join(path, f) for f in os.listdir('images')]

    face_samples = []
    ids = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_numpy = np.array(img, 'uint8')

        studentid = int(os.path.split(image_path)[-1].split('_')[0])
        faces = face_detector.detectMultiScale(img_numpy)

        # Since Cascade Classifier may detect multiple faces, get the biggest one
        max_square = -1
        for (x, y, w, h) in faces:
            if w * h > max_square:
                max_square = w * h
                _x = x
                _y = y
                _w = w
                _h = h

        face_samples.append(img_numpy[_y: _y + _h, _x:_x + _w])
        ids.append(studentid)

    recognizer.train(face_samples, np.array(ids))

    # Save the model
    recognizer.write('trained.yml')

    # Count trained faces
    count_faces = len(face_samples)
    # Count trained ids
    count_ids = len(np.unique(ids))

    return count_faces, count_ids


def base64_to_image(data):
    """
    Decode base64 image to openCV-compatible format
    :param data: base64
    :return: opencv-compatible format. Grayscale
    """
    starter = data.find(',')
    image_data = data[starter + 1:]
    image_data = bytes(image_data, encoding="ascii")
    im = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
    open_cv_image = np.array(im)
    return open_cv_image


def face_recognition(image, face_detector, recognizer, students: dict):
    faces = face_detector.detectMultiScale(
        image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(128, 128)
    )

    if len(faces) == 0:
        return None, None, None

    # Since Cascade Classifier may detect multiple faces, get the biggest one
    max_square = -1
    for (x, y, w, h) in faces:
        if w * h > max_square:
            max_square = w * h
            _x = x
            _y = y
            _w = w
            _h = h

    out_image = image.copy()
    cv2.rectangle(out_image, (_x, _y), (x + w, y + h), (0, 0, 0), 2)

    id, confidence = recognizer.predict(image[_y: _y + h, _x: _x + w])

    # The smaller the confidence, the better the match?
    if confidence < 100:
        id = str(id)
        name = students.get(id)
        confidence = "    {0}%".format(round(100 - confidence))
    else:
        id = None
        name = None
        confidence = "    {0}%".format(round(100 - confidence))

    cv2.putText(out_image, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.putText(out_image, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    return id, name, out_image


def get_image_as_base64(image):
    """
    Convert an image to base64 format
    :param image: openCV image
    :return: base64
    """
    _, im_arr = cv2.imencode('.jpg', image)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    string_b64 = str(im_b64).split('\'')[1]
    return string_b64
