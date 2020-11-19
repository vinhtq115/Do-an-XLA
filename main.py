from flask import Flask, request, render_template
import json
import os
import cv2

app = Flask(__name__, template_folder='templates')

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


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':

        # Check if the POST request has the file part
        if len(request.files) == 0:  # If 0 files were found, return error
            resp = {
                'status': 1,
                'message': 'No files were uploaded.'
            }
            return json.dumps(resp)
        elif len(request.files) != 1:
            resp = {
                'status': 2,
                'message': 'Too many files were uploaded. Only 1 is needed.'
            }
            return json.dumps(resp)
        if request.form.get('studentid') is None:
            resp = {
                'status': 3,
                'message': 'Missing student id.'
            }
            return json.dumps(resp)

        student_id = request.form['studentid']

        if request.files.get('video') is None:
            resp = {
                'status': 4,
                'message': '"video" not found.'
            }
            return json.dumps(resp)

        file = request.files['video']
        filename = student_id + '.webm'

        file.save(os.path.join('uploads', filename))
        extract_videos_to_images(os.path.join('uploads', filename), 'images', student_id)
        # TODO: cheating detection

    elif request.method == 'GET':
        return render_template('diemdanh.html')
    return ''


@app.route('/', methods=['GET'])
def root():
    return 'TODO'


if __name__ == '__main__':
    app.run()
