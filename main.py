from flask import Flask, request, render_template
import json
import os
import cv2
import glob
import api


app = Flask(__name__, template_folder='templates')

# CV2 face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Student ID and name
students = None
with open('name.json', encoding='utf-8') as json_file:
    students = json.load(json_file)


# Index page
@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')


# Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        # Check if the POST request has the file part
        if len(request.files) == 0:  # If 0 files were found, return error
            resp = {
                'status': 1,
                'message': 'No files were uploaded.'
            }
            return json.dumps(resp)
        elif len(request.files) != 1:  # More than 1 file were found, return error
            resp = {
                'status': 2,
                'message': 'Too many files were uploaded. Only 1 is needed.'
            }
            return json.dumps(resp)
        if request.form.get('studentid') is None or request.form.get('studentid') == '':
            resp = {
                'status': 3,
                'message': 'Missing student id.'
            }
            return json.dumps(resp)

        if request.files.get('video') is None:
            resp = {
                'status': 4,
                'message': '"video" not found.'
            }
            return json.dumps(resp)

        # Get student ID
        student_id = request.form['studentid']

        if not os.path.exists('images'):
            os.mkdir('images')

        if len(glob.glob('images/' + str(student_id) + '*.jpg')):
            resp = {
                'status': 5,
                'message': 'Student already registered/uploaded.'
            }
            return json.dumps(resp)

        file = request.files['video']
        filename = student_id + '.webm'

        # Save video to uploads folder
        if not os.path.exists('uploads'):
            os.mkdir('uploads')

        file.save(os.path.join('uploads', filename))

        # Extract videos to images
        success = api.extract_videos_to_images(file=os.path.join('uploads', filename),
                                               output='images', studentid=student_id,
                                               face_detector=face_detector)
        os.remove(os.path.join('uploads', filename))
        if not success:
            for file in glob.glob('images/' + student_id + '*.jpg'):
                os.remove(file)
            resp = {
                'status': 6,
                'message': 'Failed to recognize minimum number of faces in video. Please try again.'
            }
            return json.dumps(resp)
        else:
            if os.path.isfile('trained.yml'):
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read('trained.yml')
                # Check potential problems
                mis_count = 0
                for file in glob.glob('images/' + student_id + '*.jpg'):
                    m_id = api.check_potential_errors(cv2.imread(file, cv2.IMREAD_GRAYSCALE), recognizer, face_detector)
                    if m_id is not False:
                        mis_count += 1
                if mis_count > 10:
                    for file in glob.glob('images/' + student_id + '*.jpg'):
                        os.remove(file)
                    resp = {
                        'status': 7,
                        'message': 'Your face is too similar to someone else. Please try again.'
                    }
                    return json.dumps(resp)
            resp = {
                'status': 0,
                'message': 'Success. Please wait for school admin to register your face into database.'
            }

        return json.dumps(resp)


# Admin page for training
@app.route('/admin', methods=['GET'])
def admin():
    return render_template('admin.html')


# Train image
@app.route('/train', methods=['GET'])
def train():
    # If no images, do not train
    if len(glob.glob('images/*.jpg')) == 0:
        resp = {
            'status': 1,
            'message': 'No images to train. Maybe you wanna call your students to register their face?'
        }
        return json.dumps(resp)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    count_faces, count_ids = api.train_image(path='images', recognizer=recognizer, face_detector=face_detector)
    resp = {
        'status': 0,
        'message': 'Finished training. ' + str(count_faces) + ' of ' + str(count_ids) + ' students have been trained.'
    }
    return json.dumps(resp)


# Admin page showing errors
@app.route('/error', methods=['GET'])
def error():
    return render_template('error.html')


# Attendance page for student
@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'GET':
        return render_template('attendance.html')
    elif request.method == 'POST':
        # Check if the POST request has the image part
        if request.form.get('image') is None or request.form.get('image') == '':
            resp = {
                'status': 1,
                'message': 'Missing image.'
            }
            return json.dumps(resp)
        image = request.form['image']
        img = api.base64_to_image(image)
        if not os.path.isfile('trained.yml'):
            resp = {
                'status': 4,
                'message': 'You arrived too soon. The system is not ready to use.'
            }
            return json.dumps(resp)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trained.yml')
        id, name, out_image = api.face_recognition(img, face_detector, recognizer, students)
        if id is None and name is None and out_image is None:
            # No faces were found
            resp = {
                'status': 2,
                'message': 'No faces were found. Please try again.'
            }
            return json.dumps(resp)
        elif id is None and name is None and out_image is not None:
            # A face was found but not recognized
            resp = {
                'status': 3,
                'message': 'A face was found but cannot be recognized. Please try again.',
                'image': 'data:image/jpeg;base64,' + api.get_image_as_base64(out_image)
            }
            return json.dumps(resp)
        resp = {
            'status': 0,
            'message': 'Success. Student ID: ' + str(id) + ' - ' + str(name),
            'image': 'data:image/jpeg;base64,' + api.get_image_as_base64(out_image)
        }
        return json.dumps(resp)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
