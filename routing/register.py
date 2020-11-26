from flask import Blueprint, request, render_template
import json
import os

from model.face_detector import extract_videos_to_images

register1 = Blueprint('register', __name__)

@register1.route('/register', methods=['GET', 'POST'])
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
        return render_template('index.html')
    return ''
