from flask import Blueprint, request, render_template
import json
import os

check_in1 =  Blueprint('check_in', __name__)

@check_in1.route('/check_in', methods=['GET', 'POST'])
def check_in():
    if request.method == 'POST':
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
        if request.files.get('image') is None:
            resp = {
                'status': 3,
                'message': '"image" not found.'
            }
            return json.dumps(resp)

        file = request.files['image']
        filename = 'check_in' + '.jpg'
        file.save(os.path.join('uploads', filename))
    
    elif request.method == 'GET':
        return render_template('att.html')
    return ''