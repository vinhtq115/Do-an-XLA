from flask import Flask, request, render_template
import json
import os
import cv2

from routing.register import register1
from routing.check_in import check_in1

app = Flask(__name__)
app.register_blueprint(register1)
app.register_blueprint(check_in1)


@app.route('/', methods=['GET'])
def root():
    return 'TODO'

if __name__ == '__main__':
    app.run(debug=True, port=8000)
