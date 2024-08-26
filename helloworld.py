from flask import Flask,render_template, request, send_file
from ultralytics import YOLO
import base64
from PIL import Image
import io

app = Flask(__name__)
@app.route('/')
def helloworld():
    return'1212'

@app.route('/static')
def static_demo():
    return send_file("static/gb.jpg", mimetype='image/jpeg')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        img_base64 = request.form.get('picture')
        image = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(image))

        return '1212'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)