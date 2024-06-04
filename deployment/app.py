from flask import Flask, render_template, request
from ultralytics import YOLO


app = Flask(__name__)
model = YOLO('C:/Users/kunal/OneDrive/Desktop/local_env/local_env/runs/classify/train5/weights/last.pt')

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)



      
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5500, debug=True)