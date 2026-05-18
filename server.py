from flask import Flask, request, jsonify , render_template
from werkzeug.utils import secure_filename
import os
#model imports
import cv2
import torch
import numpy as np
import string
from model import CRNN 
import requests
import urllib


app = Flask(__name__)
device = torch.device("cpu")
model = CRNN(45)
state = torch.load("\\model_epoch_20.pth",map_location='cpu')
model.load_state_dict(state)
idx2char={i+1: c for i, c in enumerate(string.digits + string.ascii_lowercase + "+-*/=()^")}
idx2char[0] = "<BLANK>"
model.eval()
APPID = 0

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home_page():
    return render_template('index.html')

def decode_greedy(logits):
    preds = logits.softmax(-1).argmax(-1).cpu().numpy().T
    results = []

    for seq in preds:
        prev = -1
        text = ""
        for idx in seq:
            if idx != prev and idx != 0:
                text += idx2char[idx]
            prev = idx
        results.append(text)

    return results

def predict(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1024, 256))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(img)
        return solve(decode_greedy(logits)[0])
    
def solve(eq):
    if(APPID==0):
        return {'sol':"sorry out of keys",'eq':eq}
    url = f"https://api.wolframalpha.com/v2/query?appid={APPID}&input={urllib.parse.quote(eq)}&output=json"
    response = requests.get(url).json()
    return {'sol':response,'eq':eq}

@app.route('/upload', methods=['POST'])
def upload_image():
    print("request in")
    if 'image' not in request.files:
        print("invalid request")
        return jsonify({'error': 'לא נשלחה תמונה'}), 400

    file = request.files['image']
    if file.filename == '':
        print("invalid request name")
        return jsonify({'error': 'שם קובץ ריק'}), 400

    print("loading image")
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    res = predict(filepath)
    print(f"predict = {res}")
    response = {
        'sulotion': res.sol,
        'message': res.eq,
        'filename': filename,
        'path': filepath
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
