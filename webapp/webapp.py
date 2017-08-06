import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from config import *
from flask import render_template

app = Flask(__name__)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import time
from predicting import Predictor
predictor = Predictor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect('/uploaded')
            return redirect(url_for('uploaded', filename=filename))
    return render_template("index.html")

@app.route('/uploaded')
def uploaded():
    begin = time.time()
    pred_class, pred_score = predictor.evaluate(
        filename=request.args.get('filename'),
        model_name='inception_resnet_v2',
        graph_type='checkpoints')
    end = time.time()
    print("\nElapsed time: %0.5f seconds." % (end-begin))
    
    return render_template("uploaded.html",
        pred_class_0=str(pred_class[0]),
        pred_class_1=str(pred_class[1]),
        pred_class_2=str(pred_class[2]),
        pred_class_3=str(pred_class[3]),
        pred_class_4=str(pred_class[4]),
        pred_score_0=str(pred_score[0]),
        pred_score_1=str(pred_score[1]),
        pred_score_2=str(pred_score[2]),
        pred_score_3=str(pred_score[3]),
        pred_score_4=str(pred_score[4]))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    