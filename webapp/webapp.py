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
import predicting

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

        '''Optional: check if the post request has import_options or ml_models.
        But more elegant is to do it in the front end.

        print (request.form)
        print (type(request.form))

        if 'import_options' not in request.form:
            print('No importing options selected')
            return redirect(request.url)

        if 'ml_models' not in request.form:
            print('No ml_models selected')
            return redirect(request.url)
        '''

        #Variables for request parameters:
        file = request.files['file']
        import_options = request.form['import_options']
        ml_models = request.form['ml_models']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect('/uploaded')
            return redirect(url_for('uploaded', filename=filename, import_options=import_options, ml_models=ml_models))
    return render_template("index.html")

@app.route('/uploaded')
def uploaded():
    import_options = request.args.get('import_options')
    ml_models = request.args.get('ml_models')
    print("selected import option:",import_options)
    print("selected ml_models:",ml_models)
    return render_template("uploaded.html",
        ml_models=ml_models,
        import_options=import_options
                           )

    '''
    begin = time.time()
    pred_class, pred_score = predicting.evaluate(filename=request.args.get('filename'))
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
    '''

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    