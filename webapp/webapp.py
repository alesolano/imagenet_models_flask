import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from config import *
from flask import render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect('/uploaded')
            return redirect(url_for('uploaded', filename=filename))
    return render_template("index.html")

@app.route('/uploaded')
def uploaded():
    import predicting
    pred_class, pred_certain = predicting.evaluate(filename=request.args.get('filename'))
    return render_template("uploaded.html",
        pred_class_0=str(pred_class[0]),
        pred_class_1=str(pred_class[1]),
        pred_class_2=str(pred_class[2]),
        pred_class_3=str(pred_class[3]),
        pred_class_4=str(pred_class[4]),
        pred_certain_0=str(pred_certain[0]),
        pred_certain_1=str(pred_certain[1]),
        pred_certain_2=str(pred_certain[2]),
        pred_certain_3=str(pred_certain[3]),
        pred_certain_4=str(pred_certain[4]))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    