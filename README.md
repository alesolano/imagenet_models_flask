# Imagenet Models in Flask

Try different deep learning models for object recognition using Flask.

### Set environment

Using Anaconda:

```
conda create -n imagenet_models_flask
source activate imagenet_models_flask
```

```
#pip install tensorflow
conda install -c menpo opencv3
conda install -c anaconda flask
conda install flask-wtf tensorflow
```

### Run

```
Mac:
cd webapp
export FLASK_APP=webapp.py
export FLASK_DEBUG=1
flask run

Win:
cd webapp
set FLASK_APP=webapp.py
set FLASK_DEBUG = 1
flask run
```