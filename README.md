# Imagenet Models in Flask

Try different deep learning models for object recognition using Flask.

### Set environment

Using Anaconda:

```
conda create -n imagenet_models_flask
source activate imagenet_models_flask
```

```
pip install tensorflow
conda install -c menpo opencv3
conda install -c anaconda flask
```

### Run

```
cd webapp
export FLASK_APP=webapp.py
flask run
```