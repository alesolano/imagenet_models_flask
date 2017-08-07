# Imagenet Models in Flask

Try different deep learning models for object recognition using Flask.

## Set environment

#### Ubuntu 16.04 (tested)

```
conda create -n imagenet_flask python=3.6
source activate imagenet_flask
```

```
pip install tensorflow
conda install -c anaconda flask
```

## Download models

See README.md in `./webapp/models/`.

## Run

#### Ubuntu 16.04 (tested)

```
cd webapp
export FLASK_APP=webapp.py
flask run
```