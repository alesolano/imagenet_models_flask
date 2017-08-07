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

#### MacOS

```
conda create -n imagenet_flask python=3.6
source activate imagenet_flask
```
```
conda install -c anaconda flask
conda install flask-wtf tensorflow
pip install flask-bootstrap
```

## Download models

See README.md in `./webapp/models/`.

## Run

#### Ubuntu 16.04 (tested) and MacOS

```
cd webapp
export FLASK_APP=webapp.py
export FLASK_DEBUG=1
flask run
```

#### Windows 10

```
cd webapp
set FLASK_APP=webapp.py
set FLASK_DEBUG=1
flask run
```
