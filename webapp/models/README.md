Well, we need image recognition models to make the magic happen.

### Download and use Inception ResNet V2 model

You can run `download_and_save_inceptionresnet.py` to download the InceptionResNetV2 checkpoints and save them into the /tmp/ folder.
But first, you need to clone the repository of TensorFlow models so you can use some methods not included in TensorFlow.

```
cd path/for/cloning/tensorflow/models
git clone https://github.com/tensorflow/models.git
```

```
cd /path/to/webapp/models
python download_and_save_inceptionresnet.py
```

But, **hey**: you need to set the variables `models_slim_dir` and `download_dir` inside of `download_and_save_inceptionresnet.py` before running it.

And then just proceed to run the WebApp:

```
cd ..
export FLASK_APP=webapp.py
flask run
```

### Other models

Hopefully (not tested yet) you can download easily all the models from [here](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models) changing just a bit of the code in `download_and_save_inceptionresnet.py`.

### Freeze model

Moreover, you can freeze the InceptionResNetV2 graph using `freeze_inceptionresnet.py`.