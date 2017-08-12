Well, we need image recognition models to make the magic happen.
But first, you need to clone the repository of TensorFlow models so you can use some methods not included in TensorFlow.

```
cd path/for/cloning/tensorflow/models
git clone https://github.com/tensorflow/models.git
```

### Download and use Inception ResNet V2 model


#### Model checkpoints

You can run `download_and_save_inceptionresnet.py` to download the InceptionResNetV2 checkpoints and save them into the /models/ folder.

```
cd /path/to/webapp/models
python download_and_save_inceptionresnet.py
```

But, **hey**: you need to set the variables `models_slim_dir` and `download_dir` inside of `download_and_save_inceptionresnet.py` before running it.


- models\_slim\_dir: is located inside the cloned tensorflow models folder:

	clone_dir/models/slim from: git clone https://github.com/tensorflow/models.git


- download\_dir: is any temporal directory to save the raw file for the model



#### Freeze model

Moreover, you can freeze the InceptionResNetV2 graph using `freeze_inceptionresnet.py`. You will find that the loading speed is multiplied by a factor of 3, or even 5.

```
cd /path/to/webapp/models
python freeze_inceptionresnet.py
```


And then just proceed to run the WebApp:

```
cd ..
export FLASK_APP=webapp.py
flask run
```

#### Compiled model
TODO

### Download and use MobileNets model

#### Model checkpoints and freeze model

For MobileNets, it's a little bit more tricky. Until we discover a way to make downloading more general, you can use the script 'download\_save\_and\_freeze\_mobilenet.py'.

But, **hey**: do not forget to set the variables `models_slim_dir` and `download_dir` as previously described. 

```
cd /path/to/webapp/models
python download_save_and_freeze_mobilenet.py
```


#### Compiled model

The compiled C++ code for Imagenet models can be found [here](https://www.dropbox.com/s/mi9gtxqvgzy7gh8/imagenet_cc?dl=0). Just download it and place it in the `models` folder. Nonetheless, you can also compile it by yourself or take a look at the code.

Compile TensorFlow code in C++ could be a mess, but it's interesting because of its great performance. In order to do it, you need to install Bazel and clone the TensorFlow repository: `git clone --recursive https://github.com/tensorflow/tensorflow`. The code for Imagenet models is in the folder `compiled`. [This Medium post](https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f) may be useful. But hey, **remember that you don't need to do it**, it's already compiled [here](https://www.dropbox.com/s/mi9gtxqvgzy7gh8/imagenet_cc?dl=0).



### Other models

Hopefully (not tested yet) you can download easily all the models from [here](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models) changing just a bit of the code in `download_and_save_inceptionresnet.py`.
