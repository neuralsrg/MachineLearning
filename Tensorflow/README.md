# Tensorflow Notes

### Settings and Tensors
Notebook contains:
* tf tensors; built-in operations 
* Google Colab settings (GPU etc.)

### [hand_signs]manual
Notebook contains:
* h5py brief dataset loading tutorial 
* neural network manual implementation with Tensorflow
* tf.keras.metrics.CategoricalAccuracy(), tf.keras.losses.categorical_crossentropy
* the same model using tensorflow 

### [hand_signs]tf
Notebook contains:
* gradient checking for manual and tf implementations (bug fix)
* saving / loading model

### [tf]some_nets
Notebook contains:
* Pandas data hangling & sklearn column transformer
* Linear regression in tf
* Binary Classification 
* Plotting decision-boundaries
* Learning rate scheduler
* Multiclass classification problem (on MNIST)
* Confusion matrix

### [cnn]tf
* CNN model in tensorflow 
* manual training with layers' outputs caching
* Tensorboard features

### [tf]resnet_50
* model creating with model 
* residual blocks in tf with tf.keras.layers.Add()
* cv2 tricks for scaling images

### [tutorial] Transfer learning & fine tuning 
* downloading data with `tf.keras.utils.get_file`
* preparing dataset with `tf.keras.utils.image_dataset_from_directory`
* data augmentation with `tf.keras.layers.{RandomFlip/RandomRotation}`
* downloading pretrained model
* fine-tuning top level layers of the pretrained model AFTER training classification head
(by setting `initial_epoch` parameter of `fit()`)

### [u_net]semantic_segmentation
* `kaggle` download API
* `image preprocessing` in tf
