
# Cats VS. Dogs Classification MobileNetV2

- End-To-End Deep Learning Project on Dog/Cat Classification 
- Data Augmentation
- Bulilding a CNN using MobileNetV2 Pretrained Model Architecture
- Create a Web App that serves the model using Python `streamlit`
- App Deployment on `Streamlit` Cloud


## Dataset

The dataset used for this project is obtained from [Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

The dataset contains Two folders:
- test_set
    - Cats
    - Dogs
- training_set
    - Cats
    - Dogs

I combined all of them in cats and dogs folders then I used `splitfolders` module to create training, validation, and testing folders.
```
pip install split-folders
```

```
import splitfolders

input_folder = "D:/data_all"
output_folder = "D:/data_split"

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7,.2,.1), group_prefix=None)
```

## Setup

### Libraries
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import random
```

### Configuration
```
IMAGE_SIZE = 250
EPOCHS = 5
CHANNELS = 3
BATCH_SIZE = 256

TRAIN_PATH = "D:/cats_and_dogs/data_split/train"
VAL_PATH = "D:/cats_and_dogs/data_split/val"
TEST_PATH = "D:/cats_and_dogs/data_split/test"

```

## Data Augmentation
```
train = ImageDataGenerator(rescale = 1./255,
                           rotation_range = 25,
                           shear_range = 0.5,
                           zoom_range = 0.5,
                           width_shift_range = 0.2,
                           height_shift_range=0.2,
                           horizontal_flip=True
                          )

validation = ImageDataGenerator(rescale = 1./255,
                           rotation_range = 25,
                           shear_range = 0.5,
                           zoom_range = 0.5,
                           width_shift_range = 0.2,
                           height_shift_range=0.2,
                           horizontal_flip=True
                          )
```


## Importing Data
```
train_data = train.flow_from_directory(TRAIN_PATH, 
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                       batch_size=BATCH_SIZE, 
                                       class_mode="binary",
                                       seed=42)

val_data = validation.flow_from_directory(VAL_PATH, 
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                       batch_size=BATCH_SIZE, 
                                       class_mode="binary",
                                       seed=42)
```

## Base Model
```
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
```

## My Model
```
model = Sequential(
    [
        base_model,
        Flatten(),
        Dense(1, activation='sigmoid')  
    ]
)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
```

## Training and Validation
```
history = model.fit(train_data,
                    steps_per_epoch = 7018//BATCH_SIZE,
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    validation_data = val_data
                   )
```

[![Screenshot-2022-09-23-194123.png](https://i.postimg.cc/rF3hJbnP/Screenshot-2022-09-23-194123.png)](https://postimg.cc/YhNf2Xw6)


## Testing
### Testing Data
```
testing = ImageDataGenerator(rescale = 1./255)
test_data = testing.flow_from_directory(TEST_PATH, 
                                       target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                       batch_size=BATCH_SIZE, 
                                       class_mode="binary",
                                       seed=42)
```

### Testing
```
results = model.evaluate(test_data)
print('Test loss:', results[0])
print('Test accuracy:', results[1])
```

```
4/4 [==============================] - 5s 1s/step - loss: 0.0875 - accuracy: 0.9791
Test loss: 0.08745451271533966
Test accuracy: 0.9791044592857361
```

## APP Deployed
https://96ibman-cats-vs-dogs-streamlit-app-uzievd.streamlit.app/

## Authors

- [Ibrahim Nasser](https://github.com/96ibman)


## Connect With Me!
- [Website](https://ibrahim-nasser.com/)
- [LinkedIn](https://www.linkedin.com/in/ibrahimnasser96/)
- [Twitter](https://twitter.com/mleng_ibrahimy)
- [YT Channel](https://www.youtube.com/channel/UC7N-dy3UbSBHnwwv-vulBAA)

