import tensorflow as tf
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_dir = 'D:\PRML Major Project\Train'
test_dir = 'D:\PRML Major Project\Test'
val_dir = 'D:\PRML Major Project\Validation'

batch_size = 32
img_height = 224
img_width = 224

train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True) 

train_generator = train_data_gen.flow_from_directory(directory=train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', shuffle=True)

val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 

val_generator = val_data_gen.flow_from_directory(directory=val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(directory=test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

num_classes = len(train_generator.class_indices)
classification_head = tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(num_classes, activation='softmax')])
model = classification_head

epochs = 10
steps_per_epoch = train_generator.samples // batch_size
validation_steps = val_generator.samples // batch_size

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_generator, validation_steps=validation_steps)

model.save('D:\PRML Major Project\Models\SSD_Model.h5')
