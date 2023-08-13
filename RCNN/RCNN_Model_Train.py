from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

train_dir = 'D:\PRML Major Project\Train'
val_dir = 'D:\PRML Major Project\Validation'
test_dir = 'D:\PRML Major Project\Test'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

val_generator = val_test_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

test_generator = val_test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

rpn_model = Sequential()
rpn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
rpn_model.add(MaxPooling2D((2, 2)))
rpn_model.add(Conv2D(64, (3, 3), activation='relu'))
rpn_model.add(MaxPooling2D((2, 2)))
rpn_model.add(Conv2D(128, (3, 3), activation='relu'))
rpn_model.add(MaxPooling2D((2, 2)))

feature_input = Input(shape=rpn_model.output_shape[1:])
x = Flatten()(feature_input)
x = Dense(256, activation='relu')(x)
feature_model = Model(inputs=feature_input, outputs=x)

classifier_input = Input(shape=feature_model.output_shape[1:])
x = Dense(36, activation='softmax')(classifier_input)
classifier_model = Model(inputs=classifier_input, outputs=x)

combined_model = Model(inputs=rpn_model.input, outputs=classifier_model(feature_model(rpn_model.output)))

combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = combined_model.fit(train_generator, steps_per_epoch=train_generator.samples // BATCH_SIZE, epochs=1000, validation_data=val_generator, validation_steps=val_generator.samples // BATCH_SIZE)

test_loss, test_acc = combined_model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print('Test accuracy:', test_acc)

model_save_path = 'D:\PRML Major Project\Models\RCNN_Model.h5'
combined_model.save(model_save_path)
