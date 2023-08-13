import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

def evaluate_model(model, test_dir):
    IMG_SIZE = (224, 224)

    BATCH_SIZE = 32

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

    predictions = model.predict(test_generator)
    true_labels = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    accuracy = np.mean(predictions.argmax(axis=1) == true_labels)
    report = classification_report(true_labels, predictions.argmax(axis=1), target_names=class_names)

    return accuracy,  report

RCNN_Model = load_model('D:\PRML Major Project\Models\RCNN_Model.h5')
CNN_Model = load_model('D:\PRML Major Project\Models\CNN_Model.h5')
SSD_Model = load_model('D:\PRML Major Project\Models\SSD_Model.h5')

print("RCNN Model:")
RCNN_acc, RCNN_report = evaluate_model(RCNN_Model, 'D:\PRML Major Project\Test')
print("Accuracy: ", RCNN_acc)
print("Report", RCNN_report)

print("CNN Model:")
CNN_acc, CNN_report = evaluate_model(CNN_Model, 'D:\PRML Major Project\Test')
print("Accuracy: ", CNN_acc)
print("Report", CNN_report)

print("SSD Model:")
SSD_acc, SSD_report = evaluate_model(SSD_Model, 'D:\PRML Major Project\Test')
print("Accuracy: ", SSD_acc)
print("Report", SSD_report)