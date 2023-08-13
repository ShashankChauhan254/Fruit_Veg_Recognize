import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tensorflow.keras.models import load_model

def evaluate_model(model, test_dir):
    IMG_SIZE = (224, 224)

    BATCH_SIZE = 32

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
    y_true = []
    y_pred = []
    for i in range(len(test_generator)):
        batch_x, batch_y = test_generator.next()
        y_true.extend(np.argmax(batch_y, axis=1))
        y_pred.extend(np.argmax(model.predict(batch_x), axis=1))
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')

    return accuracy, f1, recall, precision

RCNN_Model = load_model('D:\PRML Major Project\Models\RCNN_Model.h5')
CNN_Model = load_model('D:\PRML Major Project\Models\CNN_Model.h5')
SSD_Model = load_model('D:\PRML Major Project\Models\SSD_Model.h5')

print("RCNN: ")
RCNN_acc, RCNN_f1, RCNN_recall, RCNN_pre = evaluate_model(RCNN_Model, 'D:\PRML Major Project\Test')
print('Accuracy: ', RCNN_acc)
print('F1 Score: ', RCNN_f1)
print('Recall: ', RCNN_recall)
print('Preision: ', RCNN_pre)

print("CNN: ")
CNN_acc, CNN_f1, CNN_recall, CNN_pre = evaluate_model(CNN_Model, 'D:\PRML Major Project\Test')
print('Accuracy: ', CNN_acc)
print('F1 Score: ', CNN_f1)
print('Recall: ', CNN_recall)
print('Preision: ', CNN_pre)

print("SSD: ")
SSD_acc, SSD_f1, SSD_recall, SSD_pre = evaluate_model(RCNN_Model, 'D:\PRML Major Project\Test')
print('Accuracy: ', SSD_acc)
print('F1 Score: ', SSD_f1)
print('Recall: ', SSD_recall)
print('Preision: ', SSD_pre)
