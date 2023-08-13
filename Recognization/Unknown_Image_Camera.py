import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog

def get_file_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def predict_image_class():
    RCNN_Model = load_model('D:\PRML Major Project\Models\RCNN_Model.h5')
    CNN_Model = load_model('D:\PRML Major Project\Models\CNN_Model.h5')
    SSD_Model = load_model('D:\PRML Major Project\Models\SSD_Model.h5')
    arr_model = [RCNN_Model, CNN_Model, SSD_Model]
    class_labels =  ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
    file_path = get_file_path()

    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    
    arr = []
    for model in arr_model:
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        arr.append(predicted_class)
    i = max(set(arr), key=arr.count)
    
    return class_labels[i]

def predict_from_camera():
    RCNN_Model = load_model('D:\PRML Major Project\Models\RCNN_Model.h5')
    CNN_Model = load_model('D:\PRML Major Project\Models\CNN_Model.h5')
    SSD_Model = load_model('D:\PRML Major Project\Models\SSD_Model.h5')
    arr_model = [RCNN_Model, CNN_Model, SSD_Model]
    
    class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

    camera = cv2.VideoCapture(0)
    _, image = camera.read()

    
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.

    arr = []
    for model in arr_model:
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        arr.append(predicted_class)
    i = max(set(arr), key=arr.count)
    
    return class_labels[i]

a = True
pred = []
while a:
    print('Select a mode:')
    print('1. Camera mode')
    print('2. Drive mode')
    print('0. Done')
    mode = input('Enter mode number (1 or 2): ')

    if mode == '1':
        pred.append(predict_from_camera())
        
    elif mode == '2':
        pred.append(predict_image_class())
        
    elif mode == '0':
        print('Done')
        a = False
    else:
        print('Please Enter Proper Code No. ')

print('Oh Great!')
print(f'You have {[i for i in pred]}.')
print()
