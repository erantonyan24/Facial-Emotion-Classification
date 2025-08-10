import cv2
import keras as kr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

model = kr.models.load_model('ProjectWebCam/emotion.h5')
print('loaded')

names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
nums = [0, 1, 2, 3, 4, 5, 6]
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # ret is a boolean indicating success, frame is the captured image

    if not ret: # Break the loop if reading frame fails
        print("Failed to grab frame")
        break

    img = frame
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255
    img = img.reshape(1, 48, 48, 1)

    pred = model.predict(img)
    number = np.argmax(pred)
    text = names[number]



    cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Break the loop if 'q' key is pressed (ASCII value of 'q' is 113, or use 27 for Escape key)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
