#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models


# In[2]:


import os
print(os.getcwd())
print(os.listdir())


# In[3]:


MODEL_PATH = "mnist_cnn.h5"
model = models.load_model(MODEL_PATH, compile=False)
print("[INFO] Loaded model from disk.")


# In[4]:


drawing = False


# In[5]:


ix,iy=-1,-1
canvas=np.zeros((400, 400),dtype=np.uint8)


# In[6]:


def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,canvas
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas,(x,y),12,(255,255,255),-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.circle(canvas,(x,y),12,(255,255,255),-1)


# In[7]:


def draw_label(img,text,pos,font=cv2.FONT_HERSHEY_DUPLEX,
               scale=1,color=(0,0,255),thickness=2,bg_color=(255,255,255)):
    (w,h),baseline=cv2.getTextSize(text,font,scale,thickness)
    x,y=pos
    cv2.rectangle(img,(x,y-h-baseline),(x+w, y+baseline),bg_color,-1)
    cv2.putText(img,text,(x,y),font,scale,color,thickness)


# In[8]:


menu = np.zeros((300, 600, 3), dtype=np.uint8)
cv2.putText(menu, "Choose Mode:", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
cv2.putText(menu, "Press '1' for Canvas Mode", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
cv2.putText(menu, "Press '2' for Webcam Mode", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
cv2.putText(menu, "Press 'q' to Quit", (50, 250), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Menu", menu)
mode = None


# In[9]:


while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('1'):
        mode = "canvas"
        break
    elif key == ord('2'):
        mode = "webcam"
        break
    elif key == ord('q'):
        cv2.destroyAllWindows()
        exit()
cv2.destroyWindow("Menu")


# In[10]:


if mode == "canvas":
    cv2.namedWindow("Canvas")
    cv2.setMouseCallback("Canvas", draw_circle)
    try:
        while True:
            canvas_display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            cv2.putText(canvas_display, "Draw digits (Press 'p' to predict,'c' to clear,'q' to quit)",
                        (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Canvas", canvas_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas[:] = 0
            elif key == ord('p'):
                
                thresh = cv2.adaptiveThreshold(
                    canvas, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  
                digits = []
                for c in contours:
                    if cv2.contourArea(c) < 100:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    roi = canvas[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi, (28, 28))
                    roi_reshaped = roi_resized.astype("float32").reshape(1, 28, 28, 1) / 255.0

                    probs = model.predict(roi_reshaped, verbose=0)[0]
                    pred = np.argmax(probs)
                    confidence = probs[pred] * 100

                    if confidence > 70:
                        digits.append(str(pred))
                        cv2.rectangle(canvas_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(canvas_display, f"{pred}",
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if digits:
                    number_text = "Number: " + "".join(digits)
                    print(f"[Canvas Prediction] {number_text}")
                    draw_label(canvas_display, number_text,
                               (canvas_display.shape[1]-250, 60),
                               scale=1.2, color=(0,0,255), thickness=3, bg_color=(255,255,255))
                    cv2.imshow("Canvas", canvas_display)
    finally:
        cv2.destroyAllWindows()
elif mode == "webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:   
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 100:
                    x, y, w, h = cv2.boundingRect(c)
                    roi = gray[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi, (28, 28))
                    roi_reshaped = roi_resized.astype("float32").reshape(1, 28, 28, 1) / 255.0
                    probs = model.predict(roi_reshaped, verbose=0)[0]
                    pred = np.argmax(probs)
                    confidence = probs[pred] * 100
                    if confidence > 70:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{pred} ({confidence:.1f}%)",
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        draw_label(frame, f"Digit: {pred}",
                                   (frame.shape[1]-200, 50),
                                   scale=1.5, color=(0,0,255), thickness=3, bg_color=(255,255,255))
            cv2.imshow("Digit Recognition", frame)
            key=cv2.waitKey(1) & 0xFF
            if key==ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

