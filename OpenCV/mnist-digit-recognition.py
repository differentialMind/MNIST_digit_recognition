#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,datasets,models
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns


# In[2]:


(train_images,train_labels),(test_images,test_labels)=datasets.mnist.load_data()
train_images=train_images.reshape((60000,28,28,1))/255.0
test_images=test_images.reshape((10000,28,28,1))/255.0
print("TRAIN IMAGES: ",train_images.shape)
print("TEST IMAGES: ",test_images.shape)


# In[3]:


model=Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')])


# In[4]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
    )
model.summary()


# In[5]:


epochs=10
history=model.fit(
    train_images,train_labels,
    epochs=epochs,
    validation_data=(test_images, test_labels)
    )


# In[6]:


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs_range=range(epochs)
plt.figure(figsize=(8,8))
plt.plot(epochs_range,acc,label='Training Accuracy')
plt.plot(epochs_range,val_acc,label='Validation Accuracy')
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,val_loss,label='Validation Loss')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy/Loss')
plt.show()


# In[7]:


image=train_images[1].reshape(1,28,28,1)
prediction=np.argmax(model.predict(image),axis=-1)
plt.imshow(image.reshape(28,28),cmap='gray')
print('Prediction of model:',prediction[0])


# In[8]:


images=test_images[1:5]
for i,test_image in enumerate(images,start=1):
    prediction=np.argmax(model.predict(test_image.reshape(1,28,28,1)),axis=-1)
    plt.subplot(220+i)
    plt.axis('off')
    plt.title("Predicted: {}".format(prediction[0]))
    plt.imshow(test_image.reshape(28, 28),cmap='gray')
    plt.show()


# In[9]:


model.save("mnist_cnn.h5")
loaded_model=models.load_model("mnist_cnn.h5")
image=train_images[2].reshape(1,28,28,1)
prediction=np.argmax(loaded_model.predict(image),axis=-1)
plt.imshow(image.reshape(28,28),cmap='gray')
print('Prediction of loaded model:',prediction[0])


# In[10]:


test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
print("\nTest accuracy:",test_acc)


# In[11]:


y_pred=np.argmax(model.predict(test_images),axis=-1)
cm=confusion_matrix(test_labels,y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[12]:


print("\nClassification Report:\n")
print(classification_report(test_labels,y_pred))

