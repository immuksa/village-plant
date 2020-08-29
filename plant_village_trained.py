
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import cv2
from os import listdir
import matplotlib.pyplot as plt
from googlesearch import search 
import webbrowser  
from sklearn.utils import shuffle


# In[2]:


#to be used only when training is completed i.e load trained model
with open('cnn_model.pkl','rb') as f:
    model = pickle.load(f)
with open('label_transform.pkl','rb') as d:
    labael_binarizer = pickle.load(d)
classes = labael_binarizer.classes_


# In[6]:


def pred_disease():
    adr = input("Enter adress of plant leave:")
    image = cv2.imread(f'{adr}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256,256))
    std_img = np.array(image, dtype = np.float16)/255.0
    std_img = std_img.reshape(-1,256,256,1)
    out = model.predict(std_img).ravel()
    diseas = classes[out.argmax()]
    diseas = diseas.replace('_'," ")
    print('plant is suffuering from :: {}'.format(diseas))
    diseas = diseas.replace('_'," ")
    query = diseas +"Treatment and cure"
    data = search(query, tld="co.in", num=10, stop=1, pause=2)
    url = ''
    for j in search(query, tld="co.in", num=10, stop=2, pause=2): 
        url = j
        print(url)
    webbrowser.open(url, new=0, autoraise=True)


# In[7]:


pred = True
while pred:
    x = int(input('\n\n1.Enter 1 to Predict Disease\n2.Enter 2 to Exit\n'))
    if x== 1:
        pred_disease()
    elif x == 2:
        pred = False

