import time

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ipywidgets import interact, widgets
from IPython.display import clear_output
from os import listdir
from os.path import isfile, join
import seaborn as sns
sns.set(color_codes=True)
import random

from PIL import Image



df_train = pd.read_csv("../Bluewhale/train.csv")
df_train.head()
print("******************Test data format****************************************************")
print(df_train.head())
print("**********************************************************************")



my_path = "../Bluewhale/test"

only_images = [f for f in listdir(my_path) if isfile(join(my_path, f))]

figWidth = figHeight = 10
#sample a couple of pictures
numSampled = 4
sampledPicNames = random.sample(os.listdir(my_path),numSampled)
#then read the images
readImages = [mpimg.imread(my_path + os.sep + sampledPicNames[i])
             for i in range(len(sampledPicNames))]
#then plot
fig, subplots = plt.subplots(2,2)
fig.set_size_inches(figWidth,figHeight)
for i in range(len(readImages)):
    subplots[int(i / 2),i % 2].imshow(readImages[i])
plt.show()


print("**********************************************************************")

print("Train_data image count: ",df_train["Image"].count())

print("**********************************************************************")

print("**********************************************************************")

print("Test_data image count: ",len(only_images))

print("**********************************************************************")

print("**********************************************************************")

print("types of whales: ",len(df_train.groupby(["Id"]).count()))

print("**********************************************************************")

print("******************Count of images in every type****************************************************")

print(df_train.groupby(["Id"]).count().sort_values(by=["Image"], ascending=False))

print("**********************************************************************")

