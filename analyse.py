import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ipywidgets import interact, widgets
from IPython.display import clear_output
from os import listdir
from os.path import isfile, join
import seaborn as sns
sns.set(color_codes=True)

from PIL import Image



df_train = pd.read_csv("../Bluewhale/train.csv")
df_train.head()
print("******************Test data format****************************************************")
print(df_train.head())
print("**********************************************************************")



my_path = "../Bluewhale/test"

only_images = [f for f in listdir(my_path) if isfile(join(my_path, f))]

#@interact(ix=widgets.IntSlider(min=0, max=len(only_images), step=1, value=0, continuous_update=False))
def show_test_images(ix):
    
    clear_output(wait=True)
    
    how_many = 9
    hm_sq = int(np.sqrt(how_many))
    
    f, axes = plt.subplots(hm_sq, hm_sq)
    f.set_size_inches(18, 12)
    
    for nr, i in enumerate(range(ix, ix + how_many)):
        image_path = "../Bluewhale/test/" + only_images[i]
        
        axes[int(nr / hm_sq)][nr % hm_sq].imshow(
            mpimg.imread(image_path)
        )
        
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

