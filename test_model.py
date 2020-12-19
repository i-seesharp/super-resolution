import numpy as np
import cv2
import matplotlib.pyplot as plt
import model
from model import GAN_img
from PIL import Image
import crop

#Please note that the model has already been trained, so we
#don't retrain here but use weights obtained after 2-3 days of training
#to demonstrate the implementation and the respective outputs.



#This generator takes a low resolution(LR) and upsamples it to 4x the original size
G = model.generator()
G.load_weights("weights/generator.h5")

test_images = ["0816","0817","0818"] #image numbers from dir sample_test_ds
input_dir = "sample_test_ds/"
output_dir = "model_output/"

for name in test_images:
    crop.crop_and_shrink(name,input_dir,output_dir) #Convert Image to Low Resolution
    img = np.array(Image.open(output_dir+name+"cropped_lr.png")) #Grab the low resolution image we just made
    out = GAN_img(G, img)
    out_img = Image.fromarray(out.numpy())
    out_img.save(output_dir+name+"cropped_sr.png")
    

    plt.imshow(out)
    plt.show()
    
