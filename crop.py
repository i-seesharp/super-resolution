import numpy as np
import matplotlib.pyplot as plt
import cv2

def crop_and_shrink(img_name, input_dir, output_dir):

    img = cv2.imread(input_dir+img_name+".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[0], img.shape[1]

    x = 118*2
    y = 124*2

    mid_h = h//2
    mid_w = w//2

    new = img[mid_h-2*x+75:mid_h+75,mid_w-y+50:mid_w+y+50]
    new = cv2.cvtColor(new, cv2.COLOR_RGB2BGR)
    new_lr = cv2.resize(new, (124,118))
    cv2.imwrite(output_dir+img_name+"cropped_lr.png",new_lr)
