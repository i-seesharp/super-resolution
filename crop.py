import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("sample_test_ds/0816.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = img.shape[0], img.shape[1]

x = 118*2
y = 124*2

mid_h = h//2
mid_w = w//2

new = img[mid_h-2*x+75:mid_h+75,mid_w-y+50:mid_w+y+50]
new = cv2.cvtColor(new, cv2.COLOR_RGB2BGR)
cv2.imwrite("cropped_hr.png",new)
