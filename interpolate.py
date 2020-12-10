import cv2
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_figure(image, title):
    plt.figure(figsize=(20,35))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()


def interpolate(image, factor):
    image = image.astype(np.int64)
    first_dim = np.zeros((image.shape[0], int(image.shape[1] *factor -factor), 3), dtype = np.int64)
    for i in range(first_dim.shape[0]):
        for j in range(first_dim.shape[1]):
            if j%factor == 0:
                first_dim[i][j][0] = int(image[i][j//factor][0])
                first_dim[i][j][1] = int(image[i][j//factor][1])
                first_dim[i][j][2] = int(image[i][j//factor][2])
                incrementr = int((image[i][j//factor][0] - image[i][j//factor + 1][0])//factor) * -1
                incrementg = int((image[i][j//factor][1] - image[i][j//factor + 1][1])//factor)* -1
                incrementb = int((image[i][j//factor][2] - image[i][j//factor + 1][2])//factor)* -1
            else:
#                 first_dim[i][j][0] = int(image[i][j//2][0])
#                 first_dim[i][j][1] = int(image[i][j//2][1])
#                 first_dim[i][j][2] = int(image[i][j//2][2])
                first_dim[i][j][0] = first_dim[i][j-1][0] + incrementr
                first_dim[i][j][1] = first_dim[i][j-1][1] + incrementg
                first_dim[i][j][2] = first_dim[i][j-1][2] + incrementb
    plot_figure(first_dim, "first")

    final = np.zeros((int(image.shape[0] * factor - factor), int(image.shape[1] * factor - factor), 3), dtype = int)
    for i in range(final.shape[0] - 1):
        for j in range(final.shape[1]):
            if i % factor == 0:
                final[i,:,:] = first_dim[i//factor,:,:]
                incrementr = int((first_dim[i//factor][j][0] - first_dim[i//factor + 1][j][0])//factor)* -1
                incrementg = int((first_dim[i//factor][j][1] - first_dim[i//factor + 1][j][1])//factor)* -1
                incrementb = int((first_dim[i//factor][j][2] - first_dim[i//factor + 1][j][2])//factor)* -1
            else:
                final[i][j][0] = final[i-1][j][0] + incrementr
                final[i][j][1] = final[i-1][j][1] + incrementg
                final[i][j][2] = final[i-1][j][2] + incrementb

    cv2.imwrite("camera_enhance.jpg", final)
    plot_figure(final, "final output")
    print(final.shape)
    
image_1 = cv2.imread('camera.jpg')[...,::-1]
image_1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2RGB)
print(image_1.shape)
plot_figure(image_1, "nature")
interpolate(image_1, 5)


# img_interp.py
import os
import sys
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from PIL import Image

def make_interpolated_image(nsamples):
    ix = np.random.randint(im.shape[1], size=nsamples)
    iy = np.random.randint(im.shape[0], size=nsamples)
    samples = im[iy,ix]
    int_im = griddata((iy, ix), samples, (Y, X))
    return int_im

image_1 = cv2.imread('nature.jpg')[...,::-1]
im = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)

nx, ny = im.shape[1], im.shape[0]
X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))


nrows, ncols = 2, 2
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6,4), dpi=100)
if nx < ny:
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.set_figwidth(h), fig.set_figheight(w)

get_indices = lambda i: (i // nrows, i % ncols)


for i in range(4):
    nsamples = 10**(i+2)
    axes = ax[get_indices(i)]
    axes.imshow(make_interpolated_image(nsamples),
                          cmap=plt.get_cmap('Greys_r'))
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title('nsamples = {0:d}'.format(nsamples))
filestem = os.path.splitext(os.path.basename(img_name))[0]
plt.savefig('{0:s}_interp.png'.format(filestem), dpi=100)


import random
def distort_image(image):
    distorted_image = image[:,:,:]
    def add_spots_line():
        dim = min(image.shape[0], image.shape[1])
        num_distortions = random.randint(dim//6, dim//5)
        x = np.random.randint(image.shape[1] - 1, size=num_distortions)
        y = np.random.randint(image.shape[0] - 1, size=num_distortions)
        for j in x:
            for i in y:
                for k in range(distorted_image.shape[2]):
                    distorted_image[i][j][k] = random.random() * 255
    def add_blank_line():
        dim = min(image.shape[0], image.shape[1])
        num_distortions = random.randint(dim//6, dim//5)
        x = np.random.randint(image.shape[1] - 1, size=num_distortions)
        y = np.random.randint(image.shape[0] - 1, size=num_distortions)
        for j in x:
            for i in y:
                for k in range(distorted_image.shape[2]):
                    distorted_image[i][j][k] = 255
    def add_blank():
        dim = image.shape[0] * image.shape[1]
        num_distortions = random.randint(dim//50, dim//40)
        for i in range(num_distortions):
            x = random.randint(0, image.shape[1] - 1)
            y = random.randint(0, image.shape[0] - 1)
            for j in range(image.shape[2]):
                distorted_image[y][x][j] = 0
    choice = random.random() * 3
    choices = {0:add_spots_line, 1: add_blank_line, 2: add_blank}
    choices[int(choice)]()
    return distorted_image

image_1 = cv2.imread('nature.jpg')[...,::-1]
image_1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2RGB)
a = distort_image(image_1)
plot_figure(a, "distorted")