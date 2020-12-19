import numpy as np
import matplotlib.pyplot as plt
import model
from model import GAN_img
from PIL import Image


G = model.generator()
G.load_weights("weights/generator.h5")

img = np.array(Image.open("demo_lr.png"))
demo_sr = GAN_img(G, img)

plt.imshow(demo_sr)
plt.show()

