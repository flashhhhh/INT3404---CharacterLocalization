import cv2
import matplotlib.pyplot as plt

name = "nlvnpf-0137-01-001"

image = cv2.imread("images/train/" + name + ".jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

width = image.shape[1]
height = image.shape[0]

with open("labels/train/" + name + ".txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        [a, x, y, w, h] = line.split()
        w = float(w) * width
        h = float(h) * height
        x = int(float(x) * width - w/2)
        y = int(float(y) * height - h/2)
        z = int(x + w)
        t = int(y + h)
        
        # print(x, y, w, h)
        image = cv2.rectangle(image, (x, y), (z, t), (255, 0, 0), 1)

plt.imshow(image)
plt.show()