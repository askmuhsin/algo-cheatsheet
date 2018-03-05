import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./images/digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)

# explore data
m, n, k = 1, 10, 1
plt.figure(figsize=(8,1))
for i in range(0,46,5):
    plt.subplot(m, n, k)
    plt.imshow(x[i, 0], cmap='gray')
    k+=1
plt.show()
