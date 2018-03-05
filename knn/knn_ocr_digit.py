"""https://docs.opencv.org/trunk/d8/d4b/tutorial_py_knn_opencv.html"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

## data preprocess
img = cv.imread('./images/digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)

# # explore data
# m, n, k = 1, 10, 1
# plt.figure(figsize=(8,1))
# for i in range(0,46,5):
#     plt.subplot(m, n, k)
#     plt.imshow(x[i, 0], cmap='gray')
#     k+=1
# plt.show()

## split train test
train = x[:,:50].reshape(-1, 400).astype(np.float32)
test = x[:,50:].reshape(-1, 400).astype(np.float32)

# print(x.shape, train.shape, test.shape, sep='\n')
k = np.arange(10)
train_labels = np.repeat(k, 250)[:,np.newaxis]
test_labels = train_labels.copy()

## initiate knn, change k to see effect on accuracy
k = 1
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=k)

## Accuracy evaluation
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size

print("\nAccuracy:\t", accuracy, "%\t", k, "- nn")
