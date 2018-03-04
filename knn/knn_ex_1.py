"""kNN is one of the simplest of classification algorithms available for
supervised learning. The idea is to search for closest match of the test data
in feature space
https://docs.opencv.org/trunk/d5/d26/tutorial_py_knn_understanding."""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

trainData = np.random.randint(0, 100, (25,2)).astype(np.float32)
labels = np.random.randint(0, 2, (25,1)).astype(np.float32)

red = trainData[labels.ravel()==0]
blue = trainData[labels.ravel()==1]

plt.scatter(red[:,0], red[:,1], 50, 'r', '^')
plt.scatter(blue[:,0], blue[:,1], 50, 'b', 's')

newcomer = np.random.randint(0, 100, (1,2)).astype(np.float32)
plt.scatter(newcomer[:,0], newcomer[:,1], 50, 'g', 'o')

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, labels)

ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

print( "result:  {}\n".format(results), "blue" if results else "red", sep='\t')
print( "neighbours:  {}\n".format(neighbours) )
print( "distance:  {}\n".format(dist) )

plt.show()
