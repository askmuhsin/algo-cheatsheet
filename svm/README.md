# Support vector machine (SVM)
The support-vector network is a new learning machine for two-group classification problems.
Vapnik, Cotes (1995) --> support-vector networks [paper](http://image.diku.dk/imagecanon/material/cortes_vapnik95.pdf)

![svm](https://github.com/askmuhsin/algo-cheatsheet/blob/master/svm/images/svm_basics2.png)

[reference](https://docs.opencv.org/trunk/d4/db1/tutorial_py_svm_basics.html)

### key terms   
* Support vectors  
To find the Decision Boundary, find the points which are close to the opposite group.   
Most of the time its going to be 2 from one group and 1 from the opposite group.   
In the image above from the red box group, the two are identified by red fill.   
From the blue circle group there is only support-vector.  

* Support planes   
Connect the group with two support-vectors to form a line, and draw a parallel line    
connecting the support-vector from blue group.    

* Optimal hyperplane   
This is what we want to figure out.   
its going to be represented as f(x) = wx+b.   
So a newcomer 'n' is in group 1 if f(n)>0 and group 0 if f(n)<0.

Optimal hyperplane will be the line mutually parallel to both support planes,    
while keeping maximum distance (or equal distance), to either of them.    
This makes boundary line far away from all points, preparing it to deal with data noise.   
