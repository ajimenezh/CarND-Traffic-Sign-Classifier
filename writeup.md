# **Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./trafficSignsExamples/freq.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/img1.png "Traffic Sign 1"
[image5]: ./examples/img2.png "Traffic Sign 2"
[image6]: ./examples/img3.png "Traffic Sign 3"
[image7]: ./examples/img4.png "Traffic Sign 4"
[image8]: ./examples/img5.png "Traffic Sign 5"
[image_24]: ./trafficSignsExamples/img24.png "Road narrows on the right"
[image_36]: ./trafficSignsExamples/img36.png "Go straight or right"
[image_0_color]: ./trafficSignsExamples/img_0_color.png "Color image"
[image_0_gray]: ./trafficSignsExamples/img_0_gray.png "Grayscale image"
[image_1]: ./trafficSignsExamples/img_1.png "Original image"
[image_1_rot]: ./trafficSignsExamples/img_1_rot.png "Rotated image"
[image_1_conv1]: ./trafficSignsExamples/feat_1.png "Features map 1 convolutional layer"
[image_1_conv2]: ./trafficSignsExamples/feat_1.png "Features map 2 convolutional layer"

---
### Writeup / README

### Data Set Summary & Exploration

### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

We can calculate some statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32, 32]
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I have implemented a function showSignOfType to plot an example of a sign of type i, for example, :

![alt text][image_24]
Road narrows on the right

![alt text][image_36]
Go straight or right

It is also important to see how data is distributed between the different classes.

![alt text][image1]

Here we can see that the data is not uniform, and there are some classes that are more represented than others, for example, there are much more "Keep right" signs than "Keep left".

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because, although the color may be significant to the meaning of the traffic sign, actually it doesn't help that much to differentiate between signs, and we obtain better results with grayscale.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image_0_color]

![alt text][image_0_gray]

As a last step, I normalized the image data because to make the mean zero and reduce variance, this way all the weights are in the same order.

I decided to generate additional data to increase the accuracy, adding new data generated with small modifications of the existing data (rotation), the model will predict better new images.

To add more data to the the data set, I applied rotations to the images. All the changes are small. These perturbations help predict new data. 

Here is an example of an original image and an augmented image:

![alt text][image_1]

![alt text][image_1_rot]

We create 5 new images from each images in the data set. There are other possible transformations that can be done, as shifts, adding noise, deleting data from the images, ..., but just with the rotations I've obtained good enough results (around 96% accuracy), however, using more tranformations, we probably could improve the results further.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride , valid padding, outputs 10x10x16  |
| RELU					| 		       									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten		      	| outputs 400 									|
| Fully Connected		| output 120									|
| RELU					| 		       									|
| Fully Connected		| output 84										|
| RELU					| 		       									|
| Fully Connected		| output 43										|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam's algorithm as the optimizer, a rate of 0.0005, batch size of 128, and I trained the model for 210 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.965 
* test set accuracy of 0.947

I've used the LeNet arquitecture, mainly because it has been used in a lot of similar applications, obtaining extremely good results, such as the MNIST problem. Also, I thought I could get better results working with the data input, and selecting the right parameters, than changing the arquitecture, based on the little experience I have with deep learning.

From here, I used an iterative approach to obtain results of more than 0.93 accuracy. First, I started without normalization, transformations, ..., just the original data. With this, I was getting just 0.08 of accuracy, but after normilizing the images and using grayscale images, the accuracy went to 0.78 with a batch size of 512, a rate of 0.1 and 10 epochs. After this, I changed the parameters a little to obtain a validation accuracy of 0.92. This result was obtained with the final parameters, a rate of 0.0005, a batch size of 128 and around 200 epochs. 
Finally, I augmented the data with rotations and Gauss noise. At first the results were worse because the rotations were big, and its better to only apply perturbations to the images. In the end I only applied rotations between -15 and 15 degrees, and the validation accuraty was around 0.96.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        	|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Road narrows on the right | Children crossing   							| 
| Slippery road    			| Slippery road 								|
| Road work					| Road work										|
| Priority road	      		| Priority road						 			|
| Bicycles crossing			| Right-of-way at the next intersection      	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The results does not seems very good, although because there are only 5 images, we can't draw any definitive conclusion. Also, it may be that the chosen images are very different from the ones trained.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is certain in all the images with a probability of 1.0. With these results and the test accuracy, it seems that the model may have overfit a little.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Children crossing    							| 
| 1.00     				| Slippery road  								|
| 1.00					| Road work										|
| 1.00	      			| Priority road						 			|
| 1.00				    | Right-of-way at the next intersection      	|
 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

We can see that in the first convolutional layer, the features map cover most of the image, while in the second convolutional layer there are more feature maps, and are focused in smaller parts of the images.

![alt text][image_1]

![alt text][image_1_conv1]

![alt text][image_1_conv2]



