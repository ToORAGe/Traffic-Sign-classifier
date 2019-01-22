# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train_hist.png "train"
[image2]: ./examples/valid_hist.png "valid"
[image3]: ./examples/test_hist.png "test"
[image4]: ./examples/gray_original.png "gray"
[image5]: ./examples/several_signs.jpg "examples"
[image6]: ./examples/my_signs.png "final"
[image7]: ./examples/1x.png "Traffic Sign 2"
[image8]: ./examples/2x.png "Traffic Sign 3"
[image9]: ./examples/3x.png "Traffic Sign 4"
[image10]: ./examples/5x.png "Traffic Sign 5"
[image11]: ./examples/6x.png "Traffic Sign 6"
[image12]: ./examples/8x.png "Traffic Sign 7"
[image13]: ./examples/9x.png "Traffic Sign 8"
[image14]: ./examples/10x.png "Traffic Sign 9"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The following writeup provides details of the Traffic Sign Classifier project code. 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training, validation, and test data sets. It is a bar chart showing how the data is distributed among all the classes. 

Training data 
![alt text][image1]

Validation data 
![alt text][image2]

Testing data 
![alt text][image3]

 Design and Test a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale so as to reduce the dimension of the input images. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

As a last step, I normalized the image data so as to achieve faster convergence of the gradient descent method. 

I decided to use the original data as it is without any modifications. 

2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5x1x6  	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5x6x16  | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten    	      	| outputs 400      								|
| Fully connected		| 400 inputs, 200 outputs						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| 200 inputs, 100 outputs						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| 100 inputs, 43 outputs						|
|						|												|
|						|												|
 


3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with a batch size of 128 and 50 epochs. The learning rate has been set to 0.0009.
The drop out parameter keep_prob has been set to 0.75 for training the data. 

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.964
* test set accuracy of 0.942

If a well known architecture was chosen:
* The well known LeNet architecture was chosen for this project. 
* I believe that the LeNet model works for this project as this project involves classifies traffic signs and Convolutional Neural Networks like LeNet work best in this regard. 
* The final models accuracy shows a validation data accuracy of 0.955 and a test data accuracy of 0.938(above the prescribed threshold for acceptance). These accuracies show that the model classifies the images with a high accuracy (above 90%).
 

 Test a Model on New Images

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14] 


2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

The model was able predict the signs with an accuracy of 100%. This compares favorably to the accuracy on the test set of 100%. Though a 100% accuracy cannot be expected in all situations, this is still a reliable prediction. 


3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For almost all the images, the model predicts the traffic sign with a high probability. For the last image, the following were the softmax predictions. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .7         			| Road work   									| 
| .16     				| Keep left 									|
| .08  				| Double curve										|

Following are all the softmax predictions for the traffic signs I found on the internet. 

![alt text][image6] 



