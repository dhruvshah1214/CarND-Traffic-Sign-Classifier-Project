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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./german1.jpg "Traffic Sign 1"
[image5]: ./german2.jpg "Traffic Sign 2"
[image6]: ./german3.jpg "Traffic Sign 3"
[image7]: ./german4.jpg "Traffic Sign 4"
[image8]: ./german5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410 (shown at preprocessing)
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Exploratory visualizations are included in the iPython Notebook. A bar chart showing training example distribution over the 43 classes showed that it was evident some data augmentation would be required to take out bias. (Surprisingly, my results were only about 1% higher after adding augmentation. While this is a good amount, I had expected the increase to be much more.)

Another one shows the "mean image" and the deviation image. These two images show towards what type of image the training data is biased. Descriptions and explanations are included in the notebook.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color data wouldn't be very useful and would only add a layer of complexity.

As a last step, I normalized the image data because having zero mean and low variance in a data set is important for quick convergance.

Before preprocessing, I decided to generate additional data because as I explained before, the data was biased. 

To add more data to the the data set, I used the following common techniques:
1) Scaling
2) Rotation
3) Brightness Adjusting
4) Transforming

The difference between the original data set and the augmented data set can be seen in the notebook; the augmented set has just under 3x the number of data samples. To approximately control how many images each augmentation would generate, I used random number range checking to keep the multiplicity within a reasonable range.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		      |     Description	        					                            | 
|:---------------------:|:--------------------------------------------------------:| 
| Input         		      | 32x32x1 image -> grayscaled & 0-mean normalized array   	| 

| Convolution 5x5     	 | 1x1 stride, VALID padding, outputs 28x28x6 	             |
| RELU					             |	outputs 28x28x6    								                              |
| Max pooling	      	   | 2x2 stride,  outputs 14x14x6 				                        |

LAYER 2
| Convolution 5x5     	 | 1x1 stride, VALID padding, outputs 10x10x6 	             |
| RELU					             | outputs 10x10x6                              												|
| Dropout					          | Keep probability: 0.5, outputs 10x10x6       												|
| Max pooling	      	   | 2x2 stride,  outputs 5x5x6   				                        |

LAYER 3
| Convolution 5x5     	 | 1x1 stride, VALID padding, outputs 1x1x400 	             |
| RELU					             | outputs 1x1x400                              												|
| Dropout					          | Keep probability: 0.5, outputs 1x1x400       												|

| FLATTEN LAYERS 2 + 3  | Layer 2: 400, Layer 3: 400, outputs 400 + 400 = 800      |
| Dropout					          | Keep probability: 0.5, outputs 1x1x400       												|

| Fully connected		     | 800 -> 43        									                               |

| Softmax				           | softmax cross entropy                                    |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer. Hyperparameter tuning explained below.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of ~97%
* test set accuracy of ~94%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
- The original LeNet, because it was simple and quick.

* What were some problems with the initial architecture?
- Not high enough accuracy.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
- I changed the architecture to the one described in the research paper provided in the notebook.
- Overfitting was a huge problem. At one point, I was at a dropout probability of 0.1 before I came to my senses and said to myself, "Okay, this is crazy. I need to solve this another way."
- And so I included L2 Regularization with the loss function, which helped a lot with overfitting.

* Which parameters were tuned? How were they adjusted and why?
- The epoch count, batch size, and learning rate were adjusted based on iteration and experimentation. If a pass through was taking way too long, I increased the learning rate. But then it could stop converging with a high learning rate because it'd just skip over the minima, so sometimes I had to tune it down. The epoch count was kept low for quick debugging and testing iterations, but at the end I turned it up way high. The batch size was tuned based on knowledge of reading other sources and iteration similar to the learning rate.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
- Dropout layers and regularization are important to take out the specifics that a model generally tends to learn. It's important to remember that the purpose of a model is often to classify inputs it hasn't seen, so just slamming it with a load of unimportant features and turning up all the parameters doesn't really work. The model needs to generalize and interpolate well, not try to fit every single datapoint, because if it does, it'll catch even the slight outliers and try to adjust to them, inherently decreasing the accuracy of the model.

If a well known architecture was chosen:
* What architecture was chosen?
- The Sermanet/LeCunn modified LeNet architecture
* Why did you believe it would be relevant to the traffic sign application?
- Udacity included a research paper showing off its high accuracies in traffic sign classification contests!
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
- 94% is a pretty high accuracy rate for a test set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

All images are pretty difficult to classify because they're so pixellated (I just used a cv2 resize after downloading them). They're also pretty distorted and warped, which could actually throw the classifier off quite a bit. Furthermore, the 30km/hr speed signs have an extremely unclear first digit, and in fact, it was hard for even my human eye to detect the 3 in one of them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

In-depth information about the predictions can be found in the notebook.
The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This pales in comparison to the test set accuracy, most likely becuase the test set images are a well curated dataset with more common data and non-skewed images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Visualizations and bar charts for the probabilities of each image are shown in the notebook.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Features found include mainly contrasting edges and solid color patches. One interesting thing to note is that the model developed the digits of the number 20 by itself: the MNIST classifier had this specific purpose to classify digits, and this model didn't, but still picked up on the digits (most likely to distinguish the different speed signs).

