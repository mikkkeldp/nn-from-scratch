# Building a neural network from scratch

In this project, I'll be building a simple, single layered neural network that classifies handwritten digits. We will ve making use of the MNIST dataset. The images have a size of 28x28 (single channel). The only package I'll be making use of is numpy.



## Preprocessing
The data preprocessing includes:
- flatten and normalizing the input images (divide by 255) 
- shuffle the training set to prevent biases and help training convergence
- one-hot encode the target labels

## Background
We will be using a sigmoid activation given by:
<div style='text-align: center;'>
<img src="https://latex.codecogs.com/png.latex?\sigma(z)=\frac{1}{1+e^{-z}}" />
</div>

For our cost function, we will be using cross-entropy. For a single example the cost will be:
<div style='text-align: center;'>
<img src="https://latex.codecogs.com/png.latex?L(y,\hat{y})=-\sum^n_{i=0}y_i\log(\hat{y}_i)" />
</div>
And for a set of m examples:
<div style='text-align: center;'>
<img src="https://latex.codecogs.com/png.latex?L(Y,\hat{Y})=-1/m\sum^m_{i=1}\sum^n{i=1}y_i^{(i)}\log(\hat{y}_i^{(i)})"/>
</div>


## Forward propagation
For the layers (exluding the final layer), the forward propagation is given by
<div style='text-align: center;'>
 <img src="https://latex.codecogs.com/png.latex?\hat{y}=\sigma(w^Tx+b)" />
</div>
By stacking examples, we vectorize the input and get a forward propagation equation of
<div style='text-align: center;'>
 <img src="https://latex.codecogs.com/png.latex?\hat{y}=\sigma(w^TX+b)" />
</div>
For our final layer (softmax layer), the final activations are the exponentials of its z-values
<div style='text-align: center;'>
<img src="https://latex.codecogs.com/png.latex?\hat{y}=\frac{e^{z_i}}{\sum^9_{i=0}e^{z_j}}" />
</div>

## Backwards propagation

The back propagation is given by:
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial{L}}{\partial{w_{j}}}=(\hat{y}-y)w_j" > 
</div>
For the vectorized form with m training examples:
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial{L}}{\partial{w_j}}=\frac{1}{m}X(\hat{y}-y)^T" > 
</div>
Similarly we can calculate the bias term
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial{L}}{\partial{b}}=(\hat{y}-y)" > 
</div>
and in vectorized form
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial{L}}{\partial{b}}=\frac{1}{m}\sum_{i=1}^m(\hat{y}^i-y^i)" > 
</div>

For more a more in-depth explanation see [these slides](https://nthu-datalab.github.io/ml/slides/10_NN_Design.pdf).


## Training
For training, we implement mini-batch gradient descent, with momentum (beta value of 0.9) and a batch size of 128. We also initialize the weights to 1/n, were n is the number of inputs feeding into that layer. We train for 9 epochs.

## Evaluation
After training is done, for each instance, we take the argmax of the final layer and compare to the labelled data and achieve an accuracy of 97.53%.












<!-- <img src="https://latex.codecogs.com/png.latex?s=\text { sensor reading }  " /> 

We first construct a 0 classifier. Set all y labels to 1 if zero image and 0 if not. We will be creating a logistic regression model with a single output node. The network will have 784 inputs (28x28).

## Forward propogation
The forward pass on a single example x executes the following computation
<div style='text-align: center;'>
 <img src="https://latex.codecogs.com/png.latex?\hat{y} = \sigma(w^Tx+b)" />
</div>
Here sigma is the sigmoid function:
<div style='text-align: center;'>
<img src="https://latex.codecogs.com/png.latex?\sigma(z) = \frac{1}{1 + e^{-z}}" />
</div>
here y_hat is a vector, and not a scalar as in the previous equation.  we vectorize by stacking examples side-by-side, so that our input matrix X has an example in each column.

## Cost function
We'll be using cross-entropy for our cost function. The formula for a single training example is:  
<div style='text-align: center;'>
<img src="https://latex.codecogs.com/png.latex?L(y,\hat{y}) = -y\log{(\hat{y}}) - (1-y)\log{(1-\hat{y}})" />
</div>
Averaging over a training set of m examples:  
<div style='text-align: center;'>
<img src="https://latex.codecogs.com/png.latex?L(Y,\hat{Y}) = -1/m \sum^m_{i=1} \(  -y^i \log{(\hat{y}^i}) - (1-y^i)\log{(1-\hat{y}^i}) \)"/>
</div>

## Back propagation

For backpropagation, we'll calculate the degree at which the loss changes with respect to each weight w_j. This is computed for each:
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial(L)}{\partial{w_j}}" >
</div>
Focusing on a single example makes it easier to derive the formulas required. Holding all values except for w_j, we can think of L being computed in 3 steps.
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?z = w^Tx+b">  
</div>
<br>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\hat{y} = \sigma(z)" >
</div>
<br>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?L(y,\hat{y}) = -y \log (\hat{y}) - (1-y) \log (1-\hat{y})">  
</div>
Following the chain rule, we get
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial{L}}{\partial{w_j}} = \frac{\partial{L}}{\partial{\hat{y}}} \frac{\partial{\hat{y}}}{\partial{z}} \frac{\partial{z}}{\partial{w_j}}">  
</div>
Looking at the first term on the RHS:<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial{L}}{\partial{\hat{y}}} = \frac{\partial}{\partial \hat{y}} (-y \log (\hat{y}) - (1-y) \log (1-\hat{y}))">  
</div>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?=\frac{-y}{\hat{y}} + \frac{1-y}{1-\hat{y}}">  
</div>
<br>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?=\frac{\hat{y} - y}{\hat{y}(1-\hat{y})}">  
</div>
Looking at the 2nd term:
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial \hat{y}}{\partial z} = \frac{\partial}{\partial z} \sigma (z) = \frac{\partial }{\partial (z)} \left( \frac{1}{1+e^{-z}} \right)"> 
</div> 
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?=- \frac{1}{({1+e^{-z}})^2}\frac{\partial}{\partial z} (1+e^{-z})">    
</div>
<br>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?= \frac{e^{-z}}{({1+e^{-z}})^2}" >  
</div>
<br>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?= \sigma (z){\frac{e^{-z}}{{1+e^{-z}}}" >  
</div>
<br>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?= \sigma (z)  \left( 1 - \frac{1}{1+e^{-z}} \right)" >  
</div>
<br>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?= \sigma (z)  ( 1 - \sigma(z))" > 
</div>
<br>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?= \hat{y} (1-\hat{y})" >   
</div>
Looking at the 3rd term;
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?= \frac{\partial z}{\partial w_j} = \frac{\partial}{\partial w_j} (w^Tx+b) = \frac{\partial}{\partial w_j} (w_0x_0 + w_1x_1 + \ldots + w_nx_n +b)" >  
</div>
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\therefore \frac{\partial z}{\partial w_j} = w_j" >   
</div>
We can now substitute the terms in, and get
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial L}{\partial w_j} = (\hat{y} - y)w_j" > 
</div>
For the vectorized form with m training examples:
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial L}{\partial w_j} = \frac{1}{m} X(\hat{y} - y)^T" > 
</div>
Similarly we can calculate the bias term
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial L}{\partial b} = (\hat{y} - y)" > 
</div>
and in vectorized form
<div style='text-align: center;'>
<img  src="https://latex.codecogs.com/png.latex?\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^m(\hat{y}^i - y^i)" > 
</div> -->


