# Perceptron

We know to build models using Tesnsorflow, pytorch, scikitlearn, baselinetooles, etc.. Now let us build our own perceptron using numpy from scratch and train a small model.


Lets import the required libraries here.




```python
import numpy as np
import matplotlib.pyplot as plt
```

The Equation for the perceptron is $\hat{Y}\$ = W<sub>1</sub> * X<sub>1</sub> + W<sub>2</sub> * X<sub>2</sub> + B .


Here, 



  >W<sub>1</sub> , W<sub>2</sub> are Randomly assigned weights


  >X<sub>1</sub> , X<sub>2</sub> are inputs

  
  >B is Bias

  
  >$\hat{Y}\$ is the prediction.


  So, Let us assign some random values for weights and biases

  ```python
np.random.seed(123)
random_wieghts = np.random.randn(3)
w1 ,w2, bias = random_wieghts
```

Now Its time for the perceptron. Let us create a function forward_pass to provide all the vlues to the equation and return the prediction and sum of squared error.

```python
def forward_pass(w1,w2,bias,Y):
    error = 0
    y_pred = []
    for i in range(len(X1)):
        y = w1 * X1[i] + w2 * X2[i] + bias
        y_pred.append(y)
        error += (y - Y[i])**2
        
    return y_pred , error
```
Here we are iterating by the length of input and calculating prediction for each input of X<sub>1</sub>
>Y is the Target variable

To optimize the errors we have to update the weights in the equatoin.


Equation for weight update is

> W<sub>n</sub> = W<sub>n</sub> - &alpha; * partialderivative (J , W<sub>n</sub>) 


Here,



 >J = sum ( (Y - $\hat{Y}\$)<sup>2</sup> )



>partial_derivative ( J, W<sub>b</sub> ) = 2 (Y - $\hat{Y}\$)<sup>2</sup> * X<sub>n</sub>


> Bias =  2 (Y - $\hat{Y}\$)<sup>2</sup>



Here &alpha; is the learning rate of the model.



so let us create a update function to update weights.

```python
def weight_update(w1,w2,Y,bias,y_pred, alpha):
    for i in range(len(X1)):
        dJ_dw1 = ((y_pred[i] - Y[i])**2) * (X1[i])
        dJ_dw2 = ((y_pred[i] - Y[i])**2) * (X2[i])
        dJ_dbias = ((y_pred[i] - Y[i])**2)
        w1 = w1 - alpha * dJ_dw1
        w2 = w2 - alpha * dJ_dw2
        bias = bias - alpha * dJ_dbias
    return w1 , w2 , bias
```

Now its time for the training function. Here it is.....

```python
def train_model(X1,X2,Y,epochs,learning_rate):
    np.random.seed(123)
    random_wieghts = np.random.randn(3)
    w1 ,w2, bias = random_wieghts
    for i in range(epochs):
        y_pred , sse = forward_pass(w1,w2,bias,Y)
        print(f"epoch : {i} || Error : {sse}")
        w1 , w2, bias = weight_update(w1,w2,Y,bias,y_pred,learning_rate)
        
    return y_pred
```
This function returns the predictions.
