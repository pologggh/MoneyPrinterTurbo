# Gradient Descent Optimization

Gradient descent is a first-order iterative optimization algorithm used to minimize differentiable objective loss functions in machine learning models.
At each training step, the algorithm computes the gradient of the loss function with respect to the model parameters via backpropagation.
The parameters are then updated in the direction opposite to the gradient vector, scaled by a learning rate hyperparameter.
Variants such as Stochastic Gradient Descent (SGD) and Adam introduce batch sampling and momentum to accelerate convergence and navigate saddle points.
