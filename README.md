# Naive Bayes Classifier

**This project was inspired by and utilizes much of the info described in https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
However the code differs as this aims to implement naive bayes using numpy.**

This classifier utilizes the normal distribution to calculate "probabilities" needed for bayes theorem.
It has been tested using the iris data set, both binary and multi-class.


## Understanding how probabilities are calculated
Probabilities are calculated using bayes theorem P(class|data) = (P(data|class) * P(class)) / P(data). In our case we don't need the exact probability we just want a value that is still maximized meaning we still take the highest value to be the true class. This allows us to avoid integrating between two points on the normal distribution, and dividing by the P(data) which stops needless computations. In this naive bayes implementation the probability is calculated through the formula: 
- P(class=0|X1,X2) = P(X1|class=0) * P(X2|class=0) * P(class=0) 
where each input variable's probability of being in that class is being multiplied together which treats it independently hence naive.