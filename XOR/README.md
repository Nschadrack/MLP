# Readme for the implementation of MLP to approximate the XOR operation

In the roadmap of building and training neural network, the first step is to define the problem we are trying to solve. For this case, the problem we are trying to solve is to build neural network model which can perform XOR operation on two inputs.

The second step is to collect and preprocess data. We will generate random samples with corresponding labels using built-in functions in python. Here we will define the dataset class for structing or holding our data. This is found in the module called `data.py`. 

Because we generating random samples to use, the dataset class will also split our data into train and validation dataset which is the third step in our roadmap.