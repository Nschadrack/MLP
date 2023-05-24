# Introduction to deep learning

## General introduction to Artificial Intelligence(AI)

**Artificial Intelligence(AI):** is a branch or subfield of computer science deals with the development and implementation of computer systems and algorithms that can perform tasks that typically require human intelligence. AI aims to create intelligent machines that can perceive, reason, learn, and make decisions similar to or even surpassing human capabilities in certain domains.

AI encompasses a wide range of techniques, approaches, and subfields. Here are some key aspects of AI:

1. **Machine Learning:** Machine learning is a subset of AI that focuses on enabling computers to learn and improve from experience without being explicitly programmed. It involves developing algorithms and models that can automatically analyze and interpret data, extract patterns, and make predictions or decisions.

2. **Deep Learning:** Deep learning is a subfield of machine learning that focuses on training artificial neural networks with multiple layers (deep neural networks) to learn hierarchical representations of data. Deep learning has achieved remarkable success in various domains, such as image and speech recognition, natural language processing, and autonomous driving.

3. **Natural Language Processing (NLP):** NLP involves enabling computers to understand, interpret, and generate human language. It includes tasks like language translation, sentiment analysis, question answering, and chatbots.

4. **Computer Vision:** Computer vision aims to enable computers to understand and interpret visual information from images or videos. It involves tasks such as image classification, object detection, image segmentation, and facial recognition.

5. **Robotics:** Robotics combines AI and physical systems to create intelligent machines that can interact with the physical world. It involves developing robots capable of perception, decision-making, manipulation, and autonomous navigation.

6. **Expert Systems:** Expert systems are AI systems designed to replicate the decision-making abilities of human experts in specific domains. They use knowledge representation and inference mechanisms to provide expert-level advice or solutions to complex problems.

7. **Reinforcement Learning:** Reinforcement learning involves training agents to make sequential decisions in an environment to maximize a reward signal. It is often used in tasks where an agent interacts with an environment, such as game playing, robotics, and autonomous driving.

8. **Artificial General Intelligence (AGI):** AGI refers to highly autonomous systems that possess general intelligence and can understand, learn, and perform any intellectual task that a human being can do. AGI aims to replicate human-level intelligence across a broad range of domains.

AI has a wide range of applications across various industries, including healthcare, finance, transportation, entertainment, and more. It continues to advance rapidly, driven by research breakthroughs, increased computational power, and the availability of large datasets.

**Our main focus is on deep learning which is the base of currently AI like NLP, Computer vision and Robotics**

**Deep learning** starts with an artificial neuron or perceptron which was motivated by the human brain(biological neuron). The percetron receives several or many inputs and applies matrix multiplication with the random weights to those inputs and makes addition of the result with the bias. The result is passed through an activation function which introduces non-linearity in the perceptron. Action function helps perceptron to learn any relationship of any function by providing required parameters(weights). Without activation function, the perceptron behaves as linear model.

### Mulit-Layer Perceptron(MLP) as a universal function approximator

A Multi-Layer Perceptron (MLP) is a type of artificial neural network that can serve as a universal function approximator. The universal approximation theorem states that an MLP with a single hidden layer containing a sufficient number of neurons can approximate any continuous function to arbitrary accuracy within a given range.

1. **MLP as universal boolean gate**: we will implement MLP to approximate XOR
2. **MLP as universal classifier**: We will implement MLP on classification problem with different regularization and optimization technique
3. **MLP as universal regression**: we will implement MLP on regression problem

## The roadmap for building and training neural network

Building and training a neural network involves several key steps. Here is a roadmap that outlines the general process:

1. **Define the Problem:** Clearly define the problem you want to solve using a neural network. Determine whether it is a classification, regression, or another type of task. Identify the input features, output targets, and any specific requirements or constraints.

2. **Data Collection and Preparation(data preprocessing):** Gather a suitable dataset for training and evaluation. Ensure that the data is representative, diverse, and labeled correctly. Preprocess the data by performing tasks such as data cleaning, normalization, feature scaling, and handling missing values.

3. **Split the Data:** Split the dataset into training, validation, and testing sets. The training set is used to train the model, the validation set helps with hyperparameter tuning and model selection, and the testing set evaluates the final model's performance on unseen data.

4. **Choose a Neural Network Architecture:** Select an appropriate neural network architecture based on the problem type, complexity, and available resources. Decide on the number of layers, the number of neurons in each layer, and the activation functions to be used. Common architectures include feedforward neural networks, convolutional neural networks (CNNs) for image data, and recurrent neural networks (RNNs) for sequential data.

5. **Initialize the Model:** Initialize the model parameters (weights and biases) using appropriate initialization techniques such as random initialization, Xavier initialization, or He initialization. Proper initialization can help the model converge faster and avoid getting stuck in poor local optima.

6. **Define the Loss Function:** Choose an appropriate loss function that aligns with your problem type. For example, binary cross-entropy loss for binary classification, categorical cross-entropy loss for multi-class classification, or mean squared error loss for regression. The loss function quantifies the difference between predicted and true values.

7. **Choose an Optimization Algorithm:** Select an optimization algorithm to update the model parameters during training. Popular algorithms include stochastic gradient descent (SGD), Adam, RMSprop, and others. Each algorithm has its own hyperparameters that need to be tuned for optimal performance.

8. **Train the Model:** Iterate through the training data, feeding it into the model, and adjusting the parameters using the chosen optimization algorithm. This process is typically performed in mini-batches, where a subset of the training data is used at each iteration. Monitor the training progress by evaluating the loss and other metrics on the validation set.

9. **Hyperparameter Tuning:** Experiment with different hyperparameter settings, such as learning rate, batch size, regularization techniques, and network architecture. Use the validation set to assess the performance of different configurations and select the best combination.

10. **Evaluate the Model:** Once training is complete, evaluate the final model on the testing set to assess its performance on unseen data. Calculate relevant metrics such as accuracy, precision, recall, or mean squared error, depending on the problem type.

11. **Iterate and Improve:** Analyze the model's performance and identify areas for improvement. Adjust the architecture, hyperparameters, or data preprocessing techniques as needed. Iterate on the process to refine and enhance the model's performance.

We will go through each step while implementing MLP on some problems such as XOR, regression(MLP to approximate natural logarithm), and classification(both binary classification and categorical or multi-classication problems).


The main directories for each problem domain are:
 - **XOR:** contains the implementation of MLP to classify XOR result given two inputs
 - **Regression:** contains the implementation of MLP for regression problem
 - **Binary:** contains the implementation of MLP for binary classification
 - **Categoriacl:** contains the implementation of MLP for multi-class classification problem


 **N.B:** we will pyorch as our deep learning framework. Pytorch is the most widely framework in deep learning particularly in the research domain of AI. There are other frameworks like tensorflow and keras.



