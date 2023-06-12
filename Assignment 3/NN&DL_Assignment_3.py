#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


cd /content/drive/MyDrive/Assignment_3


# In[ ]:


get_ipython().system('ls')


# # Que 1:
# Download the benchmark dataset, MNIST, from http://yann.lecun.com/exdb/mnist/. Implement multi-class classification for recognizing handwritten digits (also known as multiclass
# logistic regression ---this is simply a feedforward neural network with k output neurons, with one output neuron for each class, and each output neuron oi returns the probability that the input datapoint xj is in class i) and try it on MNIST.
# 
# Comments: No need to implement almost anything in DL by your own (this is true in general); the
# software framework (ie, the DL platform) typically provides implementations for all the things discussed in class, such as the learning algorithms, the regularizations methods, the cross-validation methods, etc.
# 
# Use your favorite deep learning platform. A few candidates:
# 1. Marvin from http://marvin.is/
# 2. Caffe from http://caffe.berkeleyvision.org)
# 3. TensorFlow from https://www.tensorflow.org
# 4. Pylearn2 from http://deeplearning.net/software/pylearn2/
# 5. Theano, Torch, Lasagne, etc. See more platforms at http://deeplearning.net/software_links/.
# 
# Read the tutorial about your selected platform (eg, for TensorFlow:
# https://www.tensorflow.org/tutorials), try it on MNIST; note that the first few examples in the tutorials are typically on MNIST or other simple image datasets, so you can follow the steps.
# 
# Comments: MNIST is a standard dataset for machine learning and also deep learning. It’s good to try it on one shallow neural network (with one output neuron; eg, for recognizing a character A from a not-A character) before trying it on a deep neural network with multiple outputs. Downloading the
# dataset from other places in preprocessed format is allowed, but practicing how to read the dataset prepares you for other new datasets you may be interested in (thus, please, read the MNIST website carefully). 
# 
# 

# ## Que 1: Part 1
# 
# ---
# 
# 1. Try the basic minibatch SGD as your learning algorithm. It is recommended to try different
# initializations, different batch sizes, and different learning rates, in order to get a sense about how to tune the hyperparameters (batch size, and, learning rate). Remember to create and use validation dataset!. it will be very useful for you to read Chapter-11 of the textbook.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import gzip
from google.colab import drive
import os

# Step 1: Download and read the MNIST dataset
def download_mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for file_name in file_names:
        urllib.request.urlretrieve(base_url + file_name, file_name)


# In[ ]:


def read_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784)

def read_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


# In[ ]:


# Create MNIST folder in Google Drive
get_ipython().system("mkdir -p '/content/drive/MyDrive/Assignment_3/MNIST_Dataset'")


# In[ ]:


# Set the Google Drive folder as the working directory
os.chdir('/content/drive/MyDrive/Assignment_3/MNIST_Dataset')


# In[ ]:


# Download MNIST dataset
download_mnist()

# Read training and testing images and labels
train_images = read_mnist_images('train-images-idx3-ubyte.gz')
train_labels = read_mnist_labels('train-labels-idx1-ubyte.gz')
test_images = read_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = read_mnist_labels('t10k-labels-idx1-ubyte.gz')


# In[ ]:


print("Size of Training Images Dataset : ", len(train_images))
print("Size of Training Images Dataset : ", len(test_labels))


# In[ ]:


train_images[1].shape


# In[ ]:


# Display some training images
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
axs = axs.flatten()

for i in range(10):
    img = train_images[i].reshape(28, 28)
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(f"Label: {train_labels[i]}")

plt.tight_layout()
plt.show()


# ### Data Preprocessing

# In[ ]:


# Step 2: Data preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Showing images after preprocessing
fig, axs = plt.subplots(1, 5, figsize=(10, 4))
axs = axs.flatten()

for i in range(5):
    img = train_images[i].reshape(28, 28)
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(f"Label: {train_labels[i]}")

plt.tight_layout()
plt.show()


# In[ ]:


# Step 2: Prepare the data
num_classes = 10
num_features = train_images.shape[1]
num_samples = train_images.shape[0]


# In[ ]:


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# In[23]:


# Preprocess the MNIST dataset
input_dim = train_images.shape[1]
x_train = train_images.reshape(-1, 784)
x_test = test_images.reshape(-1, 784)
y_train = tf.keras.utils.to_categorical(train_labels, num_classes=10)
y_test = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Create a validation dataset
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)


# In[ ]:


# Define a function to create your model with the desired architecture
def create_model(learning_rate, hidden_units):
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(num_classes, activation='sigmoid'))
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create the KerasClassifier wrapper
model = KerasClassifier(build_fn=create_model, epochs=11, batch_size=64, verbose=1)

# Define the hyperparameters grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_units': [32, 64, 128]
}

# Perform grid search cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(x_train, y_train)

print("\n\n")
# Print the best hyperparameters and accuracy
print("Best Hyperparameters: ", grid_result.best_params_)
print("Best Accuracy: ", grid_result.best_score_)


# In[ ]:


# Get the best model from the grid search
best_model = grid_result.best_estimator_

# Get the training history
history = best_model.fit(x_train, y_train, epochs=11, batch_size=64, verbose=0, validation_data=(x_val, y_val))

# Plot the learning curve
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Learning Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

print("\n\n")
# Print the best hyperparameters and accuracy
print("Best Hyperparameters: ", grid_result.best_params_)
print("Best Accuracy: ", grid_result.best_score_)


# In this experiment, our objective was to construct a simple neural network comprising only one hidden layer. To optimize its performance, we employed GridSearchCV to explore various values for the learning rate and the number of neurons in the hidden layer. In addition, we selected 'accuracy' as the evaluation metric and utilized cross-entropy categorical loss as our loss function, which is suitable for multiclass classification scenarios.
# 
# To facilitate learning, we employed mini-batch stochastic gradient descent (SGD) as our chosen learning algorithm, employing a batch size of 64. Furthermore, we integrated a validation set to conduct ongoing validation during the training process. In the results, we can see the best accuracy achieved is 97.1% with hidden units as 128 and learning rate of 0.1.

# ## Que 1: Part 2
# 
# ---
# 2. It is recommended to try, at least, another optimization method of your choice (SGD with
# momentum, RMSProp, RMSProp with momentum, AdaGrad, AdaDelta, or Adam) and compare its performances to those of the basic minibatch SGD on the MNIST dataset. Which methods you want to try and how many you want to try and compare is up to you and up to the amount of time you have left to complete the assignment. Remember, this is a research course. You may want to read Chapter-8 also.

# In[ ]:


import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam

learning_rates = [0.001, 0.01, 0.1]
optimizers = [
    ('SGD', SGD),
    ('RMSprop', RMSprop),
    ('Adagrad', Adagrad),
    ('Adadelta', Adadelta),
    ('Adam', Adam)
]
# Find the best learning rate and optimizer based on model accuracy
best_accuracy = 0.0
best_learning_rate = None
best_optimizer = None

# Define the model architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='sigmoid'))

# Train and plot accuracy for different learning rates and optimizers
fig, axs = plt.subplots(3, len(optimizers), figsize=(12, 12))

for i, lr in enumerate(learning_rates):
    for j, (optimizer_name, optimizer_class) in enumerate(optimizers):
        optimizer = optimizer_class(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=64, epochs=11, validation_split=0.2, verbose=0)
        accuracy = history.history['val_accuracy'][-1]

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_learning_rate = lr
            best_optimizer = optimizer_name

        # Plot accuracy
        axs[i, j].plot(history.history['val_accuracy'])

        # Set plot labels
        axs[i, j].set_title(f'Optimizer: {optimizer_name}\nLearning Rate: {lr}')
        axs[i, j].set_xlabel('Epochs')
        axs[i, j].set_ylabel('Validation Accuracy')

# Set layout and spacing between subplots
plt.tight_layout(pad=2.0)

# Print the best learning rate and optimizer
print(f'Best Learning Rate: {best_learning_rate}')
print(f'Best Optimizer: {best_optimizer}')
print(f'Best Accuracy: {best_accuracy}')

print("\n\n")
# Show the plots
plt.show()


# In our experiment, we investigated the performance of various optimization algorithms by testing them with three different learning rate values: 0.1, 0.01, and 0.001. To ensure a fair comparison, we kept the batch size, number of epochs, loss function, number of layers, and hidden units consistent across all methods.
# 
# By analyzing the plots of validation accuracy versus epochs for each optimization algorithm, we can draw some conclusions.
# 
# Firstly, both SGD and RMSProp consistently outperformed Adagrad and Adadelta. This observation suggests that SGD and RMSProp were more effective in finding the optimal parameters for our neural network. The reason behind their superior performance could be attributed to their ability to adapt the learning rate based on the gradients of the parameters.
# 
# On the other hand, the performance of Adam exhibited high variation in the plots. This variation could be attributed to a couple of factors. One possibility is that we trained the neural network for a relatively small number of epochs, which might not have been sufficient for Adam to converge to a stable solution. Additionally, Adam incorporates adaptive learning rates and momentum, which can introduce additional complexity in optimization, especially in scenarios with simple data.
# 
# Despite the variations observed with Adam, it is worth noting that all the optimization algorithms achieved an accuracy of over 90 percent. This outcome suggests that the dataset used in our experiment was relatively simple, allowing all the algorithms to achieve satisfactory results. It is important to consider that more complex datasets may require further fine-tuning of hyperparameters and choice of optimization algorithm to achieve optimal performance.
# 
# Overall, through these iterations we achieved 97.8% as a best accuracy and optimizer was "SGD" with learning rate 0.01.

# # Que 2:
# 
# Consider the L2-regularized multiclass logistic regression. That is, add to the logistic
# regression loss a regularization term that represents the L2-norm of the parameters. More precisely,the regularization term is
# 
#     (w, b) = λi(∥wi∥2 + ∥bi∥2)
# 
# where {wi, bi} are all the parameters in the logistic regression, and λ ∈ R is the regularization hyperparameter. Typically, λ is about C/n where n is the number of data points and C is some constant in [0.01,100] (need to tune C). Run the regularized multiclass logistic regression on MNIST, using the
# basic minibatch SGD, and compare its results to those of the basic minibatch SGD with nonregularized loss, in Question #1.

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import OneHotEncoder

# Define the logistic regression model architecture with L2 regularization
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,), kernel_regularizer='l2'))
model.add(Dense(10, activation='sigmoid'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])

# Train the model with L2 regularization
history = model.fit(x_train, y_train, batch_size=128, epochs=11, validation_split=0.2, verbose=1)

# Evaluate the model on the testing set
_, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n")
# Plot the training and validation accuracies
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('L2-Regularized Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

print("\n\n")
# Print the accuracy
print(f'Test Accuracy (L2-Regularized): {test_accuracy}')


# We applied L2 regularization to the same model that was used in the previous question. L2 regularization is a technique commonly employed to prevent overfitting in neural networks by adding a penalty term to the loss function based on the magnitudes of the model's weights.
# 
# After incorporating L2 regularization, we observed a slight decrease in the training accuracy compared to the previous result. The training accuracy now stands at 94.62% instead of the previous 97.1%. This drop in training accuracy can be attributed to the regularization term penalizing the weights and encouraging them to be smaller. While we achieved 94.88% as a test accuracy with L2-Regularization.
# 
# However, an interesting observation is that the validation accuracy is now closer to the training accuracy after the addition of L2 regularization. This outcome suggests that the regularization helped in reducing overfitting and improving the generalization capability of the model. When the validation accuracy is closer to the training accuracy, it indicates that the model's performance on unseen data is more aligned with its performance on the training data.
# 
# Overall, although the training accuracy decreased slightly due to the L2 regularization, the improved alignment between the training and validation accuracies implies that the model is now less likely to overfit and more capable of generalizing well to unseen data.

# # Que 3:
# 
# Build a three-layer feedforward neural network:
#  
#  x → h1 → h2 → o
# 
# The hidden layers h1 and h2 have width 500. Train the network for 250 epochs1 and test the classification error. Do not use regularizations. Plot the cross-entropy loss on the batches and also plot the classification error on the validation data.
# 
# Comments: 1Each epoch is a pass over the training data. Suppose you use batches of size b, and the training data set has n points, then an epoch consists of n/b batches. Note that you can divide the data set into batches, and then round robin over the batches. You can also randomly sample say 64 points for each batch. Either way is OK, and typically there is no performance difference between them. When these batches are randomly sampled, it is possible that some point are not in any of them, but we still call these batches a pass over the data.
# 
# Comments: you can also use another dataset, CIFAR-10 or CIFAR-100. Or you can pick your own
# dataset.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Define the three-layer feedforward neural network
model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(784,)))
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=64, epochs=250, validation_split=0.2, verbose=1)

# Evaluate the model on the testing set
_, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n")
# Plot the cross-entropy loss on the batches
plt.plot(history.history['loss'])
plt.title('Cross-Entropy Loss on Batches')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print("\n")

# Plot the classification error on the validation data
plt.plot(1 - np.array(history.history['val_accuracy']))
plt.title('Classification Error on Validation Data')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

print("\n\n")
# Print the test accuracy
print(f'Test Accuracy: {test_accuracy}')


# In this experiment, we constructed a three-layer neural network with a higher number of neurons in each layer compared to previous models. We also increased the training duration to 250 epochs.
# 
# After training the model, we obtained 100% accuracy while training and test accuracy of 97.99%, indicating that it was able to perfectly fit the training data. This exceptional performance suggests that the model had sufficient capacity to capture the complex patterns present in the training dataset. Also this 100% accuracy waqs acheivable due to simplicity of the data and actually it was achieved by 170 epochs only.
# 
# Furthermore, the model achieved an accuracy of 97.6% on the validation set. This result demonstrates that the model generalized well to unseen data, as it performed with high accuracy on samples it had not been exposed to during training. This validates the model's ability to capture the underlying patterns in the data and make accurate predictions.
# 
# For the purpose of comparison, we employed the SGD optimizer with a learning rate of 0.01 and a batch size of 64, consistent with the previous questions. This allows us to evaluate the impact of the increased model complexity and training duration while keeping other factors constant.

# # Que 4:
# 
# Repeat Question #3 but train the network with the following regularizations: 
# 
# L2-norm, dropout, and early-stopping. Compare with the results of Question #3.
# 
# Comments: No need to implement them by your own (this is true in general); the software framework (ie, the DL platform) typically provides implementations for all the regularizations methods discussed in class. Early stopping is done in training, so you only need to tune your training code slightly.

# In[ ]:


from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define the three-layer feedforward neural network with regularizations
model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(784,), kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='sigmoid'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model with regularizations and early stopping
history = model.fit(x_train, y_train, batch_size=64, epochs=250, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Evaluate the model on the testing set
_, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n")
# Plot the cross-entropy loss on the batches
plt.plot(history.history['loss'])
plt.title('Cross-Entropy Loss on Batches (with Regularizations)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print("\n")

# Plot the classification error on the validation data
plt.plot(1 - np.array(history.history['val_accuracy']))
plt.title('Classification Error on Validation Data (with Regularizations)')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

print("\n\n")
# Print the test accuracy
print(f'Test Accuracy: {test_accuracy}')


# In this experiment, we incorporated additional techniques to prevent overfitting and enhance the performance of the model. Firstly, we applied dropout layers with a dropout rate of 0.5 after both hidden layers. Dropout is a regularization technique that randomly sets a fraction of the input units to zero during training, which helps prevent overfitting by reducing interdependencies between neurons.
# 
# Additionally, we implemented early stopping based on validation loss values. This technique monitors the validation loss during training and stops the training process if the validation loss does not improve after a certain number of epochs (patience value of 10, in this case). Early stopping helps prevent overfitting by stopping the training process before the model starts to memorize the training data.
# 
# We also continued to apply L2 regularization to further control overfitting.
# 
# The results of these modifications are promising. The training accuracy achieved was 96.09%, indicating that the model performed well on the training set while preventing overfitting. The final test accuracy of 96.88% indicates that the model generalized well to unseen data and maintained a high level of accuracy.
# 
# By analyzing the plot, we observed a significant decrease in the classification error until around 50 epochs, suggesting that the model quickly learned relevant patterns and improved its performance. After this point, the learning process gradually converged, resulting in a slower decrease in classification error.
# 
# In summary, the incorporation of dropout layers, early stopping, and L2 regularization helped control overfitting and improve the model's performance. The achieved accuracy on both training and test sets indicates the model's ability to generalize well and make accurate predictions on unseen data. The plot demonstrates the learning progression, with a significant reduction in classification error followed by a gradual convergence.

# # Que 5:
# 
# Try CNN (convolutional neural networks) on MNIST (or CIFAR or any dataset of your
# choice). Use the basic minibatch SGD as your learning algorithm, with or without regularizations. You may need to read Chapter-7, Chapter-8 and Chapter-9.

# In[22]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Preprocess the dataset
x_train = train_images.reshape(-1, 28, 28, 1) 
x_test = test_images.reshape(-1, 28, 28, 1) 

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(train_labels, num_classes=10)
y_test = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, batch_size=64, epochs=11, validation_split=0.2, verbose=1)

# Evaluate the model on the testing set
_, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n")
# Plot the accuracy on training and validation data
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("\n\n")
# Print the test accuracy
print(f'Test Accuracy: {test_accuracy}')


# In this final experiment, we constructed a neural network model using Convolutional Neural Network (CNN) layers, a popular choice for image classification tasks. For comparison purposes, we maintained the same number of layers, 11 epochs, a batch size of 64, and a learning rate of 0.01.
# 
# The results of this experiment are highly promising. Within just 11 epochs, the CNN model achieved an accuracy of 97.6% on the test data and 98.03% on train dataset. This high accuracy demonstrates the effectiveness of the CNN architecture in capturing relevant image features and making accurate predictions.
# 
# Additionally, the validation accuracy closely aligns with the test accuracy, indicating a low chance of overfitting. When the validation accuracy closely matches the test accuracy, it suggests that the model generalizes well to unseen data and can make reliable predictions on new instances.
# 
# The utilization of CNN layers in image classification models allows for the extraction of spatial hierarchies and local patterns present in the images. This enables the model to effectively learn and distinguish features essential for accurate classification.
# 
# Overall, the achieved accuracy of 97.6% on the test data, coupled with the similar validation accuracy, demonstrates the robustness and generalization capability of the CNN model. These results highlight the suitability of CNNs for image classification tasks, providing accurate predictions while minimizing the risk of overfitting.

# In[22]:




