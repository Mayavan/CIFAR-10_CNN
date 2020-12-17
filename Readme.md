# CIFAR10 with LeNet5, AlexNet, Wide Residual Network in Keras

## Clone this project to your computer:

```
git clone https://github.com/Mayavan/CIFAR-10_CNN.git
```
## Dependencies

Keras
Python 3.6

## To build the neural model

Uncomment the models you want to build in src/createNeuralNetworkModel.py and run the file.
The model will be created and saved in src/saved_models

## To train the neural model

Uncomment the model you want to train in the file src/NeuralNetwork.py.
The results will be plotted and saved in the result folder.

![TensorBoard](./images/tensorboard.png])

```
# Use tensor board to visualize training process
tensorboard --logdir ./logs/fit/
```
