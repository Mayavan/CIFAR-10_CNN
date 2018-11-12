import numpy as np
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot
import os

weight_decay = 0.0005
save_dir = os.path.join(os.getcwd(), 'saved_models')


class NeuralNetwork:
    def __init__(self, name):
        self.name = name
        model_path = os.path.join(save_dir, name)
        self.model = load_model(model_path)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # normalize inputs from 0-255 to 0.0-1.0
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # one hot encode outputs
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)

    # Fit the model
    def trainModel(self, epochs, batch_size):
        history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, validation_data=(self.x_test, self.y_test), epochs=epochs, verbose=1)
        try:
            os.mkdir('../Results/'+self.name[0: -3])
        except FileExistsError:
            print("File exists")
        # plot metrics
        pyplot.title(''.join([self.name[0: -3], " with mini batch size of ", str(batch_size)]))
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Loss')

        pyplot.plot(history.history['val_loss'], label='Validation Loss')
        pyplot.plot(history.history['loss'], label='Training Loss')
        pyplot.legend()
        pyplot.savefig('../Results/'+self.name[0: -3]+'/loss.png')
        pyplot.show()

        pyplot.title(''.join([self.name[0: -3], " with mini batch size of ", str(batch_size)]))
        pyplot.xlabel('Epochs')
        pyplot.ylabel('Accuracy')
        pyplot.plot(history.history['val_acc'], label='Validation Accuracy')
        pyplot.plot(history.history['acc'], label='Training Accuracy')
        pyplot.legend()
        pyplot.savefig('../Results/'+self.name[0: -3]+'/Accuracy.png')
        pyplot.show()

        self.model.save(self.name)

    # evaluate the model
    def testModel(self):
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print("Test data Loss is :", scores)


if __name__ == "__main__":
    # NeuralNetwork("ResNet56.h5").trainModel(100, 100)
    NeuralNetwork("wrn.h5").trainModel(100, 100)
    # NeuralNetwork("LeNet.h5").trainModel(100, 500)
    # NeuralNetwork("custom.h5").trainModel(100, 500)
    # NeuralNetwork("AlexNet.h5").trainModel(100, 50)
