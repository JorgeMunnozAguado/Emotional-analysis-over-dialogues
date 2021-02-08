from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

import keras

class Model(ABC):
    '''
    Simple class that generates and abstract class to use different
    models and platforms.
    '''

    @abstractmethod
    def train(self, X_train, y_train, epochs, validation_data=None, verbose=0):
        '''
        Train the model.
        '''
        pass


    @abstractmethod
    def test(self, X_test, y_test):
        '''
        Test the model with test data.
        '''
        pass


    @abstractmethod
    def predict(self, X_test):
        '''
        Predict the data and returns the classes.
        '''
        pass


    #@abstractmethod
    def plot(self):
        '''
        Plot relevant information as accuracy or loss.
        '''
        
        # Comprueba la versión de Keras.
        if int(keras.__version__.split('.')[0]) >= 2:
            
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])


        # Otra version.
        else:

            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])


        # Muestra la tabla de aciertos.
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        #plt.savefig("accuracy.png", dpi=200)
        plt.clf()


        # Muestra la tabla de fallos.
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        #plt.savefig("loss.png", dpi=200)
        plt.clf()
