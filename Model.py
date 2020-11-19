from abc import ABC, abstractmethod

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


    @abstractmethod
    def plot(self):
        '''
        Plot relevant information as accuracy or loss.
        '''
        pass
