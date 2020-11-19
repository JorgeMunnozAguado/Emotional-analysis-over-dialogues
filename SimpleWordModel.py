from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras

from Model import Model


class SimpleWordModel(Model):
    
    def __init__(self, input_shape, number_classes, verbose=0):
        '''
        Generates a simple model word2word.
        '''
    
        model = Sequential()
        
        model.add(Dense(20, input_shape=input_shape, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(number_classes, activation='softmax'))
        
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
        if verbose: print(model.summary())

        self.model = model      

        
        
    def train(self, X_train, y_train, epochs, batch_size, validation_data=None, verbose=0):
        '''
        Train the model word2word with the word dataset.
        '''
        
        self.history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=validation_data)
    
    
    def test(self, X_test, y_test):
        '''
        Test the model with test data.
        '''
        
        for data in X_test:
            
            for word in data:
                
                print(word)
        
        pass
    

    def predict(self, X_test):
        '''
        Predict the data and returns the classes.
        '''
        pass


    def plot(self):
        pass