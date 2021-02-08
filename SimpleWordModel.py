from keras.models import Sequential
from keras.layers import Dense, Dropout

from Model import Model

import keras
import numpy as np


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
    

    def predict(self, X_test, y_test):
        '''
        Predict the data and returns the classes.
        '''
        
        predictions = []
        acc = 0
        
        for i, x_dialog in enumerate(X_test, start=0):

            if x_dialog.shape == (0,):  continue
            
            if x_dialog.ndim == 1:  x_dialog = np.expand_dims(x_dialog, axis=0)
        
            p = self.model.predict(x_dialog)
            
            p = np.argmax( np.mean(p, axis=0) )
            
            try:
                if y_test[i] == p:  acc += 1
                
            except:
                print(i)
            
            predictions.append( p )
            
            
        acc = acc / X_test.shape[0]

        print('Accuracy:', acc, '%')
        
        return np.asarray(predictions), acc

