
#from Data import DataLoader
from Word2vec import Word2vec

# %%

trans = Word2vec(verbose=1)

#data = DataLoader(trans)

# %%

x, y = data.dialoguesLoader()

print(x.shape)
print(y.shape)

# %%

from Data import DataLoader

data = DataLoader(trans)

x, y = data.sigleLoader(verbose=0)
print(x.shape)
print(y.shape)


#%%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

#%%
print(X_train.shape)

#%%
# First model. Analyze each word independently.

# =============================================================================
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# import keras
# 
# 
# batch_size = 200
# epochs = 20
# 
# 
# 
# input_shape    = (data.trans.shape(), )
# number_classes = len(data.emotions)
# 
# model = Sequential()
# 
# model.add(Dense(20, input_shape=input_shape, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(number_classes, activation='softmax'))
# 
# 
# model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# 
# =============================================================================

from SimpleWordModel import SimpleWordModel

input_shape    = (data.trans.shape(), )
number_classes = len(data.emotions)

model = SimpleWordModel(input_shape, number_classes)

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))


# %%

from matplotlib import pyplot as plt


try:

    ac = False

    # Comprueba la versión de Keras.
    if int(keras.__version__.split('.')[0]) >= 2:

        if 'accuracy' in history.history:
        
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])

            ac = True


    # Otra version.
    else:

        if 'acc' in history.history:

            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])

            ac = True


    if ac:

        # Muestra la tabla de aciertos.
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        #plt.savefig("accuracy.png", dpi=200)
        plt.clf()


    # Muestra la tabla de fallos.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #plt.savefig("loss.png", dpi=200)
    plt.clf()


except Exception as e:

    print('Exception Error: ', e)
    
    
    
    
# %%

# Own predict function.


# https://cloud.google.com/ai-platform/prediction/docs/custom-prediction-routine-keras
# https://keras.io/guides/writing_a_training_loop_from_scratch/

model.predict(preprocessed_inputs)

