
from Data import DataLoader
from Word2vec import Word2vec
from SimpleWordModel import SimpleWordModel

from sklearn.model_selection import train_test_split

# %%

trans = Word2vec(verbose=1)

data = DataLoader(trans)

# %%

x_dialog, y_dialog = data.dialoguesLoader()

print(x_dialog.shape)
print(y_dialog.shape)

# %%

x_single, y_single = data.singleLoader(verbose=0)

print(x_single.shape)
print(y_single.shape)


#%%

X_train, X_test, y_train, y_test = train_test_split(x_single, y_single, test_size=0.2, random_state=123)

print(X_train.shape)


# %%
'''
Single word model. Train the model with a different dataset. Then prove it with
the original one.
'''

batch_size = 200
epochs = 70

input_shape    = (data.trans.shape(), )
number_classes = len(data.emotions)

model = SimpleWordModel(input_shape, number_classes)

model.train(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))


#model.plot()

# %%

model.predict(x_dialog, y_dialog)

