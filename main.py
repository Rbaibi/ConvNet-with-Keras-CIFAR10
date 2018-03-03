#Import Modules
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.constraints import maxnorm
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


num_classes = 10
epochs = 40

#Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Normaliation
predictors = x_train.astype('float32')
predictors /= 255
test_predictors = x_test.astype('float32')
test_predictors /= 255

#One hot encoding
target = np_utils.to_categorical(y_train)
test_target = np_utils.to_categorical(y_test)

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy'
              , optimizer='adam'
              , metrics=['accuracy'])

print(model.summary())

#Early stopping
#training will automatically stop if no progress is made after 2 epochs
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target
          , validation_split=0.3
          , validation_data=(test_predictors, test_target)
          , epochs=epochs
          , callbacks = [early_stopping_monitor])

#Print Accuracy
scores = model.evaluate(test_predictors, test_target)
print('Loss:', scores[0])
print('Accuracy:', scores[1])

#saving/ loading
model.save('model_file.h5')
#my_model = load_model('model_file.h5')
