from keras.datasets import cifar10
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.utils import to_categorical

#Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# number of classes
num_classes = 10

# one hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


predictors =x_train
target = y_train


# specify architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Early stopping
#training will automatically stop if no progress is made after 5 epochs
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=8)

#fit
model.fit(predictors, target, validation_split=0.3, epochs=40,callbacks = [early_stopping_monitor])



#saving/ loading
model.save('/output/model_file.h5')
#my_model = load_model('my_model.h5')


#predict
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

