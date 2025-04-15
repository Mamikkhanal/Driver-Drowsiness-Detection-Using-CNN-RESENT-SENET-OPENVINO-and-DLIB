import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

def getModel():
    model = Sequential()
    # First convolutional layer accepts image input
    model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', 
                            input_shape=(128, 128, 3)))
    # Add a max pooling layer
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', 
                            input_shape=(64, 64, 3)))
    # Add a max pooling layer
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))

    # Add a convolutional layer
    model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu', 
                            input_shape=(32, 32, 16)))

    # Add another max pooling layer
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))

    # Add a convolutional layer
    model.add(Conv2D(filters=8, kernel_size=5, padding='same', activation='relu', 
                            input_shape=(16, 16, 8)))

    # Add another max pooling layer
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))

    # Add a convolutional layer
    model.add(Conv2D(filters=4, kernel_size=5, padding='same', activation='relu', 
                            input_shape=(8, 8, 4)))
    # Add another max pooling layer
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Flatten and feed to output layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Summarize the model
    model.summary()
    model.compile(optimizer='adam', 
              loss=['binary_crossentropy'], 
              metrics=['accuracy'])

    return model