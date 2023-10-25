from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from google.colab import drive
drive.mount('/content/drive')

model = Sequential()

model.add(Conv2D(32,(3,3),padding = 'same', activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32,(3,3),padding = 'same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding = 'same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding = 'same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding = 'same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding = 'same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding = 'same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(128,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(4,activation='softmax'))

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/mdd/train',
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'categorical'
)

val_set = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/mdd/test',
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'categorical'
)

history = model.fit(
    training_set,
    steps_per_epoch = 12,
    epochs = 10,
    validation_data = val_set,
    validation_steps = 15
) 
