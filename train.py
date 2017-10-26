from keras.preprocessing.image import ImageDataGenerator
from src import modelinit

print('Weights saved to weights.h5')

def launch():

    # dimensions of our images.
    img_width, img_height = 150, 150
    #
    # train_data_dir = 'C:/Users/Pavel/PycharmProjects/RNN/data/train'
    # validation_data_dir = 'C:/Users/Pavel/PycharmProject/RNN/data/validation'
    train_data_dir = 'C:/data/train'
    validation_data_dir = 'C:/data/validation'
    nb_train_samples = 1800
    nb_validation_samples = 200
    epochs = 90
    batch_size = 18

    # this is the augmentation configuration we will use for training

    model = modelinit.launch()

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('weights.h5')