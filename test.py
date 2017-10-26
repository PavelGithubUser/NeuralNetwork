from distlib.compat import raw_input
from keras.preprocessing import image
import numpy as np
from src import train, modelinit
import os


def launch():
    old_it = 0
    yong_it = 0
    all_old = 0
    all_yong = 0

    validation_data_dir_old = 'C:/data/validation/old'
    validation_data_dir_young = 'C:/data/validation/young'

    model = modelinit.launch()
    model.load_weights('weights.h5')

    files_old = os.listdir(validation_data_dir_old)
    files_young = os.listdir(validation_data_dir_young)

    for file in files_old:
        print('Old: ', file)
        img = image.load_img(os.path.join(validation_data_dir_old, file), target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        classes = model.predict(x, 16, 0)
        print(classes)
        ts = classes.tostring()
        tx = np.fromstring(ts, dtype=int)
        if str(tx) == '[0]':
            print("stariy")
            print('')
            old_it += 1
        elif str(tx) == '[1065353216]':
            print("molodoy")
            print('')
        else:
            print("unknown")
        print('---------------------------')
        all_old += 1

    for file in files_young:
        print('Young: ', file)
        img = image.load_img(os.path.join(validation_data_dir_young, file), target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        classes = model.predict(x, 16, 0)
        print(classes)
        ts = classes.tostring()
        tx = np.fromstring(ts, dtype=int)
        if str(tx) == '[0]':
            print("stariy")
            print('')
        elif str(tx) == '[1065353216]':
            print("molodoy")
            print('')
            yong_it += 1
        else:
            print("unknown")
        print('---------------------------')
        all_yong += 1


    print("old: ", old_it, "out of ", all_old)
    print("yong: ", yong_it, "out of ", all_yong)


    #
    # while True:
    #     try:
    #         img_path = str(raw_input("path for jpg: "))
    #         img = image.load_img(img_path, target_size=(150, 150))
    #         x = image.img_to_array(img)
    #         x = np.expand_dims(x, axis=0)
    #         classes = model.predict(x, 16, 0)
    #         print(classes)
    #         ts = classes.tostring()
    #         tx = np.fromstring(ts, dtype=int)
    #         print(tx)
    #         #print (tx)
    #         #print(str(classes))
    #
    #         if str(tx) == '[0]':
    #             print("test for " + str(img_path))
    #             print('')
    #             print("stariy")
    #             print('')
    #
    #         elif str(tx) == '[1065353216]':
    #             print("test for " + str(img_path))
    #             print('')
    #             print("molodoy")
    #             print('')
    #         else:
    #             print("unknown")
    #     except:
    #         print('err')