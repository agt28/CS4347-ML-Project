import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pickle
import os
import cv2

DATADIR = "D:\Images"

CATEGORIES = ["Fire", "Trees"]

'''
View images by folder of labeled data
'''
class cnnDataLoader(object):

    def __init__(self):
        super().__init__()
        self.IMG_SIZE = 28

    def view_dataset (self):
        for category in CATEGORIES:  
            path = os.path.join(DATADIR,category)  # create path to categories
            for img in os.listdir(path):  # iterate over each image per dogs and cats
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                plt.imshow(img_array, cmap='gray')  # graph it
                plt.show()  # display!

                #break  # we just want one for now so break
            break  #...and one more!


    def create_training_data(self):
        
        training_data = []

        for category in CATEGORIES:  # Fire and trees

            path = os.path.join(DATADIR,category)  # create path to dogs and cats
            class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

            for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
                try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                    new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))  # resize to normalize data size
                    training_data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    print ("[ Info ] [ Data Loader ] Could not load data")
                    pass
                    #print("[Info ] [Data ] Could not import data")
                #except OSError as e:
                #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                #except Exception as e:
                #    print("general exception", e, os.path.join(path,img))
        
        #print ("[ Info ] [ Data Loader ] Output list: ", sorted(training_data, key=lambda k: random.random()))
        return sorted(training_data, key=lambda k: random.random())
    
    def storeData (self, training_data):
        X = []
        y = []
        for features,label in training_data:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        
        pickle_out = open("cnnFireDetection\data\X.pickle","wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("cnnFireDetection\data\y.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    def loadData (self):
        pickle_in = open("cnnFireDetection\data\X.pickle","rb")
        X = pickle.load(pickle_in)

        pickle_in = open("cnnFireDetection\data\y.pickle","rb")
        y = pickle.load(pickle_in)
        
        return X, y