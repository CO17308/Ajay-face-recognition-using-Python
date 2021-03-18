import numpy as np # For working with arrays
import random 
import os # For interacting with the operating system
import cv2 # Open source computer vision library for computer vision tasks
from keras.models import Sequential # To group linear stack of layers into a keras Model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array

def euclidean_distance(vectors): # Function to compute the Euclidean distance between two vectors
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True) # Calculate sum of squared differences
    return K.sqrt(K.maximum(sum_square, K.epsilon())) # Square root of sum of squared differences

def contrastive_loss(Y_true, D): # Function for calculating the contrastive loss
    margin = 1
    return K.mean(Y_true * K.square(D) + (1 - Y_true) * K.maximum((margin-D),0))

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def create_pairs(X,Y, num_classes): # Generate negative (different subject) and positive (same subject) 
                                    # pairs of images for training a Siamese neural network
    pairs, labels = [], []
    # index of images in X and Y for each class
    class_idx = [np.where(Y==i)[0] for i in range(num_classes)]
    
    # The minimum number of images across all classes
    min_images = min(len(class_idx[i]) for i in range(num_classes)) - 1
  
    for c in range(num_classes):
        for n in range(min_images):
            
            # create positive pair
            img1 = X[class_idx[c][n]] # img1 from c class
            img2 = X[class_idx[c][n+1]] # img2 from c class
            pairs.append((img1, img2)) # Appending pair of images in pair list
            labels.append(1) # Appending label in label list (1: Same subject)
      
            # create negative pair
            # first, create list of classes that are different from the current class
            neg_list = list(range(num_classes))
            neg_list.remove(c)
            # select a random class from the negative list. 
            # this class will be used to form the negative pair
            neg_c = random.sample(neg_list,1)[0]
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[neg_c][n]]
            pairs.append((img1,img2))
            labels.append(0)

    return np.array(pairs), np.array(labels)

def create_shared_network(input_shape): # Used to create a Siamese neural network in Keras
    model = Sequential(name='Shared_Conv_Network')
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=input_shape)) # 2D convolution layer 
    model.add(MaxPooling2D()) # MaxPooling to select higher intensity values
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu')) # 2D convolution layer 
    model.add(Flatten()) # Convert output of conv layer into 1D feature vector
                         # Desnse layer expects data in one dimention
    model.add(Dense(units=128, activation='sigmoid')) 
    return model

def get_data(dir): # To load the respective raw images into NumPy arrays
    X_train, Y_train = [], [] # Training data
    X_test, Y_test = [], [] # Testing data
    subfolders = sorted([file.path for file in os.scandir(dir) if file.is_dir()]) #Sorted list of folders
    for idx, folder in enumerate(subfolders):
        for file in sorted(os.listdir(folder)):
            img = load_img(folder+"/"+file, color_mode='grayscale') # Load image from folder
            img = img_to_array(img).astype('float32')/255 # Normalize image
            img = img.reshape(img.shape[0], img.shape[1],1) # Reshape image array
            if idx < 35: # First 35 data in train
                X_train.append(img)
                Y_train.append(idx)
            else: # From 35 to 40 in test
                X_test.append(img)
                Y_test.append(idx-35)
    # Convert into numpy array
    X_train = np.array(X_train) 
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    return (X_train, Y_train), (X_test, Y_test)

def write_on_frame(frame, text, text_x, text_y): # To write on frames
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
    box_coords = ((text_x, text_y), (text_x+text_width+20, text_y-text_height-20))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED) # Draw rectangle
    cv2.putText(frame, text, (text_x, text_y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2) # Put text
    return frame


