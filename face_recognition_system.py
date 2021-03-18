import os # For interacting with the operating system
import sys # Exit from Python
import cv2 # Open source computer vision library for computer vision tasks
import helper # Module containing few function
from keras.models import load_model # Loads a model saved via model.save()
import face_detection # Module containing function to detect face in image
import collections # To implements specialized container datatypes providing alternatives 
                   # to Python’s general purpose built-in containers

with open("name.txt", "r") as name_file:
    name = name_file.read()

# Check availability of trained model
files = os.listdir() # Get the list of files in current directory
if 'siamese_nn.h5' not in files: 
    print("Error: Pre-trained Neural Network not found!")
    print("Please train the model first")
    sys.exit() # Exit from Python  

# Check availability of face image
if 'true_img.png' not in files:
    print("Error: True image not found!")
    print("Please run get_face.py first")
    sys.exit() # Exit from Python

# Load pre-trained Siamese neural network
model = load_model('siamese_nn.h5', custom_objects={'contrastive_loss': helper.contrastive_loss, 'euclidean_distance': helper.euclidean_distance})

# Prepare the true image obtained during onboard
true_img = cv2.imread('true_img.png', 0) # Loads the image from file
true_img = true_img.astype('float32')/255 # Cast to float32 data type
true_img = cv2.resize(true_img, (92, 112)) # Reduce the number of pixels in image
true_img = true_img.reshape(1, true_img.shape[0], true_img.shape[1], 1) # Change the shape of image array
                                                                        # 0 : No. of rows, 1 : Number of column

video_capture = cv2.VideoCapture(0) # Returns video from the first webcam of computer
preds = collections.deque(maxlen=15) # Deque : List-like container with fast appends and pops on both ends

while True:
    # Capture frames from webcam
    _, frame = video_capture.read() # Returns image as array

    # Detect Faces
    frame, face_img, face_coords = face_detection.detect_faces(frame, draw_box=False)  # Calling function 
                                                                                    # to detect face in image

    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) 	# Convert image to grayscale
        face_img = face_img.astype('float32')/255 # Cast to float32 data type
        face_img = cv2.resize(face_img, (92, 112)) # Reduce the number of pixels in image
        face_img = face_img.reshape(1, face_img.shape[0], face_img.shape[1], 1) #Change the shape of image array
                                                                        # 0 : No. of rows, 1 : Number of column
        preds.append(1-model.predict([true_img, face_img])[0][0])
        x,y,w,h = face_coords # Coordinates of face in image
        if len(preds) == 15 and sum(preds)/15 >= 0.3:
            text = "Identity: {}".format(name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
        elif len(preds) < 15:
            text = "Identifying ..."
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 5)
        else:
            text = "Identity Unknown!"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
        frame = helper.write_on_frame(frame, text, face_coords[0], face_coords[1]-10)

    else:
        preds = collections.deque(maxlen=15) # Clear existing predictions if no face detected 

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit if user press 'q' key
        break

video_capture.release() # Release software resource and hardware resource
cv2.destroyAllWindows() # Destroys all the windows that are created
