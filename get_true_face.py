import cv2 # Open source computer vision library for computer vision tasks
import math # To perform mathamatical function
import helper # Modules containing few functions
import face_detection # Module to detect face in image

video_capture = cv2.VideoCapture(0) # Returns video from the first webcam of computer
counter = 5

name = input("Enter your name : ")
text_file = open("name.txt", "w")
n = text_file.write(name)
text_file.close()

while True:
    _, frame = video_capture.read() # Returns image as array
    frame, face_box, face_coords = face_detection.detect_faces(frame) # Calling function to detect face in image.
    text = 'Image will be taken in {} seconds.'.format(math.ceil(counter)) # Display remaining sec to take image.
    if face_box is not None: 
        frame = helper.write_on_frame(frame, text, face_coords[0], face_coords[1]-10) # Calling function to draw 
                                                                                     # rectangle on image
    cv2.imshow('Video', frame) # Display an image in a window
    cv2.waitKey(1) # Wait for one millisecond
    counter -= 0.1 
    if counter <= 0:
        cv2.imwrite('true_img.png', face_box) # Save face image in current directory
        break

video_capture.release() # Release software resource and hardware resource
cv2.destroyAllWindows() # Destroys all the windows that are created
print("Face Image Captured") # Display messages
