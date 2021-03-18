import cv2 # Open source computer vision library for computer vision tasks
import os # For interacting with the operating system

# Pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img, draw_box=True):
	# Convert image to grayscale
	grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect faces
	faces = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.1, # Reduce the size by 5%
		minNeighbors=5, # Higher value results in fewer detections but with higher quality
        minSize=(30, 30), # How small size we want to detect
        flags=cv2.CASCADE_SCALE_IMAGE) # For an old cascade classifier
                                       # 0 for new format for cascade classifier 
	
	face_box, face_coords = None, []
    # Draw bounding box around detected faces
	for (x, y, w, h) in faces:
		if draw_box:
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5) # Img, Start, Color and Thickness
		face_box = img[y:y+h, x:x+w] # Crop image
		face_coords = [x,y,w,h] # Starting and ending x and Y Coordinates of face

	return img, face_box, face_coords 
