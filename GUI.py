import tkinter
from tkinter import Tk
from tkinter import Frame
from tkinter import Label
from tkinter import PhotoImage
from tkinter import Button
from tkinter import StringVar
from PIL import Image as Img, ImageTk
import datetime
import cv2
import os
import numpy as np
import math 
import time

from keras.models import load_model
#import argparse
from PIL import Image
import imutils

screen_width_home = 720
screen_height_home = 480

screen_width = 645
screen_height = 620

class GUI(Tk):
    def __init__(self):
        Tk.__init__(self)
        container = Frame(self)
        self.title("ANOMALY DETECTION IN SURVEILLANCE VIDEOS")
        self.iconbitmap(r'CCET_logo3.ico')
        
        self.current_frame="gui"
        print(self.current_frame)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
    

        for F,geometry in zip((StartPage, PageOne), (f"{screen_width_home}x{screen_height_home}", f"{screen_width}x{screen_height}", f"{screen_width}x{screen_height}", f"{screen_width}x670")):
            page_name = F.__name__

            frame = F(container, self)

            self.frames[page_name] = (frame, geometry)

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame, geometry = self.frames[page_name]

        # change geometry of the window
        self.update_idletasks()
        self.geometry(geometry)
        frame.tkraise()


    def status(self, value):
        self.statusVar = StringVar()
        self.statusVar.set(value)
        self.statusBar = Label(self, textvariable=self.statusVar, anchor="w", relief=tkinter.SUNKEN)
        self.statusBar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        
    def mean_squared_loss(self, x1, x2):
        difference=x1-x2
        a,b,c,d,e=difference.shape
        n_samples=a*b*c*d*e
        sq_difference=difference**2
        Sum=sq_difference.sum()
        distance=np.sqrt(Sum)
        mean_distance=distance/n_samples
    
        return mean_distance
        
    def video_loop(self):
        # Get frame from the video stream and show it in Tkinter
        imagedump=[]
        ret,frame=self.vs.read()
    
    
        for i in range(10):
            ret,frame=self.vs.read()
            image = imutils.resize(frame,width=1000,height=1200)
    
            frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
            gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
            gray=(gray-gray.mean())/gray.std()
            gray=np.clip(gray,0,1)
            imagedump.append(gray)
    
        imagedump=np.array(imagedump)
    
        imagedump.resize(227,227,10)
        imagedump=np.expand_dims(imagedump,axis=0)
        imagedump=np.expand_dims(imagedump,axis=4)
    
        output=model.predict(imagedump)
    
        loss=self.mean_squared_loss(imagedump,output)
    
        if frame.any()==None:
            print("No Frame")
        
        if cv2.waitKey(10) & 0xFF==ord('q'):
            self.destructor()
        
        if loss>0.00068:
            print('Abnormal Event Detected')
            cv2.putText(image,"Abnormal Event",(220,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
                        

        if ret:  # frame captured without any errors
            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            
            self.current_image = Img.fromarray(cv2image)  # convert image for PIL
            
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            
            self.panel.config(image=imgtk)  # show the image
        
        self.after(30, self.video_loop)  # call the same function after 30 milliseconds

    def take_snapshot(self):
        """ Take snapshot and save it to the file """
        ts = datetime.datetime.now() # grab the current timestamp
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
        p = os.path.join(self.output_path, filename)  # construct output path
        self.current_image.save(p, "PNG")  # save image as PNG file
        print("[INFO] saved {}".format(filename))

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        window.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

class StartPage(Frame, GUI):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        
        ##
        self.current_frame="start"
        print("func start")
        print(self.current_frame)
        ##
        
        backgroundImage = Img.open("Background12_2.jpg")
        self.reset_backgroundImage = ImageTk.PhotoImage(backgroundImage)

        backgroundLabel = Label(self, image=self.reset_backgroundImage)
        backgroundLabel.place(x=0, y=0, relwidth=1, relheight=1)
        backgroundLabel.image = backgroundImage # reference to the image, otherwise the image will be destroyed by the garbage collector when the function returns. 
                                                # Adding a reference as an attribute of the label object.

        backgroundImage2 = PhotoImage(file="Background4.png")        
                 
        head_image = Img.open("head4a.png")
        self.reset_head_image = ImageTk.PhotoImage(head_image)
        label = Label(self, image=self.reset_head_image)
        label.pack(side=tkinter.TOP, fill=tkinter.X)
    
        
        image3 = Img.open("button_start.png")
        self.reset_img3 = ImageTk.PhotoImage(image3)
        self.button3=Button(self,image=self.reset_img3, command=lambda: controller.show_frame("PageOne"))
        self.button3.place(anchor="n", relx=0.5, rely=0.5)

        self.status("HOME PAGE")
      

class PageOne(Frame, GUI):
    def __init__(self, parent, controller, output_path = "./"):
        Frame.__init__(self, parent)
        self.controller = controller
        
        #
        self.current_frame="three"
        print(self.current_frame)
        #
        '''
        backgroundImage = PhotoImage(file="Background8.png")        
         
        backgroundLabel = Label(self, image=backgroundImage)
        backgroundLabel.place(x=0, y=0, relwidth=1, relheight=1)
        backgroundLabel.image = backgroundImage # reference to the image, otherwise the image will be destroyed by the garbage collector when the function returns. 
                                                # Adding a reference as an attribute of the label object.

        # label = Label(self, text="SOCIAL DISTANCING AND FACE MASK MONITOR", font=LARGE_FONT)
        # label.pack(pady=10, padx=10)
        
        head_image = Img.open("BG.JPG")
        self.reset_head_image = ImageTk.PhotoImage(head_image)
        label = Label(self, image=self.reset_head_image, relief="groove")
        label.pack(side=tkinter.TOP, fill=tkinter.X)
        '''
        # Initialize application which uses OpenCV + Tkinter. It displays
        #    a video stream in a Tkinter window and stores current snapshot on disk 
        self.vs = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera

        self.panel = Label(self)  # initialize image panel
        self.panel.pack()
        
        
        image0 = Img.open("camera_button7.png")
        image0 = image0.resize((45,45), Img.ANTIALIAS)
        self.reset_img0 = ImageTk.PhotoImage(image0)
        self.button1=Button(self,image=self.reset_img0, command=self.take_snapshot)
        self.button1.place( relx=0.5, rely=0.92)
        
        image1 = Img.open("home_button10.png")
        image1 = image1.resize((45,45), Img.ANTIALIAS)
        self.reset_img1 = ImageTk.PhotoImage(image1)
        self.button1=Button(self,image=self.reset_img1, command=lambda: controller.show_frame("StartPage"))
        self.button1.place(relx=0, rely=0.0)
            
        image4 = Img.open("close_button1.png")
        image4 = image4.resize((45,45), Img.ANTIALIAS)
        self.reset_img4 = ImageTk.PhotoImage(image4)
        self.button4=Button(self,image=self.reset_img4, command=self.destructor)
        self.button4.place( relx=0.92, rely=0.0)
        
        self.status("ANOMALY DETECTION IN SURVEILLANCE VIDEOS")
        self.video_loop()


if __name__ == "__main__":
    '''    
    model = load_model('model_face_mask.model')  # Loading the Model
    
    face_clsfier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Casscade classifier to get the region of interest

    labels_dict={0:'WITH MASK',1:'WITHOUT MASK'}  # Creating dictionary in which 0 : WITH MASK and 1 : WITHOUT MASK
    color_dict={0:(0,255,0),1:(0,0,255)}   # Creating dictionary in which 0 : GREEN COLOR and 1 : RED COLOR
    
    labelsPath = "./coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    
    weightsPath = "./yolov3.weights"
    configPath = "./yolov3.cfg"
    
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)'''
    
    model=load_model("saved_model.h5")

    
    window = GUI()
    window.mainloop()

