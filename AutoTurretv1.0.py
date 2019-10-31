'''
DrakeCam V1.0  (Current as of 1900S18OCT19)
Author: Jeff Wright  (Twitter: @RockOrSomething)

This script uses Tensorflow to perform real-time object detection and classification of a human subject that an RCNN-InceptionV2 model
has been trained to recognize.  Once the program detects the subject (the author's son, Drake) and determines that the object is indeed
him (to a certainty of 95%), then the program uses a combination of X and Y-axis servos to track him with the sensor/LASER assembly.
The servos physically move the assembly in the direction of the object, and once the center-of-mass of the object is within 10% of the
center of the camera's frame, the LASER fires.

As of this writing, the system works flawlessly - albeit so slowly as to be useless.  It is only getting 1 frame every 5-20 seconds.

The author built heavily on Evan Juras' work for this project.  Mr. Juras' explaination of how to perform object detection on
the Raspberry Pi provided the step-by-step instructions for how to get started, and much of the object-detection code that
follows was written by him.  Many other sources were used or borrowed from as well - too many to list individually!
'''

# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera.array import PiArrayOutput 
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import RPi.GPIO as GPIO
from time import sleep
import pigpio

# Set up GPIOs on RPi and declare global variable for aiming angle
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT) #LASER Pin
pi = pigpio.pi()
pi.set_servo_pulsewidth(3,0) #X-Axis servo initialized, turned off to prevent buzzing
pi.set_servo_pulsewidth(2,0) #Y-Axis servo, same

currentX = 1700
currentY = 1200

''' This section is mothballed - the standard RPi.GPIO library doesn't work smoothly with the Pi4...creates jitter that wasn't there
in previous RPi models.  The pigpio library is far smoother and more powerful.  It will eventually replace all RPi.GPIO functions.
#GPIO.setup(3, GPIO.OUT) #X Servo Pin
#GPIO.setup(2, GPIO.OUT) #Y Servo Pin 
#xServo = GPIO.PWM(2,50)
#yServo = GPIO.PWM(3,50)
#xServo.start(0)
#yServo.start(0)
'''
   
#Define methods to aim camera/LASER mechanism, and fire LASER
def setAngleX(adjX):
    try:
        global currentX
        print("X adjustment factor = ", adjX)
        updatedX = (currentX - (adjX * 300)) 
        print("New X aimpoint = ", updatedX)
        pi.set_servo_pulsewidth(3, updatedX)
        sleep(.3)
        pi.set_servo_pulsewidth(3, 0)
        currentX = updatedX
    except:
        print("Error updating angle X")
    
def setAngleY(adjY):
    try:
        global currentY
        print("Y adjustment factor = ", adjY)
        updatedY = (currentY + (adjY * 150)) 
        print("New X aimpoint = ", updatedY)
        pi.set_servo_pulsewidth(2, updatedY)
        sleep(.3)
        pi.set_servo_pulsewidth(2, 0)
        currentY = updatedY
    except:
        print("Error updating angle Y")
         
    
def fireLaser():
    print("LASER ON")
    for x in range(0, 10):
        GPIO.output(4, True)
        sleep(.2)
        GPIO.output(4, False)
        sleep(.2)
    print("LASER OFF")    
        
#Initialize servos to point sensor/LASER assembly straight ahead.
print("Initializing X to", currentX)
pi.set_servo_pulsewidth(3, 1000)
sleep(.5)
pi.set_servo_pulsewidth(3, 2000)
sleep(.5)
pi.set_servo_pulsewidth(3, currentX)
sleep(.5)
pi.set_servo_pulsewidth(3, 0)

print("Initializing Y to", currentY)
pi.set_servo_pulsewidth(2, 1100)
sleep(.5)
pi.set_servo_pulsewidth(2, 1500) #Physical limits of my setup.  The servo is capable of more, but in testing this was max/min.
sleep(.5)
pi.set_servo_pulsewidth(2, currentY)
sleep(.5)
pi.set_servo_pulsewidth(2, 0)

fireLaser() #Test fire...#DeskPop

# Set up camera constants
IM_WIDTH = 640    
IM_HEIGHT = 480
camera_type = 'picamera'


# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = '/home/pi/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1 #This is greatly simplified for the purpose of this project.  The methods used are capable of so much more.

## Load the label map.
label_map = label_map_util.load_labelmap('/home/pi/tensorflow1/models/research/object_detection/training/labelmap.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 5
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
def trackAndShoot(yn, yx, xn, xx):
    ymin = yn  
    xmin = yx
    ymax = xn
    xmax = xx     #Validated   Top left = 0/0, bottom right = 1/1
  
    xCenterMass = (xmin + xmax)/2
    yCenterMass = (ymin + ymax)/2
    print("Center of Mass = ", xCenterMass, yCenterMass)
    
    xOffTarget = (xCenterMass - .5)
    yOffTarget = (yCenterMass - .5) 
    
    if (abs(xOffTarget) < .1) and (abs(yOffTarget) < .1):
        print("On Target!  FIRE!")
        fireLaser()
        
    else:
        #print("Adjusting X to ", xCenterMass)
        #print("Adjusting Y to ", yCenterMass)
        
        setAngleX(xOffTarget)
        setAngleY(yOffTarget)

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 1
    camera.rotation = 270
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)
        
        '''
        #Determine center of the frame
        img = np.copy(frame1.array)
        rows = img.shape[0]
        cols = img.shape[1]
        centerMassX = (rows/2) 
        centerMassY = (cols/2)
        '''
        
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})       
        print(scores[0][0])
        if scores[0][0] > .95:
            drakeDetected = True
            print("Drake detected!")
            #print("Sending coordinates ", boxes[0][0])
            yn = boxes[0][0][0]
            xn = boxes[0][0][1]
            yx = boxes[0][0][2]
            xx = boxes[0][0][3]
            trackAndShoot(yn, xn, yx, xx)
            #print("Sent coordinates ", boxes[0][0])
        
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.90)
        
            
        
        #Display FPS information
        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

cv2.destroyAllWindows()
GPIO.cleanup()
print("Successfully shut down.")
