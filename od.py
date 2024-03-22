""" od.py
This script contains the functions to do an object segmentation on a video

Authors: Alberto Castro Villasana , Ana Bárbara Quintero, Héctor Camacho Zamora
Organisation: UDEM
First created on Monday 18 March 2024
USAGE: 
    $ python od.py --video_file football-field-cropped-video.mp4 --frame_resize_percentage 30
"""

# Import standard libraries 
import cv2 
import argparse
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Union, List

# Import local libraries
#import cvlib as cvl


window_params = {'capture_window_name':'Input video',
                 'detection_window_name':'Detected object'}

def parse_cli_data()->argparse:
    parser = argparse.ArgumentParser(description='Tunning HSV bands for object detection')
    parser.add_argument('--video_file', 
                        type=str, 
                        default='camera', 
                        help='Video file used for the object detection process')
    parser.add_argument('--frame_resize_percentage', 
                        type=int, 
                        help='Rescale the video frames, e.g., 20 if scaled to 20%')
    args = parser.parse_args()

    return args


def initialise_camera(args:argparse)->cv2.VideoCapture:
    
    # Create a video capture object
    cap = cv2.VideoCapture(args.video_file)
    
    return cap

def rescale_frame(frame:NDArray, percentage:np.intc=20)->NDArray:
    
    # Resize current frame
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def segment_object(cap:cv2.VideoCapture, args:argparse)->None:

    # Main loop
    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()

        # Check if the image was correctly captured
        if not ret:
            print("ERROR! - current frame could not be read")
            break

        # Resize current frame
        frame = rescale_frame(frame, args.frame_resize_percentage)
        
        # Apply the median filter
        frame = cv2.medianBlur(frame,5)

        # Convert the current frame from BGR to HSV
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply a threshold to the HSV image
        frame_threshold = cv2.inRange(frame_HSV, 
                                      (80, 0, 33),(180, 255, 255))

        # Filter out the grassy region from current frame, but keep the moving object 
        bitwise_AND = cv2.bitwise_and(frame, frame, mask=frame_threshold)

       
        contours, _ = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:

            x, y, w, h = cv2.boundingRect(cnt)
       # print(x,y)
       # print(x + w, y + h)  
            area = cv2.contourArea(cnt)
            print(area)
            if area > 18 and area < 75 :
                
                cv2.rectangle(frame, (x-10, y-10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
            

            
       # cv2.drawContours(frame_threshold, cnt, -1 , (0, 255, 0), 2)


        # Visualise both the input video and the object detection windows
        cv2.imshow(window_params['capture_window_name'], frame)
        cv2.imshow(window_params['detection_window_name'], bitwise_AND)
        cv2.imshow('rect',frame_threshold)

        
        

        # The program finishes if the key 'q' is pressed
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Programm finished!")
            break
                

def close_windows(cap:cv2.VideoCapture)->None:
    
    # Destroy all visualisation windows
    cv2.destroyAllWindows()

    # Destroy 'VideoCapture' object
    cap.release()


def run_pipeline(args:argparse)->None:

    # Initialise video capture
    cap = initialise_camera(args)

    # Configure trackbars for the lowest and highest HSV values
   # configure_trackbars()

    # Process video
    frame, bitwise_AND = segment_object(cap, args)

    # Close all open windows
    close_windows()



if __name__=='__main__':

    # Get data from CLI
    args = parse_cli_data()

    # Run pipeline
    run_pipeline(args)

