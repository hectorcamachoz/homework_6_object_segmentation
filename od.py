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


# Define and initialise global variables
HSV_params = {'low_H': 0, 
              'high_H': 180,
              'low_S': 0,
              'high_S': 255,
              'low_V': 0,
              'high_V': 255
            }

window_params = {'capture_window_name':'Input video',
                 'detection_window_name':'Detected object'}

text_params = {'low_H_text': 'Low H',
               'low_S_text': 'Low_S',
               'low_V_text': 'Low V',
               'high_H_text': 'High H',
               'high_S_text': 'High S',
               'high_V_text': 'High V'}



def on_low_H_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['low_H'] = val
    HSV_params['low_H'] = min(HSV_params['high_H']-1, HSV_params['low_H'])
    cv2.setTrackbarPos(text_params['low_H_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['low_H'])


def on_high_H_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['high_H'] = val
    HSV_params['high_H'] = max(HSV_params['high_H'], HSV_params['low_H']+1)
    cv2.setTrackbarPos(text_params['high_H_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['high_H'])


def on_low_S_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['low_S'] = val
    HSV_params['low_S'] = min(HSV_params['high_S']-1, HSV_params['low_S'])
    cv2.setTrackbarPos(text_params['low_S_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['low_S'])


def on_high_S_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['high_S'] = val
    HSV_params['high_S'] = max(HSV_params['high_S'], HSV_params['low_S'] +1)
    cv2.setTrackbarPos(text_params['high_S_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['high_S'])


def on_low_V_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['low_V'] = val
    HSV_params['low_V'] = min(HSV_params['high_V']-1, HSV_params['low_V'])
    cv2.setTrackbarPos(text_params['low_V_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['low_V'])


def on_high_V_thresh_trackbar(val:np.int_)->None:
    global HSV_params
    HSV_params['high_V'] = val
    HSV_params['high_V'] = max(HSV_params['high_V'], HSV_params['low_V']+1)
    cv2.setTrackbarPos(text_params['high_V_text'], 
                       window_params['detection_window_name'], 
                       HSV_params['high_V'])


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

def configure_trackbars()->None:

    # Create two new windows for visualisation purposes 
    cv2.namedWindow(window_params['capture_window_name'])
    cv2.namedWindow(window_params['detection_window_name'])

    # Configure trackbars for the low and hight HSV values
    cv2.createTrackbar(text_params['low_H_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['low_H'], 
                       180, 
                       on_low_H_thresh_trackbar)
    cv2.createTrackbar(text_params['high_H_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['high_H'], 
                       180, 
                       on_high_H_thresh_trackbar)
    cv2.createTrackbar(text_params['low_S_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['low_S'], 
                       255, 
                       on_low_S_thresh_trackbar)
    cv2.createTrackbar(text_params['high_S_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['high_S'], 
                       255, 
                       on_high_S_thresh_trackbar)
    cv2.createTrackbar(text_params['low_V_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['low_V'], 
                       255, 
                       on_low_V_thresh_trackbar)
    cv2.createTrackbar(text_params['high_V_text'], 
                       window_params['detection_window_name'] , 
                       HSV_params['high_V'], 
                       255, 
                       on_high_V_thresh_trackbar)


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
                                      (HSV_params['low_H'], 
                                       HSV_params['low_S'], 
                                       HSV_params['low_V']), 
                                      (HSV_params['high_H'], 
                                       HSV_params['high_S'], 
                                       HSV_params['high_V']))

        # Filter out the grassy region from current frame, but keep the moving object 
        bitwise_AND = cv2.bitwise_and(frame, frame, mask=frame_threshold)

        # Visualise both the input video and the object detection windows
        cv2.imshow(window_params['capture_window_name'], frame)
        cv2.imshow(window_params['detection_window_name'], bitwise_AND)

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
    configure_trackbars()

    # Process video
    segment_object(cap, args)

    # Close all open windows
    close_windows(cap)



if __name__=='__main__':

    # Get data from CLI
    args = parse_cli_data()

    # Run pipeline
    run_pipeline(args)

