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
    """
    This function parses the command line arguments

    Returns:
        argparse: The command line arguments
    """
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
    """
    This function initialises the camera to capture the video
    
    Args:
        args (argparse): The command line arguments

    Returns:
        cv2.VideoCapture: The video capture object
    """
    # Create a video capture object
    cap = cv2.VideoCapture(args.video_file)
    
    return cap

def rescale_frame(frame:NDArray, percentage:np.intc=20)->NDArray:
    """
    This function rescales the current frame

    Args:
        frame (NDArray): The current frame
        percentage (np.intc): The percentage to rescale the frame

    Returns:
        NDArray: The rescaled frame
    """
    # Resize current frame
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def segment_object(cap:cv2.VideoCapture, args:argparse)->None:
    """
    This function segments the object in the video

    Args:
        cap (cv2.VideoCapture): The video capture object
        args (argparse): The command line arguments

    """
    # Main loop
    x_center_old = 0
    y_center_old = 0
    while cap.isOpened():
        # Read current frame
        ret, frame = cap.read()

        if not ret:
            print("ERROR! - current frame could not be read")
            break

        frame = rescale_frame(frame, args.frame_resize_percentage)
        frame = cv2.medianBlur(frame, 5)
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply a threshold to the HSV image
        frame_threshold = cv2.inRange(frame_HSV, 
                                      (66, 80, 33),(180, 255, 255))

        # Filter out the grassy region from current frame, but keep the moving object 
        bitwise_AND = cv2.bitwise_and(frame, frame, mask=frame_threshold)

        # Encuentra los contornos en la imagen binaria

        cnts, _ = cv2.findContours(frame_threshold, 
                                   cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_SIMPLE) 
        
        # Si no hay contornos, continúa con la siguiente iteración del bucle
        if not cnts:
            continue

        # Encuentra el contorno con la mayor área
        largest_cnt = max(cnts, key=cv2.contourArea)

        # Obtiene el rectángulo del contorno más grande y ajusta su tamaño si es necesario
        x, y, w, h = cv2.boundingRect(largest_cnt)  
        # x, y son las coordenadas del punto superior izquierdo, w, h son el ancho y la altura
        
        # Aquí puedes ajustar el tamaño del rectángulo fijo
        fixed_width, fixed_height = 30, 30  # Tamaño fijo para el rectángulo
        x_center, y_center = x + w // 2, y + h // 2

        # Establece una condicion para evitar que el rectangulo dibujado continue con el contorno que deseamos
        if x_center - 10  < x_center_old < x_center + 10 or x_center_old == 0 and y_center - 10  < y_center_old < y_center + 10 or y_center_old == 0 :

            # Ajusta las coordenadas para centrar el rectángulo fijo sobre el contorno más grande
            x_fixed = max(0, x_center - fixed_width // 2)
            y_fixed = max(0, y_center - fixed_height // 2)

             # Dibuja el rectángulo fijo
            cv2.rectangle(frame, (x_fixed, y_fixed), 
                        (x_fixed + fixed_width, y_fixed + fixed_height), 
                        (0, 255, 0), 2)
            x_center_old = x_center
        
        # Muestra los videos
        cv2.imshow(window_params['capture_window_name'], frame)
        cv2.imshow(window_params['detection_window_name'], bitwise_AND)
        cv2.imshow('rect', frame_threshold)

        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Program finished!")
            break

                

def close_windows(cap:cv2.VideoCapture)->None:
    
    # Destroy all visualisation windows
    cv2.destroyAllWindows()

    # Destroy 'VideoCapture' object
    cap.release()

"""
def run_pipeline(args:argparse)->None:

    # Initialise video capture
    cap = initialise_camera(args)

    # Process video
    segment_object(cap, args)
    
    # Close all open windows
    close_windows(cap)



if __name__=='__main__':

    # Get data from CLI
    args = parse_cli_data()

    # Run pipeline
    run_pipeline(args)
"""            





