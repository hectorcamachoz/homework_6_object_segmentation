""" test-object-detection.py
This script contains the pipeline to test the object detection on a video

Authors: Alberto Castro Villasana , Ana Bárbara Quintero, Héctor Camacho Zamora
Organisation: UDEM
First created on Monday 18 March 2024
USAGE: 
    $ python test-object-detection.py --video_file football-field-cropped-video.mp4 --frame_resize_percentage 30
"""

# Import standard libraries
import cv2
import argparse
import numpy as np
from numpy.typing import NDArray
from od import parse_cli_data, initialise_camera, rescale_frame, segment_object, close_windows


# Pipeline function
def run_pipeline():
    # Parse the command line arguments
    args = parse_cli_data()

    # Initialise video capture
    cap = initialise_camera(args)

    # Process video
    segment_object(cap, args)
    
    # Close all open windows
    close_windows(cap)

if __name__ == "__main__":
    # Run the pipeline
    run_pipeline()