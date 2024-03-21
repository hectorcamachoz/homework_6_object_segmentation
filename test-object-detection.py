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
from od import parse_cli_data


# Pipeline function
def run_pipeline():
    parse_cli_data()

if __name__ == "__main__":
    # Run the pipeline
    run_pipeline()