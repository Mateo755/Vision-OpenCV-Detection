import argparse
import json
from pathlib import Path

import cv2

from processing.utils import perform_processing


def main():


    results = {}
    cap = cv2.VideoCapture("./videos/wycinek_11m20s_12m02s_people.mp4")
    #cap = cv2.VideoCapture("./videos/wycinek_8m25s_10m30s_variety.mp4")
    #cap = cv2.VideoCapture("./videos/wycinek_2trams.mp4")
    #cap = cv2.VideoCapture("./videos/wycinek_13m00s_14m00s.mp4")
    #cap = cv2.VideoCapture("./videos/wycinek_0m25s_1m41s.mp4")
    #cap = cv2.VideoCapture("./videos/wycinek_0m33s_2m10s.mp4")
    if cap is None:
        print('Error loading video')
    else:
        print('Processing video')

    perform_processing(cap)



if __name__ == '__main__':
    main()