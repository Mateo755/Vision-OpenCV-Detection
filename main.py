import argparse
import json
from pathlib import Path

import cv2

from processing.utils import perform_processing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    results_file = Path(args.results_file)

    videos_paths = sorted([video_path for video_path in videos_dir.iterdir() if video_path.name.endswith('.mp4')])
    results = {}
    for video_path in videos_paths:
        cap = cv2.VideoCapture(str(video_path))
        if cap is None:
            print(f'Error loading video {video_path}')
            continue
        else:
            print(f'Processing video {video_path}')

        results[video_path.name] = perform_processing(cap)

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()