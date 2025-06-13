from processing.detection import detect_from_background_image

def perform_processing(cap):
    all_objects = detect_from_background_image(cap, show=True)

