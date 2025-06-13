import cv2

def save_background_from_time(video_path: str, seconds: int, output_path: str = "background.jpg"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Zapisano tło do: {output_path}")
    else:
        print("Nie udało się pobrać klatki.")

    cap.release()


# wywołanie
save_background_from_time("videos/00000.mp4", seconds=137)