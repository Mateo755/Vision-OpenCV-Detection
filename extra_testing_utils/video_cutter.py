import cv2

def cut_video(input_path: str, output_path: str, start_time: float, end_time: float):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video segment saved to {output_path}")

# Przykład użycia
cut_video("./videos/00001.mp4", "wycinek_11m20s_12m02s_people.mp4", 680, 722)
