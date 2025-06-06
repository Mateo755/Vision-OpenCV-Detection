import cv2
import numpy as np

class ObjectTracker:
    def __init__(self):
        self.tracked = {}
        self.counted_ids = set()
        self.object_id = 0

    def update(self, current_centroids, frame_objects, frame_hsv):
        updated = []
        for cx, cy in current_centroids:
            matched_id = None
            best_score = float('inf')

            for obj_id, data in self.tracked.items():
                prev_cx, prev_cy = data['centroid']
                dist = np.hypot(cx - prev_cx, cy - prev_cy)
                if dist > 60:
                    continue

                # Compare histogram
                candidate = next((o for o in frame_objects if o['centroid'] == (cx, cy)), None)
                if candidate is None:
                    continue

                x, y, w, h = candidate['bbox']
                roi = frame_hsv[y:y+h, x:x+w]
                hist = cv2.calcHist([roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist)

                score = cv2.compareHist(data['descriptor'], hist, cv2.HISTCMP_BHATTACHARYYA)
                if score < best_score:
                    best_score = score
                    matched_id = obj_id

            if matched_id is not None:
                self.tracked[matched_id]['centroid'] = (cx, cy)
                updated.append((matched_id, cx, cy))
            else:
                candidate = next((o for o in frame_objects if o['centroid'] == (cx, cy)), None)
                if candidate is None:
                    continue
                x, y, w, h = candidate['bbox']
                roi = frame_hsv[y:y+h, x:x+w]
                hist = cv2.calcHist([roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist)

                self.tracked[self.object_id] = {
                    'centroid': (cx, cy),
                    'descriptor': hist,
                    'counted': False
                }
                updated.append((self.object_id, cx, cy))
                self.object_id += 1

        return updated

def detect_from_background_image(cap: cv2.VideoCapture, background_path="background.jpg", show=True) -> list[dict]:
    detected_objects = []
    tracker = ObjectTracker()
    counts = {
        "osobowy_lewo_prawo": 0,
        "osobowy_prawo_lewo": 0,
        "ciezarowy_lewo_prawo": 0,
        "ciezarowy_prawo_lewo": 0
    }

    background = cv2.imread(background_path)
    if background is None:
        print("Nie można załadować background.jpg")
        return []

    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    if show:
        cv2.namedWindow("Detekcja zmian", cv2.WINDOW_NORMAL)

    # Zdefiniuj strefy
    strefa_lewo = ((125, 195), (682, 342))
    strefa_prawo = ((127, 385), (963, 611))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.rectangle(frame, strefa_lewo[0], strefa_lewo[1], (255, 0, 0), 2)
        cv2.rectangle(frame, strefa_prawo[0], strefa_prawo[1], (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(background_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=9)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_objects = []
        centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

            if h < 50:
                continue
            #print(f"x:{x}, y:{y}, w:{w}, h:{h}")
            centroids.append((cx, cy))

            aspect_ratio = max(w, h) / min(w, h)
            if cy < frame.shape[0] * 0.6:
                strefa = "jezdnia"
            else:
                strefa = "chodnik"

            if strefa == "jezdnia":
                if area > 40000 and w > 150:
                    label = "ciezarowy/autobus"
                    color = (0, 165, 255)
                else:
                    label = "osobowy"
                    color = (0, 255, 0)
            else:
                label = "pieszy/rowerzysta"
                color = (0, 0, 255)

            frame_objects.append({
                "label": label,
                "bbox": (x, y, w, h),
                "area": area,
                "centroid": (cx, cy)
            })

            if show:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        tracked = tracker.update(centroids, frame_objects, frame_hsv)
        for obj_id, cx, cy in tracked:
            obj_data = tracker.tracked[obj_id]
            if obj_id in tracker.counted_ids:
                continue

            if strefa_lewo[0][0] <= cx <= strefa_lewo[1][0] and strefa_lewo[0][1] <= cy <= strefa_lewo[1][1]:
                direction = "prawo_lewo"
            elif strefa_prawo[0][0] <= cx <= strefa_prawo[1][0] and strefa_prawo[0][1] <= cy <= strefa_prawo[1][1]:
                direction = "lewo_prawo"
            else:
                continue

            label = next((o["label"] for o in frame_objects if o["centroid"] == (cx, cy)), None)
            if label == "osobowy":
                counts[f"osobowy_{direction}"] += 1
            elif label == "ciezarowy/autobus":
                counts[f"ciezarowy_{direction}"] += 1
            tracker.counted_ids.add(obj_id)

            if show:
                arrow = "→" if direction == "lewo_prawo" else "←"
                cv2.putText(frame, arrow, (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            print(f"[LICZNIK] Osobowe: ← {counts['osobowy_prawo_lewo']} | → {counts['osobowy_lewo_prawo']} | "
                  f"Ciężarowe: ← {counts['ciezarowy_prawo_lewo']} | → {counts['ciezarowy_lewo_prawo']}")

        if show:
            cv2.imshow("Detekcja zmian", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        detected_objects.append(frame_objects)

    cap.release()
    if show:
        cv2.destroyAllWindows()

    return detected_objects
