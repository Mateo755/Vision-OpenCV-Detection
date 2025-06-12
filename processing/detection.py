import cv2
import numpy as np
from scipy.spatial.distance import euclidean

class ObjectTracker:
    def __init__(self):
        self.tracked = {}
        self.counted_ids = set()
        self.object_id = 0
        self.prev_tracked = {}

    def compute_hog_descriptor(self, image):
        winSize = (64, 64)
        blockSize = (8, 8)
        blockStride = (4, 4)
        cellSize = (4, 4)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        resized = cv2.resize(image, winSize)
        descriptor = hog.compute(resized)
        return descriptor

    def find_matching_prev_id(self, cx, cy, roi):
        descriptor = self.compute_hog_descriptor(roi)
        best_id = None
        best_score = float('inf')

        for obj_id, data in self.prev_tracked.items():
            prev_cx, prev_cy = data['centroid']
            dist = np.hypot(cx - prev_cx, cy - prev_cy)
            if dist > 50:
                continue
            score = euclidean(data['descriptor'], descriptor)
            if score < best_score and score < 0.4:
                best_score = score
                best_id = obj_id

        return best_id, descriptor

    def update(self, current_centroids, frame_objects, frame_rgb):
        updated = []
        for cx, cy in current_centroids:
            matched_id = None
            best_score = float('inf')

            for obj_id, data in self.tracked.items():
                prev_cx, prev_cy = data['centroid']
                dist = np.hypot(cx - prev_cx, cy - prev_cy)
                if dist > 60:
                    continue

                candidate = next((o for o in frame_objects if o['centroid'] == (cx, cy)), None)
                if candidate is None:
                    continue

                x, y, w, h = candidate['bbox']
                roi = frame_rgb[y:y+h, x:x+w]
                descriptor = self.compute_hog_descriptor(roi)
                score = euclidean(data['descriptor'], descriptor)
                if score < best_score:
                    best_score = score
                    matched_id = obj_id

            if matched_id is not None:
                self.tracked[matched_id]['centroid'] = (cx, cy)
                self.tracked[matched_id]['descriptor'] = descriptor
                self.tracked[matched_id]['bbox'] = candidate['bbox']
                updated.append((matched_id, cx, cy))
            else:
                candidate = next((o for o in frame_objects if o['centroid'] == (cx, cy)), None)
                if candidate is None:
                    continue
                x, y, w, h = candidate['bbox']
                roi = frame_rgb[y:y+h, x:x+w]

                match_id, descriptor = self.find_matching_prev_id(cx, cy, roi)
                if match_id is not None and match_id not in self.tracked:
                    assigned_id = match_id
                else:
                    assigned_id = self.object_id
                    self.object_id += 1

                self.tracked[assigned_id] = {
                    'centroid': (cx, cy),
                    'descriptor': descriptor,
                    'bbox': (x, y, w, h),
                    'counted': False
                }
                updated.append((assigned_id, cx, cy))

        self.prev_tracked = {
            obj_id: {
                'centroid': data['centroid'],
                'descriptor': data['descriptor']
            } for obj_id, data in self.tracked.items()
        }

        return updated

    def is_stable(self, obj_id):
        return True
    
    def iou_with_tram_zone(self, bbox, tram_zone):
        x1, y1, w1, h1 = bbox
        x2, y2 = x1 + w1, y1 + h1
        tx1, ty1 = tram_zone[0]
        tx2, ty2 = tram_zone[1]

        # Obszar przecięcia
        ix1 = max(x1, tx1)
        iy1 = max(y1, ty1)
        ix2 = min(x2, tx2)
        iy2 = min(y2, ty2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter_area = iw * ih

        # Obszar strefy tramwaju
        tram_area = (tx2 - tx1) * (ty2 - ty1)

        return inter_area / float(tram_area) if tram_area > 0 else 0.0




def detect_from_background_image(cap: cv2.VideoCapture, background_path="background.jpg", show=True) -> list[dict]:
    detected_objects = []
    tracker = ObjectTracker()
    counts = {
        "osobowy_lewo_prawo": 0,
        "osobowy_prawo_lewo": 0,
        "ciezarowy_lewo_prawo": 0,
        "ciezarowy_prawo_lewo": 0,
        "tramwaj": 0,
        "pieszy": 0
    }

    background = cv2.imread(background_path)
    if background is None:
        print("Nie można załadować background.jpg")
        return []

    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    if show:
        cv2.namedWindow("Detekcja zmian", cv2.WINDOW_NORMAL)

    strefa_lewo = ((300, 110), (600, 342))
    strefa_prawo = ((500, 355), (1000, 611))
    strefa_tramwaju1 = ((0, 220), (280, 390))
    strefa_tramwaju2 = ((620, 220), (900, 390))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.rectangle(frame, strefa_lewo[0], strefa_lewo[1], (255, 0, 0), 2)
        cv2.rectangle(frame, strefa_prawo[0], strefa_prawo[1], (0, 255, 0), 2)
        cv2.rectangle(frame, strefa_tramwaju1[0], strefa_tramwaju1[1], (0, 0, 255), 2)
        cv2.rectangle(frame, strefa_tramwaju2[0], strefa_tramwaju2[1], (0, 0, 255), 2)
        cv2.line(frame, (0, int(frame.shape[0] * 0.6)), (frame.shape[1], int(frame.shape[0] * 0.6)), (0,255,255), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(background_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=7)

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
            bottom_y = y + h

            

            if h < 50:
                continue

            centroids.append((cx, cy))

            aspect_ratio = max(w, h) / min(w, h)

            label = None
            strefa = None 

            if cy < frame.shape[0] * 0.6:
                if tracker.iou_with_tram_zone((x, y, w, h), strefa_tramwaju1) > 0.99 or tracker.iou_with_tram_zone((x, y, w, h), strefa_tramwaju2) > 0.99:
                    strefa = "tory"
                elif ((110 < bottom_y < 342) or (447 < bottom_y < 711)):
                    strefa = "jezdnia"
            else:
                strefa = "chodnik"


            if strefa == "jezdnia":
                if area > 60000 and w > 550:
                    label = "ciezarowy/autobus"
                    color = (0, 165, 255)
                else:
                    label = "osobowy"
                    color = (0, 255, 0)
            elif strefa == "tory" and h > 200 and w > 1050 and area < 150000:
                print(f"Label: {label}, bbox: {(x, y, w, h)}, bottom_y: {bottom_y}, area:{area}")
                label = "tramwaj"
                color = (0, 255, 255)

            elif strefa == "chodnik":
                if w < 100:
                    label = "pieszy"
                    color = (0, 0, 255)

            if label is None:
                continue

            #print(f"Label: {label}, bbox: {(x, y, w, h)}, bottom_y: {bottom_y}, area:{area}")

            frame_objects.append({
                "label": label,
                "bbox": (x, y, w, h),
                "area": area,
                "centroid": (cx, cy)
            })

            if show:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        tracked = tracker.update(centroids, frame_objects, frame)
        for obj_id, cx, cy in tracked:
            if obj_id in tracker.counted_ids:
                continue

            obj_data = tracker.tracked[obj_id]
            if not tracker.is_stable(obj_id):
                continue

            label = next((o["label"] for o in frame_objects if o["centroid"] == (cx, cy)), None)

            if label == "pieszy":
                counts["pieszy"] += 1
                tracker.counted_ids.add(obj_id)
                print(f"[PIESZY] Pieszy wykryty! Liczba: {counts['pieszy']}")
                continue
            if label == "tramwaj":
                counts["tramwaj"] += 1
                tracker.counted_ids.add(obj_id)
                print(f"[TRAMWAJ] Tramwaj wykryty! Liczba: {counts['tramwaj']}")
      
                continue

            if strefa_lewo[0][0] <= cx <= strefa_lewo[1][0] and strefa_lewo[0][1] <= cy <= strefa_lewo[1][1]:
                direction = "prawo_lewo"
            elif strefa_prawo[0][0] <= cx <= strefa_prawo[1][0] and strefa_prawo[0][1] <= cy <= strefa_prawo[1][1]:
                direction = "lewo_prawo"
            else:
                continue

            if label == "osobowy":
                counts[f"osobowy_{direction}"] += 1
            elif label == "ciezarowy/autobus":
                counts[f"ciezarowy_{direction}"] += 1

            tracker.counted_ids.add(obj_id)
            print(f"[LICZNIK] Osobowe: ← {counts['osobowy_prawo_lewo']} | → {counts['osobowy_lewo_prawo']} | "
                  f"Ciężarowe: ← {counts['ciezarowy_prawo_lewo']} | → {counts['ciezarowy_lewo_prawo']} || Tramwaje: {counts['tramwaj']} || Piesi: {counts['pieszy']}")

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
