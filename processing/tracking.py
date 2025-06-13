import cv2
import numpy as np
from scipy.spatial.distance import euclidean

class ObjectTracker:
    """
    Główna klasa do śledzenia obiektów
    """

    def __init__(self):
        """
        Inicjalizacja parametrów obiektu klasy
        """
        self.tracked = {}           # Aktualnie śledzone obiekty: obj_id -> dane
        self.counted_ids = set()    # Zbiór ID obiektów już zliczonych
        self.object_id = 0          # Licznik do przypisywania nowych ID
        self.prev_tracked = {}      # Dane z poprzedniej klatki (do porównania)

    def compute_hog_descriptor(self, image):
        """
        Oblicza deskryptor HOG (Histogram of Oriented Gradients) z podanego fragmentu obrazu
        """
        winSize = (64, 64)    # Rozmiar okna detekcji HOG – obraz wejściowy (ROI) zostanie przeskalowany do tego rozmiaru
        blockSize = (8, 8)    # Rozmiar bloku, czyli obszaru, z którego będą normalizowane gradienty
        blockStride = (4, 4)  # Krok przesuwania bloku po obrazie  
        cellSize = (4, 4)     # Rozmiar pojedynczej komórki, w której liczony jest histogram gradientów
        nbins = 9             # Liczba przedziałów histogramu kierunku gradientu, 9 oznacza kierunki co 20 stopni
        
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        resized = cv2.resize(image, winSize)
        descriptor = hog.compute(resized)  # Zwraca wektor cech opisujący teksturę i orientację gradientów.
        return descriptor

    def find_matching_prev_id(self, cx, cy, roi):
        """
        Znalezienie najlepszego dopasowania pomiędzy obiektem z bieżącej klatki a obiektami z poprzedniej klatki
        """
        descriptor = self.compute_hog_descriptor(roi)           # Oblicz deskryptor HOG dla bieżącego obiektu
        best_id = None                                          # ID najlepiej pasującego obiektu z poprzedniej klatki
        best_score = float('inf')                               # Najniższa znaleziona odległość między deskryptorami

        for obj_id, data in self.prev_tracked.items():          # Iteracja po wszystkich obiektach z poprzedniej klatki
            prev_cx, prev_cy = data['centroid']                 # Środek poprzedniego obieku
            
            dist = np.hypot(cx - prev_cx, cy - prev_cy)         # Odległość euklidesowa między centroidami
            
            if dist > 50:           # Odfiltrowuje oczywiste nietrafione kandydaty (zbyt daleko),
                continue            # Wstępna filtracja, zbyt daleko – pomiń dopasowanie

            score = euclidean(data['descriptor'], descriptor)  # Pozwala dokładnie dobrać najlepiej pasujący obiekt po wyglądzie.  
                                                               # Każdy obiekt jest reprezentowany przez wektor HOG, wektor liczb opisujących orientację krawędzi i teksturę
                                                               # Im bardziej podobne są te wektory, tym mniejsza jest ich odległość euklidesowa
            # Warunek oznacza:
            # 1) Czy znalezione jest najlepsze dopasowanie ?
            # 2) Czy dopasowanie jest wystarczająco dobre ? 
            if score < best_score and score < 0.4:     # 0.4 to próg akceptacji podobieństwa wyglądu. Im mniejsza ta wartość: tym bardziej rygorystyczne porównanie
                best_score = score
                best_id = obj_id

        return best_id, descriptor
    

    def update(self, current_centroids, frame_objects, frame_rgb, frame_number):      
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
                    'counted': False,
                    'first_seen_frame': frame_number,
                    'confirmed': False,
                    'in_pedestrian_zone': False 
                }
                updated.append((assigned_id, cx, cy))

        self.prev_tracked = {
            obj_id: {
                'centroid': data['centroid'],
                'descriptor': data['descriptor']
            } for obj_id, data in self.tracked.items()
        }

        return updated