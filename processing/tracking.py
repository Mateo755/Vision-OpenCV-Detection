import cv2
import numpy as np
from scipy.spatial.distance import euclidean

class ObjectTracker:
    """
    Główna klasa do śledzenia obiektów
    """

    def __init__(self, fps=30, max_seconds_missing=15):
        """
        Inicjalizacja parametrów obiektu klasy
        """
        self.tracked = {}                                           # Aktualnie śledzone obiekty: obj_id -> dane
        self.counted_ids = set()                                    # Zbiór ID obiektów już zliczonych
        self.object_id = 0                                          # Licznik do przypisywania nowych ID
        self.prev_tracked = {}                                      # Dane z poprzedniej klatki (do porównania)
        self.max_missed_frames = int(fps * max_seconds_missing)     # Maksymalna liczba kolejnych klatek, w których obiekt może być niewidoczny
                                                                    # zanim zostanie usunięty ze śledzenia
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



    def update(self, current_centroids, frame_objects, frame_rgb, frame_number):
        updated = []

        # Iteracja po wszystkich centroidach wykrytych obiektów
        for cx, cy in current_centroids:

            # Znajdź obiekt w frame_objects odpowiadający aktualnemu centroidowi
            candidate = next((obj for obj in frame_objects if obj['centroid'] == (cx, cy)), None)
            if candidate is None:
                continue    # pomiń, jeśli nie znaleziono obiektu

            x, y, w, h = candidate['bbox']
            roi = frame_rgb[y:y+h, x:x+w]
            descriptor = self.compute_hog_descriptor(roi)       # Oblicz deskryptor HOG dla bieżącego obiektu

            matched_id = None
            best_score = float('inf')                           # Najniższa znaleziona odległość między deskryptorami

            for obj_id, data in self.tracked.items():           # Przeszukaj aktualnie śledzone obiekty
                prev_cx, prev_cy = data['centroid']
                dist = np.hypot(cx - prev_cx, cy - prev_cy)
                if dist > 60:                                   # Jeśli nowy obiekt jest zbyt daleko od wcześniej śledzonego, następuje pominięcie
                    continue                                    
                
                # Liczymy odległość euklidesową pomiędzy opisem obecnego obiektu a opisem wcześniej śledzonego
                score = euclidean(data['descriptor'], descriptor)
                if score < best_score:      # Im mniejsza ta odległość, tym bardziej podobny wygląd                
                    best_score = score
                    matched_id = obj_id

            if matched_id is not None:
                # Dopasowano do istniejącego obiektu – aktualizuj dane
                self.tracked[matched_id]['centroid'] = (cx, cy)
                self.tracked[matched_id]['descriptor'] = descriptor
                self.tracked[matched_id]['bbox'] = candidate['bbox']
                self.tracked[matched_id]['missed_frames'] = 0  # resetuj licznik
                updated.append((matched_id, cx, cy))
            
            else: # Nie pasuje do żadnego aktualnie śledzonego obiektu

                assigned_id = self.object_id
                self.object_id += 1     # Nowy obiekt

                self.tracked[assigned_id] = {
                    'centroid': (cx, cy),               # Środek obiektu (x, y) — używane do śledzenia i kierunku
                    'descriptor': descriptor,           # Deskryptor HOG — cecha wyglądu (do porównania podobieństwa)
                    'bbox': (x, y, w, h),               # Bounding box — prostokąt ograniczający obiekt
                    'first_seen_frame': frame_number,   # Numer klatki, w której obiekt został wykryty po raz pierwszy
                    'in_pedestrian_zone': False,        # Czy obiekt znajduje się w strefie pieszych
                    'missed_frames': 0                  # Licznik kolejnych klatek, w których obiekt nie był widoczny
                }

                updated.append((assigned_id, cx, cy))

        
        # Usuwanie obiektów, które zniknęły na zbyt długo
        
        # Zbiera ID wszystkich obiektów, które zostały zaktualizowane (czyli wykryte w tej klatce)
        updated_ids = set(obj_id for obj_id, _, _ in updated) 	   
        
        # Lista ID do późniejszego usunięcia
        to_remove = []

        for obj_id, data in self.tracked.items():
            if obj_id in updated_ids:                # Jeśli obiekt był widoczny, pomiń
                continue

            data['missed_frames'] += 1               # Jeśli obiekt nie był widoczny, zwiększ licznik „nieobecności w kadrze"
            
            # Jeśli nie był widziany zbyt długo -> dodaj go do to_remove
            if data['missed_frames'] > self.max_missed_frames:
                to_remove.append(obj_id)             # Za długo niewidoczny — oznacz do usunięcia

        for obj_id in to_remove:
            del self.tracked[obj_id]                 # Każdy obiekt, który zniknie z obrazu i nie pojawi się przez 15 sekund, zostaje usunięty 

        return updated




