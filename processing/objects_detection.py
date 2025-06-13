import cv2
from processing.tracking import ObjectTracker

def iou_with_tram_zone(bbox, tram_zone):
    """
    Funkcja sprawdza w jakim stopniu obiekt pokrywa się z tramwajową strefą detekcji
    """
    x1, y1, w1, h1 = bbox
    x2, y2 = x1 + w1, y1 + h1  # Obliczenie współrzędnych prawego dolnego rogu obiektu.
    tx1, ty1 = tram_zone[0]    # Rozpakowanie współrzędnych lewego górnego i prawego dolnego rogu strefy tramwaju.
    tx2, ty2 = tram_zone[1]

    # Wyznaczenie współrzędnych przecięcia prostokątów (strefy i obiektu)
    ix1 = max(x1, tx1)
    iy1 = max(y1, ty1)
    ix2 = min(x2, tx2)
    iy2 = min(y2, ty2)

    # Obliczenie obszaru części wspólnej (intersekcji)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih

    # Obszar strefy tramwaju
    tram_area = (tx2 - tx1) * (ty2 - ty1)

    # Zwraca procent pokrycia obiektu przez strefę tramwaju
    return inter_area / float(tram_area) if tram_area > 0 else 0.0


def detection(cap: cv2.VideoCapture, background_path="background.jpg", show=True, debug=False) -> dict:
    """
    Główna funkcja detekcji i zliczania obiektów pojawiających się na kolejnych kaltkach przetwarzanego wideo
    """

    # Inicjalizacja trackera i zmiennych pomocniczych
    tracker = ObjectTracker()
    tramwaj_block_until = -1  # blokada zliczania tramwajów (dla uniknięcia wielokrotnego zliczania)

    # Słownik wynikowy – ilość zliczonych obiektów każdej klasy
    counts = {
        "osobowy_lewo_prawo": 0,
        "osobowy_prawo_lewo": 0,
        "ciezarowy_lewo_prawo": 0,
        "ciezarowy_prawo_lewo": 0,
        "tramwaj": 0,
        "pieszy": 0,
        "rowerzysta": 0
    }

    # Wczytanie obrazu tła
    background = cv2.imread(background_path)
    if background is None:
        print("Nie można załadować background.jpg")
        return []

    # Zamiana obrazu tła na obraz w skali szarości
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Okno podglądu procesu detekcji i zawartości pliku wideo
    if show:
        cv2.namedWindow("Video frames with detection", cv2.WINDOW_NORMAL)

    # Zdefiniowanie stref analizy, okien w których zachodzi zliczanie obiektów

    ## Strefa jezdni
    strefa_lewo = ((300, 110), (600, 342))
    strefa_prawo = ((500, 355), (1000, 611))

    ## Strefa torów tramwajowych
    strefa_tramwaju1 = ((0, 225), (280, 390))
    strefa_tramwaju2 = ((620, 225), (900, 390))

    ## Strefa pieszych, rowerzystów
    strefa_piesi = ((900, 739), (1100, 1020))

    # Główna pętla przetwarzania wideo
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rysowanie okien w których zachodzi zliczanie obiektów
        cv2.rectangle(frame, strefa_lewo[0], strefa_lewo[1], (255, 0, 0), 2)
        cv2.rectangle(frame, strefa_prawo[0], strefa_prawo[1], (0, 255, 0), 2)
        
        cv2.rectangle(frame, strefa_tramwaju1[0], strefa_tramwaju1[1], (0, 0, 255), 2)
        cv2.rectangle(frame, strefa_tramwaju2[0], strefa_tramwaju2[1], (0, 0, 255), 2)

        cv2.rectangle(frame, strefa_piesi[0], strefa_piesi[1], (180, 75, 120), 2)

        # Rysowanie linii oddzielającej jezdnie od ścieżki pieszej, rowerowej
        cv2.line(frame, (0, int(frame.shape[0] * 0.6)), (frame.shape[1], int(frame.shape[0] * 0.6)), (0,255,255), 2)
        

        # Wykrywanie różnic na podstawie modelu tła (foreground mask)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)              # Zamiana pobranej klatki wideo na obraz w skali szarości
        diff = cv2.absdiff(background_gray, gray)                   # Obraz różnicowy
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) # Progowanie binarne

        # Czyszczenie binarnej maski operacjami morfologicznymi
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # Operacja ma usunąć szumy i zakłócenia (migotanie pojedynczych białych pikseli)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=7) # Operacja ma wypełnić czarne dziury wewnątrz białych obszarów (np. wewnątrz pojazdów)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      # Wydobywa kształty obiektów (kontury)
                                                                                                # cv2.RETR_EXTERNAL - uwzględnia tylko zewnętrzne kontury, ignorując np. dziury w obiektach
                                                                                                # cv2.CHAIN_APPROX_SIMPLE - upraszcza kontury, zapisując tylko kluczowe punkty

        frame_objects = []  # Lista wykrytych obiektów w tej klatce (z etykietą, pozycją, rozmiarem itd.)
                            # Lista wykorzystywana potem do: śledzenia obiektów przez tracker.update(...), zliczania obiektów


        centroids = []  # Lista środków geometrycznych (cx, cy) obiektów w danej klatce
                        # Wykorzystywana jako uproszczony zestaw danych wejściowych do trackera


        # Iterowanie po wykrytych konturach - "obiektach"
        for contour in contours:
            area = cv2.contourArea(contour) # Wyliczenie pola powierzchni wykrytego obiektu
            if area < 3000:                 # Ignorowanie małych szumów, pod względem pola powierzchni
                continue

            x, y, w, h = cv2.boundingRect(contour)    # x, y - współrzędne lewego górnego rogu prostokąta, w, h - szerokość i wysokość prostokąta.
            cx = x + w // 2                           # cx, cy - współrzędne środka obiektu
            cy = y + h // 2
            bottom_y = y + h                          # Oblicza współrzędną dolnej krawędzi prostokąta, przydatną np. do sprawdzenia, na jakiej wysokości kończy się obiekt (czy znajduje się na jezdni, torach itp.).

            if h < 50: # Ignorowanie małych szumów, pod względem wysokości
                continue

            centroids.append((cx, cy))

            #aspect_ratio = max(w, h) / min(w, h)

            label = None     # Etykieta klasy obiektu (np. osobowy, pieszy, tramwaj)
            strefa = None    # Strefa, w której znajduje się obiekt (jezdnia, tory, chodnik)
            

            #===Identyfikacja obiektu===

            # Określenie strefy i klasy obiektu

            if cy < frame.shape[0] * 0.6:  # Obiekt znajduje się w górnej części obrazu
                
                # Sprawdzenie, czy obiekt znajduje się na torach tramwajowych
                if iou_with_tram_zone((x, y, w, h), strefa_tramwaju1) > 0.99 or iou_with_tram_zone((x, y, w, h), strefa_tramwaju2) > 0.99:
                    strefa = "tory"

                # Sprawdzenie położenia dolnej krawędzi - czy w zakresie jezdni
                elif ((110 < bottom_y < 342) or (447 < bottom_y < 711)):
                    strefa = "jezdnia"
            else:
                # Obiekt znajduje się w dolnej części obrazu 
                strefa = "chodnik"


            if strefa == "jezdnia":
                if area > 60000 and w > 550:                    
                    label = "ciezarowy/autobus"
                    color = (0, 165, 255)
                else:
                    label = "osobowy"
                    color = (0, 255, 0)
            
            elif strefa == "tory":
                if h > 200 and w > 800:
                    label = "tramwaj"
                    color = (0, 255, 255)

            elif strefa == "chodnik":
                    label = "pieszy"
                    color = (0, 0, 255)

            if label is None:
                continue # Obiekt nie został zaklasyfikowany - pominięcie

            # Print debugowy
            #print(f"Label: {label}, bbox: {(x, y, w, h)}, bottom_y: {bottom_y}, area:{area}")
            
            # Dodanie opisu obiektu do listy
            frame_objects.append({
                "label": label,
                "bbox": (x, y, w, h),
                "area": area,
                "centroid": (cx, cy)
            })

            if show:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)                            # Wizualizacja bboxów wykrytych obiektów
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        # Aktualizacja śledzenia obiektów
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))                        # Pobranie numeru aktualnej klatki
        tracked = tracker.update(centroids, frame_objects, frame, frame_number)
                                                                                    # Dla każdej klatki wywoływany jest tracker.update(...), aby:
                                                                                    # przypisać ID obiektom,
                                                                                    # powiązać je z obiektami z poprzednich klatek,
                                                                                    # utworzyć nowe ID dla nowych obiektów.


        # Iteracja po aktualnie śledzonych obiektach, z ich ID oraz pozycją (środkiem geometrycznym).
        for obj_id, cx, cy in tracked:
            if obj_id in tracker.counted_ids:
                continue # obiekt już wcześniej zliczony

            obj_data = tracker.tracked[obj_id]

            # Odtworzenie etykiety (label) dla danego obiektu, szukając jej po jego (cx, cy) w danych detekcji
            label = next((o["label"] for o in frame_objects if o["centroid"] == (cx, cy)), None)  

            # Zliczanie tramwajów z blokadą czasową
            if label == "tramwaj":
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_number < tramwaj_block_until:
                    continue  # zablokowane zliczanie tramwajów

                counts["tramwaj"] += 1
                tracker.counted_ids.add(obj_id)
                tramwaj_block_until = frame_number + 125
                #print(f"[TRAMWAJ] Zliczono tramwaj. Kolejny możliwy po klatce {tramwaj_block_until}")
                continue
            
            # Zliczanie pieszych 
            if label == "pieszy":
                in_zone = (
                    strefa_piesi[0][0] <= cx <= strefa_piesi[1][0]
                    and strefa_piesi[0][1] <= cy <= strefa_piesi[1][1]
                )
                obj_data['in_pedestrian_zone'] = in_zone

                if obj_id in tracker.counted_ids and in_zone:
                    continue  # pieszy już zliczony i nadal w strefie

                if obj_id not in tracker.counted_ids and in_zone:
                    counts["pieszy"] += 1
                    tracker.counted_ids.add(obj_id)
                    #print(f"[PIESZY] Zliczony. ID={obj_id}, centroid={cx, cy}")
                    continue

            # Detekcja kierunku jazdy pojazdów na jezdni – z lewej do prawej lub odwrotnie
            if strefa_lewo[0][0] <= cx <= strefa_lewo[1][0] and strefa_lewo[0][1] <= cy <= strefa_lewo[1][1]:
                direction = "prawo_lewo"
            elif strefa_prawo[0][0] <= cx <= strefa_prawo[1][0] and strefa_prawo[0][1] <= cy <= strefa_prawo[1][1]:
                direction = "lewo_prawo"
            else:
                continue
            
            # Zliczanie pojazdów według typu i kierunku
            if label == "osobowy":
                counts[f"osobowy_{direction}"] += 1
            elif label == "ciezarowy/autobus":
                counts[f"ciezarowy_{direction}"] += 1

            # Oznaczamy obiekt jako zliczony, aby nie był brany pod uwagę w kolejnych klatkach
            tracker.counted_ids.add(obj_id)
            
            if debug:
                print(f"[LICZNIK] Osobowe: ← {counts['osobowy_prawo_lewo']} | → {counts['osobowy_lewo_prawo']} | "
                    f"Ciężarowe: ← {counts['ciezarowy_prawo_lewo']} | → {counts['ciezarowy_lewo_prawo']} || "
                    f"Tramwaje: {counts['tramwaj']} || Piesi: {counts['pieszy']}")

        
        # Podgląd na żywo procesu detekcji, zliczania
        if show:
            #cv2.imshow("Thresholding", thresh)
            cv2.imshow("Video frames with detection", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break


    cap.release()
    if show:
        cv2.destroyAllWindows()

    return counts
