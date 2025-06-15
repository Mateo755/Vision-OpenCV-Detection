# Detekcja i Zliczanie Obiektów w Sekwencjach Wideo

## Opis projektu

Projekt realizuje zadanie automatycznej detekcji i zliczania różnych klas obiektów na drogach miejskich na podstawie nagrań wideo. Program identyfikuje i klasyfikuje obiekty takie jak:

- samochody osobowe (jadące z lewej i prawej),
- ciężarówki i autobusy (jadące z lewej i prawej),
- tramwaje,
- piesi,
- rowerzyści.

Wynikiem działania programu jest plik `.json`, zawierający liczbę obiektów dla każdej sekwencji wideo.

---

## Struktura katalogów

```
VISION-OPENCV-DETECTION/
│
├── main.py                         # Główny plik uruchomieniowy projektu
├── requirements.txt                # Lista zależności Pythona
├── README.md                       # Ten plik
├── raport.md                       # Opis projektu
├── background.jpg                  # Obraz referencyjny tła do detekcji zmian
│
├── extra_testing_utils/            # Skrypty używane podczas tworzenia projektu
│   ├── create_zone.py
│   ├── save_background.py
│   └── video_cutter.py
│
├── processing/                     # Główna logika detekcji i śledzenia
│   ├── objects_detection.py        # Detekcja obiektów na podstawie różnic z tłem
│   ├── tracking.py                 # Śledzenie i identyfikacja obiektów
│   └── utils.py                    # Skrypt który uruchamia detekcje na wskazanym wycinku wideo
│
├── results/                        # Pliki ground truth oraz wyniki programu
│   ├── Grand_Truth.txt
│   └── results.json
│
├── videos/                         # Zestaw testowych nagrań wideo
│   ├── 00000.mp4
│   ├── wycinek_2trams.mp4
│   └── ...
└── Opis_projektu_2024_2025.pdf     # Specyfikacja projektu
```

## Konfiguracja projektu

1. Sklonuj repozytorium:

```bash
git clone https://github.com/Mateo755/Vision-OpenCV-Detection.git
cd VISION-OPENCV-DETECTION
```

2. Utwórz i aktywuj wirtualne środowisko (opcjonalnie, ale zalecane):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate   # Windows
```

3. Zainstaluj wymagane biblioteki:

```bash
pip install -r requirements.txt
```

4. Przygotuj pliki wideo i `background.jpg` w odpowiednich folderach.

5. Uruchom skrypt `main.py` z odpowiednimi ścieżkami.

---

## Uruchomienie

Skrypt przyjmuje **dwa argumenty**:
1. Ścieżka do katalogu z nagraniami `.mp4`
2. Ścieżka do pliku wyjściowego `.json`, gdzie zapisane zostaną wyniki

### Przykład:

```bash
python main.py ./videos ./results/results.json
```

---

## Format wynikowego pliku JSON

```json
{
  "wycinek_wideo.mp4": {
    "liczba_samochodow_osobowych_z_prawej_na_lewa": 3,
    "liczba_samochodow_osobowych_z_lewej_na_prawa": 5,
    "liczba_samochodow_ciezarowych_autobusow_z_prawej_na_lewa": 2,
    "liczba_samochodow_ciezarowych_autobusow_z_lewej_na_prawa": 1,
    "liczba_tramwajow": 2,
    "liczba_pieszych": 4,
    "liczba_rowerzystow": 1
  }
}
```

---

## Zasada działania

1. Detekcja obiektów przez różnicowanie z tłem (`background.jpg`)
2. Klasyfikacja obiektów na podstawie położenia i wymiarów
3. Śledzenie obiektów przy użyciu HOG i metryki euklidesowej
4. Zliczanie obiektów w zdefiniowanych strefach (jezdnia, tory, chodnik)

---


## Dodatkowe informacje

- Do działania wymagany jest plik `background.jpg` (tło sceny)
- Każde uruchomienie generuje pełne wyniki dla wszystkich filmów z katalogu