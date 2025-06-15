# Opis zastosowanych metod w celu zrealizowania detekcji obiektów

## Wstępne przetwarzanie danych

W projekcie zastosowano metodę różnicową (model tła), opartą na porównywaniu każdej klatki wideo z wcześniej wygenerowaną klatką tła, przedstawiającą scenę bez żadnych ruchomych obiektów . Przetwarzanie obejmuje:

* Konwersję obrazu do skali szarości.
* Wyznaczenie różnicy pomiędzy bieżącą klatką a obrazem tła (`cv2.absdiff`).
* Progowanie binarne uzyskanego obrazu różnicowego (`cv2.threshold`), generując maskę pierwszoplanową.

Ze względu na perspektywę kamery ustawionej od boku, niezwykle trudno jest rozdzielić nakładające się obiekty, zwłaszcza wykorzystując wyłącznie cechy obrazowe, co dodatkowo komplikuje proces detekcji. Zauważalne jest także, że obiekty znajdujące się bliżej kamery mają większe wymiary, co wpływa na identyfikację i śledzenie obiektów.

## Morfologia obrazowa

Aby usunąć drobne szumy oraz poprawić jakość masek binarnych, zastosowano następujące operacje morfologiczne, które odgrywały kluczową rolę w konstruowaniu konturów:

* **Otwarcie morfologiczne** (`cv2.morphologyEx` z parametrem `cv2.MORPH_OPEN`), które skutecznie redukowało szumy, jednak zbyt silne ustawienia degradują kontury istotnych obiektów. Konieczne było staranne dobranie liczby iteracji, by znaleźć optymalną granicę między usuwaniem szumów a zachowaniem detali obiektów.
* **Zamknięcie morfologiczne** (`cv2.morphologyEx` z parametrem `cv2.MORPH_CLOSE`), które wypełnia niewielkie dziury wewnątrz wykrytych obiektów, tworząc spójniejsze kontury. Również wymagało ono dużej liczby eksperymentów w celu ustalenia właściwej liczby iteracji.

## Generowanie konturów

Kontury generowane są metodą `cv2.findContours` z ustawieniami `cv2.RETR_EXTERNAL` i `cv2.CHAIN_APPROX_SIMPLE`, co pozwala identyfikować obiekty o istotnej powierzchni, ignorując małe szumy.

## Umiejscowienie i przeznaczenie stref detekcji

Strefy detekcji zostały precyzyjnie rozmieszczone, aby minimalizować wpływ przeszkód takich jak drzewa czy wysokie słupki, które negatywnie wpływają na jakość generowanych konturów. Detekcja odbywa się miejscowo, wyłącznie w specjalnie wydzielonych obszarach, co znacząco poprawia precyzję detekcji w warunkach utrudnionej widoczności.

Poszczególne strefy zostały zdefiniowane w następujący sposób:

* **Żółta pozioma linia** – oddziela strefę pieszych od innych stref.
* **Fioletowy bbox** – przeznaczony do miejscowej detekcji pieszych.
* **Czerwone bboxy** – przeznaczone są do detekcji tramwajów. Tramwaje mają podwójną strefę, ponieważ gdy tramwaj jadący w prawo wjedzie w pierwszą strefę przy lewej krawędzi obrazu, oznacza to obecność obiektu na torach. W takim przypadku detekcja w niebieskiej strefie (przeznaczonej dla samochodów jadących z prawej na lewo) jest wyłączona dla tego obiektu, gdyż tramwaj wizualnie wchodzi również w tę strefę. Analogiczne rozwiązanie zastosowano dla tramwajów jadących w lewo.
* **Niebieski bbox** – do detekcji samochodów poruszających się z prawej na lewo.
* **Zielony bbox** – do detekcji samochodów poruszających się z lewej na prawo.

Takie umiejscowienie i przeznaczenie stref znacząco poprawia wiarygodność detekcji poprzez unikanie obszarów wizualnie problematycznych.

## Klasyfikacja obiektów

Klasyfikacja obiektów odbywa się poprzez analizę centroidów wycinków wykrytych obiektów i porównanie ich pozycji z odpowiednio zdefiniowanymi strefami detekcji. Proces ten przebiega według następujących kroków:

1. **Wyznaczenie pozycji centroidu obiektu:**

   * Sprawdzana jest wartość współrzędnej `cy` centroidu, aby określić czy obiekt znajduje się w górnej czy dolnej części obrazu:

     * Górna część: `cy < frame.shape[0] * 0.6`
     * Dolna część: pozostałe przypadki (oznacza ścieżkę pieszo/rowerową)

2. **Identyfikacja stref:**

   * **Tory tramwajowe:** Jeśli wskaźnik Intersection-over-Union (IoU) wycinka obiektu z jedną z dwóch stref tramwajowych wynosi ponad 99%, obiekt klasyfikowany jest jako znajdujący się na torach. Warto zaznaczyć, że nie jest to standardowa definicja Intersection-over-Union (IoU), stosowana w uczeniu maszynowym, gdzie stosunek liczony jest względem sumy pól obiektu i strefy pomniejszonej o ich część wspólną. W omawianym przypadku wykorzystywana funkcja oblicza, w jakim stopniu obszar obiektu pokrywa się ze strefą tramwajową, ale wynik dzielony jest wyłącznie przez powierzchnię strefy tramwaju. 

   * **Jezdnia:** Sprawdzana jest dolna krawędź wycinka (bottom_y) oraz jej położenie względem predefiniowanych zakresów współrzędnych. Zamiast centroidu zdecydowano się na wykorzystanie dolnej krawędzi, ponieważ w przypadku dużych obiektów środek bounding boxa często wypadał poza właściwą strefę – na przykład przesuwał się nad górną krawędź jezdni lub w stronę torów tramwajowych – co prowadziło do błędnej klasyfikacji.

   * **Chodnik:** Obiekty niezakwalifikowane do powyższych stref przypisane są do strefy chodnika.

3. **Przypisanie etykiety (label):**

   * Jeżeli obiekt jest na jezdni:

     * Powierzchnia wycinka powyżej 60000 pikseli i szerokość powyżej 550 pikseli: „ciężarowy/autobus”
     * Pozostałe: „osobowy”
   * Jeżeli obiekt jest na torach tramwajowych:

     * Wysokość wycinka powyżej 200 pikseli i szerokość powyżej 800 pikseli: „tramwaj”
   * Jeżeli obiekt jest na chodniku:

     * Szerokość wycinka powyżej 60 pikseli i wysokość powyżej 110 pikseli: „pieszy”

Obiekty, które nie spełniają tych kryteriów, są pomijane.

## Warunki inkrementacji/finalnej detekcji obiektów

* **Tramwaje:** Na obiekty z etykietą "tramwaj" stosowana jest blokada czasowa (5 sekund), aby uniknąć wielokrotnego zliczania tego samego tramwaju.
* **Piesi:** Obiekty z etykietą "pieszy" zliczani są tylko gdy centroid obiektu znajduje się wewnątrz dedykowanej strefy pieszej. 
* **Samochody (osobowe i ciężarowe/autobusy):** Obiekty z etykietami "osobowy" oraz "ciezarowy/autobus" identyfikowane są w strefach również na podstawie centroidu oraz zliczane są w zależności od kierunku jazdy:
  * Obiekty w zielonym bbox – poruszające się z lewej na prawo.
  * Obiekty w niebieskim bbox – poruszające się z prawej na lewo.

Aby ten sam obiekt w stefie zliczany był tylko raz, dodajemy ID zliczonego obiektu w celu potwierdzenia, że był już uwzgledniony przy liczeniu. 


## Śledzenie obiektów

Śledzenie obiektów realizowane jest za pomocą algorytmu opartego na deskryptorach HOG (Histogram of Oriented Gradients). Detekcje przetrzymywane są w specjalnej strukturze danych, która zawiera:

* **ID obiektu:** unikalny numer przypisany każdemu obiektowi.
* **Centroid:** środek geometryczny obiektu.
* **Deskryptor HOG:** zestaw cech opisujących wygląd obiektu.
* **Bounding box:** prostokąt opisujący obiekt.
* **Licznik zniknięcia:** liczba klatek, przez które obiekt nie jest widoczny.

Algorytm działa według następujących kroków:

* W każdej klatce analizowane są aktualnie wykryte obiekty, dla których obliczany jest deskryptor HOG.
* Następuje porównanie aktualnych obiektów z obiektami przetrzymywanymi w pamięci:
  * Na samym początku przeprowadzana jest wstępna weryfikacja podobieństwa w postaci odległości centroidów porównywanych obiektów
  * Następnie wybierane jest najlepsze dopasowanie na podstawie odległości euklidesowej między deskryptorami.
  * Dopasowane obiekty mają aktualizowane centroidy, deskryptory i bounding boxy.
  * Jeśli obiekt nie jest widoczny przez zdefiniowaną liczbę klatek, usuwany jest ze struktury danych.
* Nowe, wcześniej niewidziane obiekty otrzymują nowe ID i są dodawane do struktury danych.

Śledzenie zapewnia spójność identyfikacji obiektów w kolejnych klatkach, umożliwiając ich precyzyjne zliczanie według trajektorii przemieszczania.


## Szczegółowy schemat logiki detekcji

### 1. Inicjalizacja

* Wczytanie obrazu tła.
* Konwersja obrazu tła do skali szarości.

### 2. Pętla przetwarzania klatek wideo

#### Dla każdej klatki:

* Wczytanie bieżącej klatki wideo.
* Konwersja do skali szarości.
* Obliczenie obrazu różnicowego względem tła (`cv2.absdiff`).
* Progowanie binarne obrazu różnicowego, uzyskując maskę pierwszoplanową (`cv2.threshold`).
* Morfologiczne otwarcie dla redukcji szumów (`cv2.morphologyEx`, `cv2.MORPH_OPEN`).
* Morfologiczne zamknięcie dla poprawienia konturów obiektów (`cv2.morphologyEx`, `cv2.MORPH_CLOSE`).
* Generowanie konturów metodą `cv2.findContours`.

### 3. Analiza i klasyfikacja konturów

* Iteracja po każdym wykrytym konturze:

  * Obliczenie powierzchni konturu i filtrowanie małych obiektów.
  * Wyznaczenie prostokąta ograniczającego (bounding box) i centroidu obiektu.
  * Identyfikacja strefy (chodnik, jezdnia, tory tramwajowe) na podstawie centroidu i współrzędnych.
  * Klasyfikacja obiektu na podstawie powierzchni i wymiarów (samochód osobowy, ciężarowy/autobus, tramwaj, pieszy).

### 4. Aktualizacja danych śledzenia

* Obliczenie deskryptora HOG dla każdego sklasyfikowanego obiektu.
* Porównanie deskryptorów HOG z poprzednimi klatkami:

  * Dopasowanie obiektów na podstawie minimalnej odległości euklidesowej.
  * Aktualizacja istniejących obiektów (ID, pozycja, bounding box, deskryptor).
  * Dodanie nowych obiektów do struktury danych śledzenia.
  * Usuwanie obiektów, które nie były widoczne przez określoną liczbę klatek.

### 5. Zliczanie obiektów

* Sprawdzenie warunków specyficznych dla poszczególnych typów obiektów (piesi, samochody, tramwaje):

  * Tramwaje: stosowanie blokady czasowej 5 sekund między kolejnymi zliczeniami.
  * Samochody: zliczanie według określonych kierunków ruchu w strefach jezdni (lewo-prawo lub prawo-lewo).

### 6. Zapis wyników

* Inkrementacja odpowiednich liczników w wynikowej strukturze danych.
* Zapis wyników po zakończeniu przetwarzania wszystkich klatek.

## Zastosowane algorytmy i techniki

- **Model tła**: Porównanie z pojedynczym obrazem tła (`background.jpg`)
- **Binary Thresholding**: Wartość progowa = 30
- **Operacje morfologiczne**: `cv2.MORPH_OPEN`, `cv2.MORPH_CLOSE`
- **Ekstrakcja cech**: Histogram Oriented Gradients (HOG) dla obiektów
- **Śledzenie obiektów**: Matching na podstawie deskryptorów HOG i odległości euklidesowej
- **Zliczanie obiektów**: Na podstawie pozycji centroidu w zdefiniowanych strefach
- **Heurystyki klasyfikacyjne**: Rozmiar, pozycja i proporcje bounding boxa

---

## Brak modelu ML

Projekt **nie używał modelu uczenia maszynowego**. Cała logika została oparta o klasyczne przetwarzanie obrazu z wykorzystaniem OpenCV.


# Podsumowanie projektu 

Celem projektu było opracowanie systemu detekcji i klasyfikacji pojazdów, tramwajów oraz pieszych na podstawie analizy sekwencji wideo. Zastosowano klasyczne metody wizji komputerowej, obejmujące model różnicowy tła, operacje morfologiczne, analizę konturów oraz śledzenie obiektów przy wykorzystaniu deskryptorów HOG.

## Najważniejsze trudności projektowe

### 1. Separacja nachodzących na siebie obiektów

Zdecydowanie największym wyzwaniem projektu była separacja obiektów, których kontury nakładały się wizualnie w kadrze. W klasycznych zastosowaniach systemów śledzenia i detekcji pojazdów zazwyczaj wykorzystywane są ujęcia z kamery ustawionej wysoko i skierowanej bespośrednio frontalnie na obszar, często dodatkowo każdy pas ruchu ma osobną osobną kamerę. W tym projekcie natomiast wykorzystano kadr boczny, a taka perspektywa boczna powodowała liczne problemy. Między innymi to, że pojazdy się przysłaniały lub były zakryte przez inne elementy lokacji (np. drzewa, słupki). Dodatkowo pojazdy znajdujące się bliżej kamery były znacznie większe, dominując wizualnie obiekty dalsze. W przypadku samochodów osobowych problem częściowo udało się zminimalizować poprzez precyzyjne dobranie liczby iteracji operacji morfologicznych (otwarcia i zamknięcia). Jednakże gdy większy pojazd znajdował się w tle, jego kontur dominował, co uniemożliwiało poprawną separację mniejszych pojazdów.

### 2. Dobór operacji morfologicznych

Ze względu na złożoność sceny oraz charakterystykę nagrania, jedynym skutecznym podejściem w celu uzyskania bardziej optymalnie stabilnych konturów było dobieranie różnych ilości iteracji operacji morfologicznych. Zastosowanie zaawansowanego przetwarzania konturów (postprocessing, próby przetwarzania wszystkich konturów w obiekcie i ich grupowania, a nie tylko zewnętrznego) nie przynosiło pożądanych rezultatów, gdyż blisko znajdujące się obiekty lub większe obiekty w tle nadal powodowały zlewanie się konturów. W efekcie ograniczono się do intensywnych eksperymentów empirycznych.

### 3. Stabilność klasyfikacji

Kolejnym istotnym wyzwaniem była stabilność klasyfikacji obiektów. Obiekty często były widoczne zbyt krótko z powodu przeszkód (drzewa, słupy, inne obiekty w ruchu), co znacznie utrudniało stabilną klasyfikację. Wiele pojazdów było poprawnie identyfikowanych jedynie przez krótką chwilę, co skutkowało trudnościami w utrzymaniu ciągłości ich detekcji. 

W tym kontekście kluczową rolę odegrała jakość aktualizacji wykrytych obiektów – ostatecznie najlepsze rezultaty uzyskano dzięki zastosowaniu deskryptorów HOG, które wykazywały największą odporność na zakłócenia konturów i zmienność wyglądu obiektów. Wcześniej podejmowano próby zastosowania porównania histogramów HSV, jednak podejście to okazało się zdecydowanie mniej skuteczne w warunkach dużej zmienności sceny.

Dodatkowym problemem była niestabilność konturów, która prowadziła do znacznych skoków w rozmiarze bounding boxa lub dzieliła jeden obiekt na kilka mniejszych. W efekcie obiekt mógł być interpretowany jako kilka nowych, co całkowicie uniemożliwiało jego poprawne śledzenie. Z tego względu, przy tak niestabilnym i dynamicznie zmieniającym się kształcie obiektów, nie było możliwe zastosowanie filtra Kalmana - zakładane przez nie założenia ciągłości i przewidywalności ruchu nie były spełnione.

### 4. Problemy specyficzne dla tramwajów

W przypadku tramwajów wielokrotnie dochodziło do nadmiernego zliczania pojedynczych obiektów ze względu na długie kontury tramwajów, które rozciągały się na kilka stref jednocześnie (wymiary tramwaju powodowały, że kompletnie przykrywał całą górną jezdnię i też jej strefę detekcji). Aby przeciwdziałać temu zjawisku, zastosowano rozwiązanie polegające na blokadzie czasowej (5-sekundowej), uniemożliwiającej ponowne zliczanie tego samego tramwaju. To uproszczone rozwiązanie skutecznie ograniczyło błędy w tym zakresie.

### 5. Detekcja pieszych

Detekcja pieszych okazała się również problematyczna, zwłaszcza w przypadku osób przemieszczających się z wózkami dziecięcymi, które trudno było jednoznacznie sklasyfikować jako pojedynczego pieszego. Konieczne było dobranie kryteriów wymiarowych, co jednak nie rozwiązało całkowicie problemu dla wszystkich możliwych przypadków. Dodatkowo czasami pojawiały się dzieci obok dorosłego co powodowało znowu problemy z jednoznaczną klasyfikacją bo ich bboxy się łączyły a to kompletnie zaburzało klasyfikację. Na samym początku dzieci generowały bardzo małe bounding boxy, co prowadziło do ich odrzucania podczas filtrowania obiektów na podstawie minimalnych wymiarów. 

### 6. Detekcja rowerzystów

Rowerzyści nie zostali uwzględnieni w warunkach detekcji ze względu na znikomą liczbę wystąpień w analizowanym materiale wideo - w trakcie testów nie odnotowano żadnego jednoznacznego przypadku obecności rowerzysty. Co więcej, próby definiowania warunków klasyfikacyjnych dla rowerzystów mogłyby prowadzić do błędnych identyfikacji, np. sklasyfikowania osoby z wózkiem dziecięcym jako rowerzysty, ze względu na podobieństwo rozmiaru i proporcji obiektu. W związku z tym zrezygnowano z prób wykrywania tej klasy obiektów, aby nie pogarszać dokładności pozostałych klasyfikacji.


## Wnioski

* Zastosowanie klasycznych operacji na obrazie (model tła, morfologia) okazało się niewystarczające do pełnej separacji złożonych scen dynamicznych, szczególnie gdy obiekty różnią się znacznie rozmiarami i nakładają się wizualnie.
* Skuteczna i stabilna detekcja w scenach o tak trudnym kadrze wymaga zastosowania bardziej zaawansowanych technik, takich jak głębokie sieci neuronowe, które efektywniej radzą sobie z nakładającymi się obiektami i trudnymi warunkami wizualnymi.
* Operacje morfologiczne wymagają empirycznego doboru parametrów, co jest procesem czasochłonnym i nie zawsze zapewniającym pełną skuteczność.
* Proste metody śledzenia oparte na deskryptorach HOG zapewniły podstawową ciągłość śledzenia, lecz nie eliminowały wszystkich problemów klasyfikacji i separacji obiektów.

Podsumowując, projekt pozwolił na dokładne przeanalizowanie ograniczeń klasycznych metod wizji komputerowej w kontekście detekcji dynamicznych i złożonych scen. Potwierdził konieczność stosowania nowoczesnych metod uczenia maszynowego, szczególnie sieci neuronowych, do rozwiązywania zaawansowanych problemów detekcji obiektów w trudnych warunkach wizualnych.

