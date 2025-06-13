import cv2

points = []
rectangles = []

def draw_rectangle(event, x, y, flags, param):
    global points, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        if len(points) == 2:
            pt1, pt2 = points
            rectangles.append((pt1, pt2))
            print(f"Strefa: Lewy górny {pt1}, Prawy dolny {pt2}")
            points = []  # Reset for next rectangle

background = cv2.imread("background.jpg")
if background is None:
    raise FileNotFoundError("Nie znaleziono background.jpg")

clone = background.copy()
cv2.namedWindow("Wyznacz strefy")
cv2.setMouseCallback("Wyznacz strefy", draw_rectangle)

while True:
    temp = clone.copy()
    for pt1, pt2 in rectangles:
        cv2.rectangle(temp, pt1, pt2, (0, 255, 255), 2)
    if len(points) == 1:
        cv2.circle(temp, points[0], 5, (0, 0, 255), -1)
    cv2.imshow("Wyznacz strefy", temp)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
