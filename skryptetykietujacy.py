import cv2
import os

# Ścieżka do folderu z obrazami
image_folder = "testowe_zdjecia"
output_folder = "etykiety_testowe"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Pobierz listę plików z obrazami
image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # Wczytaj obraz
    img = cv2.imread(image_path)
    clone = img.copy()

    # Ustal rozmiar okna etykietowania jako 80% rozmiaru obrazu
    window_size = (int(img.shape[1] * 0.8), int(img.shape[0] * 0.8))
    
    # Stworzenie okna do ręcznego etykietowania
    cv2.namedWindow("Etykietowanie", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Etykietowanie", window_size)

    # Funkcja do obsługi zdarzeń
    def click_and_crop(event, x, y, flags, param):
        # Kliknięcie lewym przyciskiem myszy - zaznacz obszar
        if event == cv2.EVENT_LBUTTONDOWN:
            # Koordynaty początkowe
            global start_x, start_y
            start_x, start_y = x, y

        # Puść lewy przycisk myszy - zaznacz obszar
        elif event == cv2.EVENT_LBUTTONUP:
            # Koordynaty końcowe
            end_x, end_y = x, y

            # Zapisz etykietę do pliku
            label_filename = os.path.join(output_folder, f"{image_file.split('.')[0]}.txt")
            with open(label_filename, "a") as label_file:
                label_file.write(f"rejestracja_samochodowa {start_x} {start_y} {end_x} {end_y}\n")

            # Narysuj prostokąt na obrazie
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.imshow("Etykietowanie", img)

    # Podpięcie funkcji do obsługi zdarzeń
    cv2.setMouseCallback("Etykietowanie", click_and_crop)

    while True:
        # Wyświetlanie obrazu w oknie
        cv2.imshow("Etykietowanie", img)

        # Oczekiwanie na klawisz 'c' (continue) do przejścia do następnego obrazu
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

    # Czyść okno i przygotuj się do następnego obrazu
    cv2.destroyAllWindows()

print("Etykietowanie zakończone.")
#W tym kodzie cv2 to skrót od biblioteki OpenCV (Open Source Computer Vision Library), która jest używana do przetwarzania obrazów i wideo w języku Python. OpenCV zapewnia szereg funkcji do manipulacji obrazami, takich jak wczytywanie, zapisywanie, modyfikowanie i analizowanie obrazów.