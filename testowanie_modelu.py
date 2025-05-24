import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw


class CustomDataset:
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(
            self.image_folder) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name  # Zwraca również nazwę pliku dla celów identyfikacyjnych


def get_ground_truth_labels(image_filename, transform=None):
    label_folder = "./etykiety_testowe"
    label_filename = os.path.join(
        label_folder, f"{image_filename.split('.')[0]}.txt")

    with open(label_filename, 'r') as file:
        lines = file.readlines()

        ground_truth_labels = []
        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5 or any(not val.isdigit() for val in parts[1:]):
                continue

            class_name = parts[0]
            box = list(map(int, parts[1:]))

            # Dodaj warunki sprawdzające poprawność pudełek obwiedniętych
            if box[2] > box[0] and box[3] > box[1]:
                MAX_WIDTH = 5000
                MAX_HEIGHT = 5000
                box[2] = min(box[2], MAX_WIDTH)
                box[3] = min(box[3], MAX_HEIGHT)

                # Zastosuj transformację do współrzędnych prostokąta
                if transform:
                    box = transform([box])[0]

                ground_truth_labels.append({'boxes': box, 'class': class_name})

        return ground_truth_labels



def calculate_iou(box1, box2):
    # Obliczenie Intersection over Union (IoU) pomiędzy dwoma prostokątami
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersect_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersect_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    area_intersect = intersect_x * intersect_y
    area_union = w1 * h1 + w2 * h2 - area_intersect

    iou = area_intersect / max(area_union, 1e-6)
    return iou


def calculate_precision_recall(predictions, ground_truth_labels, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Przejście przez wszystkie predykcje
    for pred_box in predictions[0]['boxes']:
        pred_box = [round(val, 2) if isinstance(
            val, (int, float)) else val for val in pred_box]

        # Sprawdzenie, czy istnieje zgodna etykieta ziemi w ground_truth_labels z IOU powyżej progu
        matched_gt = False
        for gt_label in ground_truth_labels:
            gt_box = gt_label['boxes']

            iou = calculate_iou(pred_box, gt_box)
            if iou > iou_threshold and gt_label['class'] == 'rejestracja_samochodowa':
                true_positives += 1
                matched_gt = True
                break

        if not matched_gt:
            false_positives += 1

    # Obliczenie liczby fałszywych negatywów
    false_negatives = len(ground_truth_labels) - true_positives

    # Obliczenie precyzji i czułości
    precision = true_positives / max((true_positives + false_positives), 1)
    recall = true_positives / max((true_positives + false_negatives), 1)

    return precision, recall


def get_false_positives(predictions, ground_truth_labels, iou_threshold=0.5):
    false_positives = 0

    # Przejście przez wszystkie etykiety ziemi
    for gt_label in ground_truth_labels:
        gt_box = gt_label['boxes']

        # Sprawdzenie, czy istnieje zgodna predykcja w predictions z IOU poniżej progu
        matched_pred = False
        for pred_box in predictions[0]['boxes']:
            pred_box = [round(val, 2) if isinstance(
                val, (int, float)) else val for val in pred_box]

            iou = calculate_iou(pred_box, gt_box)
            if iou > iou_threshold and gt_label['class'] == 'rejestracja_samochodowa':
                matched_pred = True
                break

        if not matched_pred:
            false_positives += 1

    return false_positives


def get_false_negatives(predictions, ground_truth_labels, iou_threshold=0.5):
    false_negatives = 0

    # Przejście przez wszystkie predykcje
    for pred_box in predictions[0]['boxes']:
        pred_box = [round(val, 2) if isinstance(
            val, (int, float)) else val for val in pred_box]

        # Sprawdzenie, czy istnieje zgodna etykieta ziemi w ground_truth_labels z IOU poniżej progu
        matched_gt = False
        for gt_label in ground_truth_labels:
            gt_box = gt_label['boxes']

            iou = calculate_iou(pred_box, gt_box)
            if iou > iou_threshold and gt_label['class'] == 'rejestracja_samochodowa':
                matched_gt = True
                break

        if not matched_gt:
            false_negatives += 1

    return false_negatives


def calculate_overall_accuracy(true_positives, false_positives, false_negatives, total_samples):
    accuracy = (true_positives) / max((true_positives +
                                       false_positives + false_negatives), 1) * 100
    return accuracy


# Dodane transformacje
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

# Utworzenie instancji zbioru danych z dodaną transformacją
test_dataset = CustomDataset(
    image_folder="./testowe_zdjecia", transform=transform)

# Utworzenie DataLoader
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False)

# Wczytanie wytrenowanego modelu
model = fasterrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load('finalny_model.pth'))
model.eval()

# Zainicjowanie zmiennych do obliczeń
true_positives_total = 0
false_positives_total = 0
false_negatives_total = 0
total_samples = 0

# Przetestowanie modelu na nowych obrazach
for images, file_names in test_data_loader:
    with torch.no_grad():
        predictions = model(images)

    # Przygotowanie obrazu do rysowania prostokątów
    image = transforms.ToPILImage()(images[0])
    draw = ImageDraw.Draw(image)

    # Ocena skuteczności modelu i analiza błędów
    ground_truth_labels = get_ground_truth_labels(file_names[0])
    true_positives = len(predictions[0]['boxes'])
    false_positives = get_false_positives(predictions, ground_truth_labels)
    false_negatives = get_false_negatives(predictions, ground_truth_labels)

    # Dodanie wyników dla obecnego obrazu do ogólnej liczby
    true_positives_total += true_positives
    false_positives_total += false_positives
    false_negatives_total += false_negatives
    total_samples += 1
    for box in predictions[0]['boxes']:
        box = [round(val, 2) if isinstance(val, (int, float)) else val for val in box]

        draw.rectangle(box, outline="green", width=5)
    # Zapisanie obrazu z zaznaczonymi prostokątami
    output_path = os.path.join('./przetworzone_zdjecia', file_names[0])
    image.save(output_path)

# Obliczenie ogólnej skuteczności
overall_accuracy = calculate_overall_accuracy(
    true_positives_total, false_positives_total, false_negatives_total, total_samples)
print(f'Ogólna skuteczność modelu: {overall_accuracy:.2f}%')

