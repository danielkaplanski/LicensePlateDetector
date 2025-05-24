import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from PIL import Image
from torchvision.transforms import functional as F

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_folder = os.path.join(data_folder, "tmp_png")
        self.label_folder = os.path.join(data_folder, "etykiety")
        self.image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.png')]
        self.label_files = [f for f in os.listdir(self.label_folder) if f.endswith('.txt')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        label_name = img_name.replace('.png', '.txt')
        label_path = os.path.join(self.label_folder, label_name)
        labels = self.parse_labels(label_path)

        if self.transform:
            image = self.transform(image)

        return image, labels

    def parse_labels(self, label_path):
        labels = {}
        boxes = []
        with open(label_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) < 5 or any(not val.isdigit() for val in parts[1:]):
                    print(f"Ignoring invalid line: {line}")
                    continue

                class_name = parts[0]
                box = list(map(int, parts[1:]))
                
                # Dodaj warunki sprawdzające poprawność pudełek obwiedniętych
                if box[2] > box[0] and box[3] > box[1]:
                    # Ogranicz rozmiary pudełka obwiedniętego
                    MAX_WIDTH = 5000
                    MAX_HEIGHT = 5000
                    box[2] = min(box[2], MAX_WIDTH)
                    box[3] = min(box[3], MAX_HEIGHT)

                    boxes.append(box)
                else:
                    print(f"Ignoring invalid box: {box}")

            labels['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            labels['labels'] = torch.tensor([1] * len(boxes), dtype=torch.int64)

        return labels


def custom_collate(batch):
    images, targets = zip(*batch)

    # Skip converting to tensor if images are already tensors
    images = [image if isinstance(image, torch.Tensor) else F.to_tensor(image) for image in images]

    # Filter out samples with invalid boxes
    valid_targets = [target for target in targets if 'boxes' in target and 'labels' in target]
    valid_targets = [{'boxes': target['boxes'], 'labels': target['labels']} for target in valid_targets if target['boxes'].dim() == 2 and target['boxes'].size(1) == 4]

    return images, valid_targets

# Dodane transformacje
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])



# Utworzenie instancji zbioru danych z dodaną transformacją
dataset = CustomDataset(data_folder="./", transform=transform)

# Utworzenie DataLoader z niestandardową funkcją collate_fn
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)


# Reszta kodu szkoleniowego
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.train()
criterion = torch.nn.SmoothL1Loss()
# Zmiana parametrów optymalizatora
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)





# Pętla treningowa
epochs = 6
for epoch in range(epochs):
    for i, (images, targets) in enumerate(data_loader):
        optimizer.zero_grad()
        try:
            predictions = model(images, targets)
        except Exception as e:
            print(f"Error in forward pass: {e}")
            continue

        loss = sum([predictions[key].mean() for key in predictions if predictions[key] is not None])

        loss.backward()
        optimizer.step()

        # Wyświetlanie straty
        print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item()}")

    # Zapisanie wytrenowanego modelu po każdej epoce
    model_save_path = f'trained_model_epoch{epoch + 1}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved at: {model_save_path}')

# Zapisanie ostatecznego wytrenowanego modelu
torch.save(model.state_dict(), 'final_trained_model.pth')
