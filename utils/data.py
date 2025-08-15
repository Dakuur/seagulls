import re
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import pickle

def load_crop_yolo_image(img_path):
    # Path del txt correspondiente
    txt_path = os.path.join(
        os.path.dirname(os.path.dirname(img_path)),  # sube un nivel
        "labels",
        os.path.splitext(os.path.basename(img_path))[0] + ".txt"  # mismo nombre, extensión .txt
    )

    # Leer label YOLO (class, x_center, y_center, w, h, conf)
    with open(txt_path, "r") as f:
        line = f.readline().strip()
        class_id, x_center, y_center, w, h, conf = map(float, line.split())
        class_id = int(class_id)

    # Cargar imagen original
    image = Image.open(img_path).convert('RGB')
    img_width, img_height = image.size

    # Convertir coordenadas normalizadas a píxeles
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    w_px = w * img_width
    h_px = h * img_height

    # Calcular esquinas del bounding box
    x_min = int(x_center_px - w_px / 2)
    y_min = int(y_center_px - h_px / 2)
    x_max = int(x_center_px + w_px / 2)
    y_max = int(y_center_px + h_px / 2)

    # Asegurar que los límites están dentro de la imagen
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    # Recortar imagen al bounding box
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    # Redimensionar a 300x300
    cropped_image = cropped_image.resize((128, 128))

    # Transformar (normalizar y tensorizar)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    cropped_image = transform(cropped_image)

    return cropped_image


def show_tensor_image(tensor_img):
    # Desnormalizar (valores originales 0-1 para mostrar correctamente)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor_img * std + mean  # revertir normalización

    # Convertir a formato numpy (C,H,W) -> (H,W,C)
    img = img.permute(1, 2, 0).detach().cpu().numpy()

    # Clipping para asegurar valores entre 0 y 1
    img = img.clip(0, 1)

    # Mostrar
    plt.imshow(img)
    plt.axis('off')
    plt.show()

class SeagullDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.images = []
        self.load_data()

    def load_data(self):
        """
        Gets image names from specified directory.
        :return:
        """
        images_dir = os.path.join(self.directory, 'images')

        if not os.path.exists(images_dir):
            print(f"Error: Images directory {images_dir} does not exist.")
            return

        images = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
        print(f"Found {len(images)} images")
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.images):
            raise IndexError("Index out of range.")

        # IMAGE
        image = load_crop_yolo_image(self.images[idx])

        # LABEL IN IMAGE FILE NAME
        match = re.search(r'\((.*?)\)', self.images[idx])
        if not match:
            raise ValueError(f"No label found in image path: {self.images[idx]}")
        label = match.group(1)

        return image, label

if __name__ == "__main__":
    #path = "/home/dakur/Downloads/ringreadingcompetition/datasets/lyngoy"
    path = "/home/dakur/Downloads/ringreadingcompetition/datasets/ringmerkingno"
    dataset = SeagullDataset(path)
    print(f"Loaded {len(dataset.images)} images.")
    # Example usage: print the first image and label
    print(dataset.images[0])
    # show image
    show_tensor_image(dataset[0][0])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

