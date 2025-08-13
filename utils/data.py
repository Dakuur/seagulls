import os
import re
from PIL import Image

class Dataset:
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

        image = Image.open(os.path.join(self.images[idx]))
        match = re.search(r'\((.*?)\)', self.images[idx])
        if not match:
            raise ValueError(f"No label found in image path: {self.images[idx]}")
        label = match.group(1)

        return image, label

if __name__ == "__main__":
    #path = "/home/dakur/Downloads/ringreadingcompetition/datasets/lyngoy"
    path = "/home/dakur/Downloads/ringreadingcompetition/datasets/ringmerkingno"
    dataset = Dataset(path)
    print(f"Loaded {len(dataset.images)} images.")
    # Example usage: print the first image and label
    print(dataset.images[0])
    print(dataset[0])  # This will print the first image
    for i in range(100):
        print(dataset[i][1])
