import os

class Dataset:
    def __init__(self, directory):
        self.directory = directory
        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        """
        Gets image and label names from specified directory.
        :return:
        """
        images_dir = os.path.join(self.directory, 'images')
        labels_dir = os.path.join(self.directory, 'labels')

        if not os.path.exists(images_dir):
            print(f"Error: Images directory {images_dir} does not exist.")
            return
        if not os.path.exists(labels_dir):
            print(f"Error: Labels directory {labels_dir} does not exist.")
            return

        images = [os.path.splitext(f)[0] for f in os.listdir(images_dir)]
        labels = [os.path.splitext(f)[0] for f in os.listdir(labels_dir)]
        common =  list(set(images) & set(labels))
        print(f"Found {len(images)} images and {len(labels)} labels. Common files: {len(common)}")

        if not common:
            print("Warning: No common files found between images and labels directories.")
            return

        for name in sorted(common):
            img_path = os.path.join(images_dir, name + '.jpg')  # Assuming images are in JPG format
            label_path = os.path.join(labels_dir, name + '.txt')
            self.images.append(img_path)
            self.labels.append(label_path)

        if len(self.images) != len(self.labels):
            print(f"Warning: Number of images ({len(self.images)}) does not match number of labels ({len(self.labels)}).")

if __name__ == "__main__":
    #path = "/home/dakur/Downloads/ringreadingcompetition/datasets/lyngoy"
    path = "/home/dakur/Downloads/ringreadingcompetition/datasets/ringmerkingno"
    dataset = Dataset(path)
    print(f"Loaded {len(dataset.images)} images and {len(dataset.labels)} labels.")
    # Example usage: print the first image and label
    if dataset.images and dataset.labels:
        print(dataset.images[0])
        print(dataset.labels[0])
