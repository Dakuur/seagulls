import os

class Dataset:
    def __init__(self, directory):
        self.directory = directory
        self.images = []
        self.labels = []
        self.load_data()
        self.pair_data()

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

        # Load images
        image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        for img_file in sorted(image_files):
            #img_path = os.path.join(images_dir, img_file)
            self.images.append(img_file)

        # Load labels
        label_files = [os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
        for label_file in sorted(label_files):
            #label_path = os.path.join(labels_dir, label_file)
            self.labels.append(label_file)

        if len(self.images) != len(self.labels):
            print(f"Warning: Number of images ({len(self.images)}) does not match number of labels ({len(self.labels)}).")

    def pair_data(self):
        common =  set(self.images) & set(self.labels)
        if not common:
            print("No common files found between images and labels.")
            return
        paired_images = [img for img in self.images if img in common]
        paired_labels = [lbl for lbl in self.labels if lbl in common]
        print(f"Paired {len(paired_images)} images with labels.")
        self.images = sorted(paired_images)
        self.labels = sorted(paired_labels)

if __name__ == "__main__":
    #path = "/home/dakur/Downloads/ringreadingcompetition/datasets/lyngoy"
    path = "/home/dakur/Downloads/ringreadingcompetition/datasets/ringmerkingno"
    dataset = Dataset(path)
    print(f"Loaded {len(dataset.images)} images and {len(dataset.labels)} labels.")
    # Example usage: print the first image and label
    if dataset.images and dataset.labels:
        print(dataset.images[0])
        print(dataset.labels[0])
    dataset.pair_data()