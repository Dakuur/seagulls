import matplotlib.pyplot as plt
import pickle
import os
import re

def plot_max_sizes_histogram():
    with open("max_sizes.pkl", "rb") as f:
        max_sizes = pickle.load(f)

    plt.hist(max_sizes, bins=50)
    plt.xlabel("Max Size")
    plt.ylabel("Frequency")
    plt.title("Histogram of Max Sizes of Cropped Images")
    plt.show()

def characters_types():
    path = "/home/dakur/Downloads/ringreadingcompetition/datasets/ringmerkingno/images"
    file_list = os.listdir(path)
    characters = set()

    for file_name in file_list:
        match = re.search(r'\((.*?)\)', file_name)
        if match:
            label = match.group(1)
            #print(f"File: {file_name}, Label: {label}")
            for c in label:
                characters.add(c)
        else:
            print(f"No label found in file: {file_name}")

    print("Characters found in labels:")
    for char in sorted(characters):
        print(char, end='')

characters_types()

chars = ["+","-","0","1","2","3","4","5","6","7","8","9",":","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","R","S","T","U","V","W","X","Y","Z"]