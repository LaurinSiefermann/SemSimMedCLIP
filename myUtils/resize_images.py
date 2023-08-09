import pandas as pd
import os
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

# Base directory
base_dir = '/scratch1/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/'
new_base_dir = '/scratch1/MIMIC/physionet.org/files/mimic-cxr-jpg-resized/2.0.0/'

# Define the resize transformation
resize = Resize((256, 256), interpolation=Image.BICUBIC)


def resize_images(df, dataset_name):
    print(f"Resizing images for {dataset_name}...")
    # Iterate over the rows in the dataframe
    for _, row in tqdm(df.iterrows(), total=len(df)):
        jpg_paths = row['jpg_paths']

        # Iterate over the paths in the jpg_paths list
        for jpg_path in jpg_paths:
            # Full path to the original image
            full_path = os.path.join(base_dir, jpg_path)

            # Open the image
            img = Image.open(full_path)

            # Resize the image
            img_resized = resize(img)

            # Create the new directory if it doesn't exist
            new_path_dir = os.path.join(
                new_base_dir, os.path.dirname(jpg_path))
            if not os.path.exists(new_path_dir):
                os.makedirs(new_path_dir)

            # Full path to the resized image
            new_full_path = os.path.join(new_base_dir, jpg_path)

            # Save the resized image
            img_resized.save(new_full_path)
    print(f"Resizing for {dataset_name} is complete.\n")


# Read and resize images from different DataFrames
df_train = pd.read_pickle(
    '/home/lsiefermann/open_clip_based_thesis/myUtils/train_mimic.pkl')
df_val = pd.read_pickle(
    '/home/lsiefermann/open_clip_based_thesis/myUtils/val_mimic.pkl')
df_eval = pd.read_pickle(
    '/home/lsiefermann/open_clip_based_thesis/myUtils/eval_mimic_5x200.pkl')

resize_images(df_train, "Train Dataset")
resize_images(df_val, "Validation Dataset")
resize_images(df_eval, "Evaluation Dataset")
