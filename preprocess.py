# coding=utf8
import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path


class DataPreprocessor:
    """
    Handles the preprocessing pipeline for the SLE dataset, including:
    1. Parsing JSON annotations to extract lesion boundaries.
    2. Rasterizing polygon coordinates into binary segmentation masks.
    3. Performing Region of Interest (ROI) cropping to standardize input spatial domains.
    4. Organizing processed data into stratified splits (Train/Test) for reproducibility.
    """

    def __init__(self, raw_data_dir, output_dir, splits=None):
        """
        Initialize the preprocessor.

        Args:
            raw_data_dir (str): Path to the raw dataset containing 'images' and 'labels' subdirectories.
            output_dir (str): Path where the processed (cropped) dataset will be saved.
            splits (list): List of dataset splits to process (default: ['train', 'test']).
        """
        self.base_dir = raw_data_dir
        self.output_dir = output_dir
        self.splits = splits if splits else ["train", "test"]

        # Ensure output directory structure exists
        self._init_directories()

    def _init_directories(self):
        """Initialize the directory tree for processed data."""
        for split in self.splits:
            for subdir in ["images", "masks", "labels"]:
                os.makedirs(os.path.join(self.output_dir, split, subdir), exist_ok=True)

    def _rasterize_mask(self, points, width, height):
        """
        Convert polygon points into a binary mask.

        Args:
            points (list): List of (x, y) coordinates defining the lesion boundary.
            width (int): Image width.
            height (int): Image height.

        Returns:
            np.ndarray: Binary mask array.
        """
        mask = Image.new("L", (width, height), 0)
        # Draw polygon with fill=1 for the lesion area
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
        return np.array(mask)

    def _crop_roi(self, image, mask_np, box_points):
        """
        Crop the image and mask based on the lesion's bounding box.

        Args:
            image (PIL.Image): Original RGB image.
            mask_np (np.ndarray): Binary mask corresponding to the image.
            box_points (list): Polygon points (used here to determine bounds).

        Returns:
            tuple: (cropped_image, cropped_mask) as PIL objects.
        """
        # Determine bounding box from mask content
        ys, xs = np.where(mask_np)
        if len(xs) == 0 or len(ys) == 0:
            return None, None

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        # Perform cropping
        cropped_img = image.crop((xmin, ymin, xmax + 1, ymax + 1))
        # Convert mask back to image for cropping and rescaling values to 0-255
        mask_pil = Image.fromarray(mask_np)
        cropped_mask = mask_pil.crop((xmin, ymin, xmax + 1, ymax + 1)).point(lambda p: p * 255)

        return cropped_img, cropped_mask

    def process_split(self, split_name):
        """
        Process a single dataset split (e.g., 'train').

        Reads JSON labels, generates masks, crops ROIs, and saves artifacts.
        """
        image_dir = os.path.join(self.base_dir, split_name, "images")
        label_dir = os.path.join(self.base_dir, split_name, "labels")

        out_img_dir = os.path.join(self.output_dir, split_name, "images")
        out_mask_dir = os.path.join(self.output_dir, split_name, "masks")
        out_lbl_dir = os.path.join(self.output_dir, split_name, "labels")

        if not os.path.exists(label_dir):
            print(f"[Warning] Label directory not found: {label_dir}")
            return

        counter = 0
        file_list = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

        print(f"Processing {split_name} set ({len(file_list)} files)...")

        for fname in tqdm(file_list):
            label_path = os.path.join(label_dir, fname)

            # Load annotation
            with open(label_path, "r", encoding="utf-8") as f:
                try:
                    label_data = json.load(f)
                except json.JSONDecodeError:
                    continue

            # Load corresponding image
            img_name = fname.replace(".txt", ".jpg")
            img_path = os.path.join(image_dir, img_name)

            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")
            width, height = image.size

            # Iterate through annotated lesions (boxes)
            for i, box in enumerate(label_data.get("boxes", [])):
                points = box.get("points", [])
                # Filter invalid polygons
                if len(points) < 3:
                    continue

                # Format points for PIL
                points_tuple = [tuple(p) for p in points]

                # Generate binary mask
                mask_np = self._rasterize_mask(points_tuple, width, height)

                # Crop ROI
                cropped_img, cropped_mask = self._crop_roi(image, mask_np, points_tuple)

                if cropped_img is None:
                    continue

                # Save processed files
                save_id = f"{split_name}_img_{counter}"
                cropped_img.save(os.path.join(out_img_dir, f"{save_id}.png"))
                cropped_mask.save(os.path.join(out_mask_dir, f"{save_id}.png"))

                # Save class label (metadata)
                label_txt = box.get("label", "unknown")
                with open(os.path.join(out_lbl_dir, f"{save_id}.txt"), "w") as lf:
                    lf.write(label_txt + "\n")

                counter += 1

        print(f"Finished {split_name}: Generated {counter} samples.")

    def run(self):
        """Execute the full preprocessing pipeline."""
        print("Starting Data Preprocessing Pipeline...")
        for split in self.splits:
            self.process_split(split)
        print("Preprocessing Complete.")


if __name__ == "__main__":
    # Configuration for standalone execution
    # Ensure these paths match your environment setup
    RAW_DATASET_PATH = "./SLE"
    PROCESSED_DATASET_PATH = "./SLE_cropped_output"

    preprocessor = DataPreprocessor(
        raw_data_dir=RAW_DATASET_PATH,
        output_dir=PROCESSED_DATASET_PATH
    )

    preprocessor.run()
