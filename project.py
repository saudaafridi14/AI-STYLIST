import os
import shutil
import random
import sys
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

DATASET_DIR = r"C:\Users\pc\OneDrive\Desktop\women"

CATEGORIES = [
    "co-ord eastern sets",
    "frock",
    "fusion wear women",
    "khaddar dresses",
    "long shirts with cigarette pants",
    "office wear"
]

IMG_SIZE = (224, 224)

TRAIN_DIR = r"C:\dataset\fashion\train"
VAL_DIR = r"C:\dataset\fashion\val"

# If True the script will move files into train/val (removing originals).
# Can be enabled by passing `--overwrite` or `-o` on the command line.
OVERWRITE_ORIGINALS = False

# Enable from CLI
if "--overwrite" in sys.argv or "-o" in sys.argv:
    OVERWRITE_ORIGINALS = True

# ─────────────────────────────────────────
# CHECK DATASET
# ─────────────────────────────────────────

def check_dataset():

    print("\nDATASET SUMMARY")
    print("=" * 50)

    total = 0

    for category in CATEGORIES:

        folder_path = os.path.join(DATASET_DIR, category)

        images = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"{category} : {len(images)} images")

        total += len(images)

    print("=" * 50)
    print(f"TOTAL IMAGES : {total}")

# ─────────────────────────────────────────
# PREPROCESS IMAGES
# ─────────────────────────────────────────

def preprocess_images():

    print("\nPREPROCESSING IMAGES...\n")

    for category in CATEGORIES:

        folder_path = os.path.join(DATASET_DIR, category)

        processed = 0
        removed = 0

        for filename in os.listdir(folder_path):

                file_path = os.path.join(folder_path, filename)

                # skip directories or non-files
                if not os.path.isfile(file_path):
                    continue

                try:
                    img = Image.open(file_path).convert("RGB")

                    # Resize image
                    img = img.resize(IMG_SIZE)

                    # Save cleaned image
                    img.save(file_path)

                    processed += 1

                except Exception:
                    try:
                        os.remove(file_path)
                        removed += 1
                    except Exception:
                        # couldn't remove (e.g., permission), just skip
                        pass

        print(f"{category}")
        print(f"Processed : {processed}")
        print(f"Removed   : {removed}\n")

# ─────────────────────────────────────────
# VISUALIZE SAMPLES
# ─────────────────────────────────────────

def visualize_samples():

    fig, axes = plt.subplots(len(CATEGORIES), 3, figsize=(10, 20))

    fig.suptitle("AI Stylist Dataset Samples", fontsize=18)

    for row, category in enumerate(CATEGORIES):

        folder_path = os.path.join(DATASET_DIR, category)

        images = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        samples = random.sample(images, min(3, len(images)))

        for col, filename in enumerate(samples):

            img_path = os.path.join(folder_path, filename)

            img = Image.open(img_path)

            axes[row, col].imshow(img)

            axes[row, col].axis("off")

            if col == 0:
                axes[row, col].set_title(category)

    plt.tight_layout()

    plt.savefig("fashion_dataset_samples.png")

    print("\nSample visualization saved as:")
    print("fashion_dataset_samples.png")

    plt.show()

# ─────────────────────────────────────────
# SPLIT DATASET
# ─────────────────────────────────────────

def split_dataset(val_split=0.2):

    print("\nSPLITTING DATASET...\n")

    for category in CATEGORIES:

        source_folder = os.path.join(DATASET_DIR, category)

        images = [
            f for f in os.listdir(source_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        split_index = int(len(images) * (1 - val_split))

        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create folders
        train_category_path = os.path.join(TRAIN_DIR, category)
        val_category_path = os.path.join(VAL_DIR, category)

        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(val_category_path, exist_ok=True)

        # Copy or move train images
        for img in train_images:
            src_path = os.path.join(source_folder, img)
            dst_path = os.path.join(train_category_path, img)
            if OVERWRITE_ORIGINALS:
                try:
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"Error moving {src_path} -> {dst_path}: {e}")
            else:
                shutil.copy(src_path, dst_path)

        # Copy or move validation images
        for img in val_images:
            src_path = os.path.join(source_folder, img)
            dst_path = os.path.join(val_category_path, img)
            if OVERWRITE_ORIGINALS:
                try:
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"Error moving {src_path} -> {dst_path}: {e}")
            else:
                shutil.copy(src_path, dst_path)

        print(f"{category}")
        print(f"Train Images : {len(train_images)}")
        print(f"Validation   : {len(val_images)}\n")

    print("DATASET SPLITTING COMPLETE!")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("         AI STYLIST DATASET PREPARATION")
    print("=" * 60)

    check_dataset()

    preprocess_images()

    # Uncomment if you want visualization
    # visualize_samples()

    split_dataset()

    print("\nDATASET IS READY FOR TENSORFLOW TRAINING!")