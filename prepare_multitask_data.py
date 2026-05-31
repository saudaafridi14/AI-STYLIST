import os
import cv2
import numpy as np
import pandas as pd
import json

# Paths relative to the project directory
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
MODELS_DIR = "models"
LABELS_JSON_PATH = os.path.join(MODELS_DIR, "multitask_labels.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# Define label catalogs
PATTERNS = ["Solid", "Printed", "Embroidered"]
COLORS = ["White", "Black", "Grey", "Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink"]

def get_style_classes():
    # Dynamically find the class names from the train directory
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory '{TRAIN_DIR}' not found. Please run dataset split first.")
    classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    return classes

def infer_pattern_from_filename(filename):
    name_lower = filename.lower()
    # Keywords indicating prints or floral patterns
    print_keywords = ["print", "printed", "pattern", "patterned", "motif", "motifs", "stripe", 
                      "stripes", "striped", "check", "checks", "checkered", "bloom", "floral", "flower"]
    # Keywords indicating heavy embroidery or festive wear
    emb_keywords = ["embroidered", "embroidery", "festive", "lace", "work", "eid", "emb", "ethnic", "motifs"]
    
    if any(k in name_lower for k in print_keywords):
        return "Printed"
    elif any(k in name_lower for k in emb_keywords):
        return "Embroidered"
    else:
        return "Solid"

def extract_dominant_color(image_path):
    """
    Load image, crop center area, perform K-Means to find dominant color,
    and classify it into one of the pre-defined COLORS.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Grey"
        
        # Crop center 50% of the image to focus on the clothing item
        h, w, _ = img.shape
        cy_start, cy_end = int(h * 0.25), int(h * 0.75)
        cx_start, cx_end = int(w * 0.25), int(w * 0.75)
        crop = img[cy_start:cy_end, cx_start:cx_end]
        
        # Resize to speed up clustering
        crop = cv2.resize(crop, (50, 50))
        
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        
        # Filter out background pixels that are extremely white/black/grey if possible
        # but keep them if the whole outfit is white/black/grey.
        # Run K-Means to find top 3 color clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, flags)
        
        # Find the most frequent cluster
        counts = np.bincount(labels.flatten())
        dominant_hsv = centers[np.argmax(counts)]
        
        h, s, v = dominant_hsv
        
        # Color classification logic
        if v < 45:
            return "Black"
        if s < 30:
            if v > 200:
                return "White"
            else:
                return "Grey"
        
        # HSV hue classification
        if h < 10 or h >= 170:
            if h >= 155 and v > 100:
                return "Pink"
            return "Red"
        elif h < 25:
            return "Orange"
        elif h < 35:
            return "Yellow"
        elif h < 85:
            return "Green"
        elif h < 135:
            return "Blue"
        elif h < 160:
            return "Purple"
        else:
            return "Pink"
    except Exception:
        return "Grey"

def label_directory(directory, style_classes):
    data = []
    style_map = {style: idx for idx, style in enumerate(style_classes)}
    pattern_map = {pattern: idx for idx, pattern in enumerate(PATTERNS)}
    color_map = {color: idx for idx, color in enumerate(COLORS)}
    
    for style_name in style_classes:
        style_dir = os.path.join(directory, style_name)
        if not os.path.exists(style_dir):
            continue
        
        filenames = [f for f in os.listdir(style_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        print(f"Processing {len(filenames)} images in '{style_name}' ({directory})...")
        
        for filename in filenames:
            file_path = os.path.join(style_dir, filename)
            
            # Inferred labels
            style_idx = style_map[style_name]
            
            pattern_name = infer_pattern_from_filename(filename)
            pattern_idx = pattern_map[pattern_name]
            
            color_name = extract_dominant_color(file_path)
            color_idx = color_map[color_name]
            
            # Keep path relative to dataset directory or project folder for portability
            rel_path = os.path.join(directory, style_name, filename)
            
            data.append({
                "image_path": rel_path,
                "style_label": style_idx,
                "pattern_label": pattern_idx,
                "color_label": color_idx
            })
            
    return pd.DataFrame(data)

def main():
    styles = get_style_classes()
    print("Found styles:", styles)
    
    # Save the labels metadata to JSON
    metadata = {
        "styles": styles,
        "patterns": PATTERNS,
        "colors": COLORS
    }
    with open(LABELS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved label mappings to {LABELS_JSON_PATH}")
    
    # Process train and val sets
    print("\n--- Labeling Train Set ---")
    train_df = label_directory(TRAIN_DIR, styles)
    train_csv = os.path.join(DATASET_DIR, "train_labels.csv")
    train_df.to_csv(train_csv, index=False)
    print(f"Saved training labels to {train_csv} (Total: {len(train_df)} samples)")
    
    print("\n--- Labeling Validation Set ---")
    val_df = label_directory(VAL_DIR, styles)
    val_csv = os.path.join(DATASET_DIR, "val_labels.csv")
    val_df.to_csv(val_csv, index=False)
    print(f"Saved validation labels to {val_csv} (Total: {len(val_df)} samples)")
    
    print("\nDataset labeling complete!")

if __name__ == "__main__":
    main()
