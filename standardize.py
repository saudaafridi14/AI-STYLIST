from PIL import Image
import os

# Current dataset folder
dataset_path = "."

# Standard image size
SIZE = (224, 224)

# Loop through category folders
for category in os.listdir(dataset_path):

    category_path = os.path.join(dataset_path, category)

    if os.path.isdir(category_path):

        count = 1

        for filename in os.listdir(category_path):

            file_path = os.path.join(category_path, filename)

            try:
                # Open image
                img = Image.open(file_path).convert("RGB")

                # Resize image
                img = img.resize(SIZE)

                # New clean filename
                new_name = f"{category}_{count}.jpg"

                save_path = os.path.join(category_path, new_name)

                # Save image
                img.save(save_path, "JPEG")

                # Delete old image if name changed
                if file_path != save_path:
                    os.remove(file_path)

                count += 1

            except Exception as e:
                print("Skipped:", file_path)

print("Dataset standardized successfully!")