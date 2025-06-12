import splitfolders  # Install using: pip install split-folders

input_folder = r"C:\Users\Vansh\Downloads\Monika"  # Change this to your dataset path
output_folder = r"C:\Users\Vansh\OneDrive\Desktop\dataset"  # New location for split dataset

# Split dataset into 80% train, 10% validation, and 10% test
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.1, 0.1), group_prefix=None)
