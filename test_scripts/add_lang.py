import h5py
import numpy as np
import os

# Path to the folder containing the HDF5 files
folder_path = '../dataset_test/aloha_fork_pick_up_compressed/'  # Replace with the path to your folder

# Loop through all files in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith('.hdf5'):  # Process only .h5 files
        file_path = os.path.join(folder_path, filename)
        
        # Open the existing HDF5 file in append mode ('a' mode)
        with h5py.File(file_path, 'a') as file:
            # Check if 'language_raw' already exists, if not, create it with the desired text
            if 'language_raw' not in file:
                language_raw_data = "Can you pick up the fork, please?"
                file.create_dataset('language_raw', data=np.string_(language_raw_data))  # Add the string dataset
                print("Languge addition successful")
            else:
                print(f"language_raw dataset already exists in {filename}.")

print("All HDF5 files processed.")
