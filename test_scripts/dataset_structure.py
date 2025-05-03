import h5py

# Open the HDF5 file
file_path = '../dataset_test/aloha_fork_pick_up_compressed/episode_43.hdf5'  # Replace with your HDF5 file path
with h5py.File(file_path, 'r') as file:
    # Print all the top-level keys (groups and datasets)
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name} - Shape: {obj.shape} - Dtype: {obj.dtype}")
    
    # Walk through the HDF5 file and print its structure
    file.visititems(print_structure)
