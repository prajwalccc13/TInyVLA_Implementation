import h5py

# Open the HDF5 file in read mode
file_path = 'dataset_test/aloha_fork_pick_up_compressed/episode_43.hdf5'  # Replace with your file path


with h5py.File(file_path, 'r') as file:
    # Print all the top-level keys (groups and datasets)
    print("Top-level keys:", list(file.keys()))

    # Iterate over all keys at the top level
    for key in file.keys():
        print(f"\n--- {key} ---")
        
        # Check if the item is a group or a dataset
        item = file[key]
        
        # If it's a group, iterate over its subitems
        if isinstance(item, h5py.Group):
            print(f"Group: {key}")
            # If you want to show contents of subgroups
            for subkey in item.keys():
                print(f"  Subgroup or Dataset: {subkey}")
                # Display the data of each subgroup or dataset
                subitem = item[subkey]
                if isinstance(subitem, h5py.Dataset):
                    print(f"    Dataset shape: {subitem.shape}")
                    print(f"    Dataset content: {subitem[:]}")  # Display data
                else:
                    print(f"    Group content: {list(subitem.keys())}")
                    
        # If it's a dataset, just print its shape and content
        elif isinstance(item, h5py.Dataset):
            print(f"Dataset: {key}")
            print(f"  Dataset shape: {item.shape}")
            print(f"  Dataset content: {item[:]}")
