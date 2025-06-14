"""
Slice a hdf5 file into a new hdf5 file.
Target file contains a large group 'data' with many episodes.
Extract a subset of episodes from the target file, and store each episode into a new hdf5 file.

Usage:
python slice.py <path_to_target_hdf5_file> <dir_for_output_hdf5_files> <start_index> <end_index>
"""

import h5py
import sys
import os

def copy_group(src_group, dest_group):
    """Recursively copy a group and all its contents to the destination."""
    for key in src_group:
        if isinstance(src_group[key], h5py.Group):
            # Create a new group in the destination
            new_group = dest_group.create_group(key)
            # Recursively copy the group contents
            copy_group(src_group[key], new_group)
        else:
            # Copy the dataset
            src_group.copy(key, dest_group)

def slice_hdf5(path, output_dir, start_index, end_index):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(path, 'r') as f:
        data = f['data']
        for episode_name in data:
            episode_index = int(episode_name.split('_')[-1])  # Get episode index from dataset name
            if start_index <= episode_index <= end_index:
                # Create output file name for this episode
                output_file = os.path.join(output_dir, f'episode_{episode_index}.hdf5')
                with h5py.File(output_file, 'w') as f_out:
                    # Copy the entire episode group structure
                    copy_group(data[episode_name], f_out)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python slice.py <path_to_target_hdf5_file> <dir_for_output_hdf5_files> <start_index> <end_index>")
        sys.exit(1)
    
    target_path = sys.argv[1]
    output_dir = sys.argv[2]
    start_index = int(sys.argv[3])
    end_index = int(sys.argv[4])
    
    slice_hdf5(target_path, output_dir, start_index, end_index)