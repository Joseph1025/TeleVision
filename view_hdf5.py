"""
Prints the contents of a hdf5 file.
For each key value pair, output the key and the value to a txt log file.
If value exceeds 100 rows, truncate.

Usage:
python view_hdf5.py <path_to_hdf5_file>

"""

import h5py
import sys
import os
from datetime import datetime

def process_dataset(dataset):
    """Process a dataset and return its contents as a string."""
    data = dataset[()]
    if len(data.shape) > 0 and len(data) > 100:
        return f"{data[:100]}\n... (truncated, total rows: {len(data)})"
    return str(data)

def process_group(group, log_file):
    """Process a group and its contents iteratively, writing directly to the log file."""
    # Use a stack to keep track of groups to process
    # Each stack item is a tuple of (group, prefix)
    stack = [(group, "")]
    i = 0
    
    while stack and i < 100:
        current_group, prefix = stack.pop()
        i += 1
        
        print(f"Processing group: {prefix}")

        # Process all items in the current group
        for key in current_group.keys():
            item = current_group[key]
            full_key = f"{prefix}/{key}" if prefix else key
            
            if isinstance(item, h5py.Dataset):
                log_file.write(f"Dataset: {full_key}\n")
                log_file.write(f"Value:\n{process_dataset(item)}\n\n")
            elif isinstance(item, h5py.Group):
                log_file.write(f"Group: {full_key}\n")
                # Add the group to the stack for later processing
                stack.append((item, full_key))

def main():
    if len(sys.argv) != 2:
        print("Usage: python view_hdf5.py <path_to_hdf5_file>")
        sys.exit(1)

    hdf5_path = sys.argv[1]
    if not os.path.exists(hdf5_path):
        print(f"Error: File {hdf5_path} does not exist")
        sys.exit(1)

    # Create log directory if it doesn't exist
    log_dir = "./log"
    os.makedirs(log_dir, exist_ok=True)

    # Create log file name based on input file and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.basename(hdf5_path)
    log_file_path = os.path.join(log_dir, f"{os.path.splitext(input_filename)[0]}_{timestamp}.txt")

    try:
        with h5py.File(hdf5_path, 'r') as f, open(log_file_path, 'w') as log_file:
            # Write header
            print(f"HDF5 File Contents: {hdf5_path}\n")
            
            # Process the file contents and write directly to log file
            process_group(f, log_file)
            
            print(f"Contents have been written to: {log_file_path}")

    except Exception as e:
        print(f"Error processing HDF5 file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()


