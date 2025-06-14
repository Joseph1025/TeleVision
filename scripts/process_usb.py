"""
Post process episode images.
Input structure of an episode:
-root
    - absolute_actions
    - absolute_actions_6d
    - actions
    - actions_6d
    - dones
    - obs
        - agentview_images
        - joint_positions
        - robot0_eye_in_hand_image
        - robot1_eye_in_hand_image
    - rewards

Target structure:
processed_{episode_number}.hdf5
├── observation.image.left      # Left camera images (uint8 array)
├── observation.image.right     # Right camera images (uint8 array)
├── cmds                       # Command data (float32 array)
├── observation.state          # State data (float32 array)
├── qpos_action               # Joint position actions (float32 array)
└── Attributes:
    ├── sim                   # Boolean flag (False for real data)
    └── init_action          # Initial command (float32 array)
We map robot0_eye_in_hand_image to left_hand_image, and 
robot1_eye_in_hand_image to right_hand_image.
"""

import h5py
import numpy as np
import os
from pathlib import Path
import argparse


def load_episode_data(file_path):
    """Load data from the input HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        # Load images
        left_imgs = np.array(f['obs/robot0_eye_in_hand_image'])
        right_imgs = np.array(f['obs/robot1_eye_in_hand_image'])
        
        # Load states and actions
        states = np.array(f['obs/joint_positions'])
        actions = np.array(f['actions'])
        cmds = np.array(f['absolute_actions'])
        
        # Get timestamps from the data
        timestamps = np.arange(len(states))
        
    return left_imgs, right_imgs, states, actions, cmds, timestamps


def process_episode(input_file, output_file):
    """Process a single episode and save it in the target format."""
    # Load data
    left_imgs, right_imgs, states, actions, cmds, timestamps = load_episode_data(input_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save processed data
    with h5py.File(output_file, 'w') as hf:
        # Save datasets
        hf.create_dataset('observation.image.left', data=left_imgs.astype(np.uint8))
        hf.create_dataset('observation.image.right', data=right_imgs.astype(np.uint8))
        hf.create_dataset('cmds', data=cmds.astype(np.float32))
        hf.create_dataset('observation.state', data=states.astype(np.float32))
        hf.create_dataset('qpos_action', data=actions.astype(np.float32))
        
        # Save attributes
        hf.attrs['sim'] = False
        hf.attrs['init_action'] = cmds[0].astype(np.float32)


def find_all_episodes(path):
    """Find all episode files in the given directory."""
    episodes = [f for f in os.listdir(path) if f.endswith('.hdf5') and not f.startswith('processed_')]
    return episodes


def main():
    parser = argparse.ArgumentParser(description='Process USB data into the target format')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing episode files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed files')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all episodes
    episodes = find_all_episodes(args.input_dir)
    
    # Process each episode
    for episode in episodes:
        input_file = os.path.join(args.input_dir, episode)
        episode_number = episode.split('.')[0]  # Remove .hdf5 extension
        output_file = os.path.join(args.output_dir, f'processed_{episode_number}.hdf5')
        
        print(f'Processing {episode}...')
        process_episode(input_file, output_file)
        print(f'Processed {episode} -> {output_file}')

if __name__ == '__main__':
    main()