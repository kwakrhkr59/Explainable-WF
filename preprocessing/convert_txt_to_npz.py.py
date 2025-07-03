import os
import numpy as np
from tqdm import tqdm
from natsort import natsorted, ns

import logging
logger = logging.getLogger(__name__)

def parse_timestamp_dirsize(instance):
    dirs = []
    for line in map(str.strip, instance):
        timestamp, dirsize = map(float, line.split('\t'))
        dirs.append([timestamp, dirsize])
    return dirs

def save_as_npz(dataset_dir, output_file, length=5000):
    all_files = [file for file in os.listdir(dataset_dir)]
    sorted_files = natsorted(all_files, alg=ns.IGNORECASE)

    X_data = []
    y_labels = []

    for file in tqdm(sorted_files, desc="Processing Files", unit="file"):
        file_path = os.path.join(dataset_dir, file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = parse_timestamp_dirsize(f)
                label = file.split('-')[0]

            if len(data) > length:
                data = data[:length]
            elif len(data) < length:
                if len(data) == 0:
                    data = [[0,0] * length]
                else:
                    last_timestamp = data[-1][0]
                    data = data + [[last_timestamp, 0]] * (length - len(data))

            X_data.append(np.array(data, dtype=np.float32))
            y_labels.append(label)
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            continue

    if not X_data:
        raise ValueError("No data loaded. Check dataset_dir and file contents.")

    X_data = np.array(X_data, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=str)

    if os.path.exists(output_file):
        input("Output file already exists. Do you want to overwrite it? (y/n): ")
        if input().strip().lower() != 'y':
            logger.info("Exiting without saving.")
            return
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"Saving data to {output_file}...")
    logger.info(f"Data shape: {X_data.shape}, Labels shape: {y_labels.shape}")
    np.savez_compressed(output_file, X=X_data, y=y_labels)

if __name__ == "__main__":
    dataset_dir = "/scratch2/TrafficSliver/DeepCoAST/BE-original/mon_standard/"
    output_file = "output/data.npz"

    try:
        save_as_npz(dataset_dir, output_file)
        logger.info(f"Data saved to {output_file} successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")