import os
import re
import numpy as np
import pandas as pd
import glob
import argparse
from tqdm import tqdm
from natsort import natsorted
from extractors import *
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

extractor_functions = [
    get_burst,
    get_direction,
    get_ipd,
    get_ipd_filtered,
    get_cumulative_size,
    get_size,
    get_1dtam,
    get_upload_tam,
    get_download_tam,
]

def get_feature_name(func):
    if isinstance(func, type(lambda: 0)):
        raw_name = func.__code__.co_name
    else:
        raw_name = func.__name__

    name = re.sub(r'^get_?', '', raw_name)
    return name

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process network traffic data for feature extraction.')
    parser.add_argument('--input-path', type=str, default="/home/kwakrhkr59/XAI_WF/defense/wtfpad/results/bigenough_tor_fiber",
                        help='Root folder for input traffic traces. Assumes structure like input_path/label_code/instance_file.')
    parser.add_argument('--output-path', type=str, default="/home/kwakrhkr59/XAI_WF/dataset/wdfpad",
                        help='Path to save the output NPZ files.')
    parser.add_argument('--csv-path', type=str, default="/home/kwakrhkr59/starlink/filter/result.csv",
                        help='Path to the CSV file that contains average accuracy (currently unused in this script).')
    parser.add_argument('--start-site', type=int, default=0,
                        help='Starting index of the site_code to use (currently unused in this script).')
    parser.add_argument('--classNum', type=int, default=95,
                        help='Number of classes to use (currently unused in this script).')
    parser.add_argument('--start-instance', type=int, default=0,
                        help='Starting index of instances to use per class (currently unused in this script).')
    parser.add_argument('--instanceNum', type=int, default=200,
                        help='Number of instances to use per class (currently unused in this script).')
    return parser.parse_args()

def process_instance(file_path):
    with open(file_path, 'r') as f:
        instance = []
        for line in f:
            parts = line.strip().split('\t')
            instance.append((float(parts[0]), int(float(parts[1]))))

    return instance

def extract_to_ndarray(traces, label, func, desired_length=5000):
    try:
        result = func([traces])[0]
    except Exception as e:
        logging.error(f"Error calling {func.__name__} for label {label}: {e}")
        return None

    if len(result) < desired_length:
        result += [0.0] * (desired_length - len(result))
    elif len(result) > desired_length:
        result = result[:desired_length]

    result.append(int(label))

    return np.array(result, dtype=np.float32)

def save_npz(features, save_path):
    if not features:
        logging.warning(f"No features to save for {save_path}. Skipping save operation.")
        return

    features_array = np.array(features)

    features_array = features_array[features_array[:, -1].argsort()]

    X = features_array[:, :-1]
    y = features_array[:, -1].astype(np.int64)

    np.savez(save_path, data=X.astype(np.float32), labels=y)
    logging.info(f"Successfully saved {len(X)} instances to: {save_path}")


def extract_features_for_group(parsed_traces, input_path, output_path): 
    ensure_dir(output_path)

    for func in extractor_functions:
        func_name = func.__name__ if not isinstance(func, type(lambda: 0)) else func.__code__.co_name
        feature_name = get_feature_name(func)
        logging.info(f"[*] Extracting feature: {feature_name} (using {func_name})")

        data = []
        for trace_info in tqdm(parsed_traces, desc=f"Extracting {feature_name}"):
            label = trace_info['label']
            traces = trace_info['data']

            feature = extract_to_ndarray(traces, label, func)
            if feature is not None:
                data.append(feature)
        
        extracted_features_by_type[feature_name] = data

        ensure_dir(output_path)
        save_npz(data, os.path.join(output_path, f"bigenough_{feature_name}_all.npz"))

    logging.info("[+] Feature extraction completed.")

def parse_trace(input_path):
    logging.info(f"\n[+] Starting extraction from: {input_path}")

    all_trace_files = natsorted(glob.glob(os.path.join(input_path, '*-*')))

    parsed_traces = []
    logging.info("Scanning and parsing all trace files...")
    for file_path in tqdm(all_trace_files, desc="Parsing Files"):
        label = file_path.split('/')[-1].split('-')[0]
        
        traces_data = process_instance(file_path)
        if traces_data is not None:
            parsed_traces.append({'label': label, 'data': traces_data})

    if not parsed_traces:
        logging.error(f"No valid trace files found in {input_path}. Exiting.")
        return None

    return parsed_traces

def extract_to_ndarray2d(traces, desired_length=5000):
    traces = np.array(traces)
    current_length = traces.shape[0]

    if current_length < desired_length:
        pad_len = desired_length - current_length
        padding = np.zeros((pad_len, 2), dtype=np.float32)
        traces = np.vstack([traces, padding])
    elif current_length > desired_length:
        traces = traces[:desired_length]

    return traces

def save_npz2d(data, labels, save_path):
    if not data:
        logging.warning(f"No data to save for {save_path}. Skipping save operation.")
        return

    data_array = np.array(data)

    np.savez(save_path, data=data_array.astype(np.float32), labels=labels)
    logging.info(f"Successfully saved {len(data_array)} instances to: {save_path}")

def extract_features2d(parsed_traces, input_path, output_path): 
    ensure_dir(output_path)

    logging.info(f"[*] Extracting feature: original")

    data = []
    labels = []
    for trace_info in tqdm(parsed_traces, desc=f"Extracting original"):
        label = trace_info['label']
        traces = trace_info['data']

        traces_array = extract_to_ndarray2d(traces)
        if traces_array is not None:
            data.append(traces_array)
            labels.append(label)
    
    ensure_dir(output_path)
    save_npz2d(data, labels, os.path.join(output_path, f"bigenough_original_all.npz"))

    logging.info("[+] Feature extraction completed.")

if __name__ == "__main__":
    args = parse_arguments()
    parsed_traces = parse_trace(args.input_path)
    if parsed_traces:
        extract_features2d(parsed_traces, args.input_path, args.output_path)
        extract_features_for_group(parsed_traces, args.input_path, args.output_path)