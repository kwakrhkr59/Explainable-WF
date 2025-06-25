import os
import matplotlib.pyplot as plt
from natsort import natsorted, ns
from tqdm import tqdm

xlim = (0, 10)
ylim = (-3000, 300)
zero_start = True
base_path = "/home/kwakrhkr59/XAI_WF/data/firefox_fiber"

def plot_wave_from_files(directory_path, output_base_path):
    all_files = [file for file in os.listdir(directory_path)]
    sorted_files = natsorted(all_files, alg=ns.IGNORECASE)

    for file in tqdm(sorted_files, desc="Processing Files", unit="file"):
        all_timestamps = []
        all_values = []

        file_path = os.path.join(directory_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            start_time = float(f.readline().strip()[0]) if zero_start else 0.0
            for line in f:
                parts = list(map(float, line.split()))
                all_timestamps.append(parts[0] - start_time)
                all_values.append(parts[1])

        save_folder = output_base_path + file.split('-')[0]
        os.makedirs(save_folder, exist_ok=True)

        plt.figure(facecolor='white') 
        plt.plot(all_timestamps, all_values, label=file, linewidth=1.5, color='black')  # connected line plot
        # plt.bar(all_timestamps, all_values, width=0.05, label=file)       # discrete bar plot
        # plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)    # horizontal line at y=0
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks([])
        plt.yticks([])

        save_path = os.path.join(save_folder, f"{file.split('-')[1]}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white', transparent=False)
        plt.close()

def main(input_dir, output_dir):
    plot_wave_from_files(input_dir, output_dir)


if __name__ == "__main__":
    input_dir = f"/scratch4/starlink/WFdata75x80/firefox_fiber"
    if zero_start: output_dir = f"{base_path}/data/firefox_fiber_{ylim[0]}_{ylim[1]}_{xlim[1]}_zero/"
    else: output_dir = f"{base_path}/data/firefox_fiber_{ylim[0]}_{ylim[1]}_{xlim[1]}/"
    main(input_dir, output_dir)
    