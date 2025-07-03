import os
import matplotlib.pyplot as plt
from natsort import natsorted, ns
from tqdm import tqdm

xlim = (0, 10)
ylim = (-3000, 300)
zero_start = True
base_path = "/home/kwakrhkr59/XAI_WF/data/firefox_fiber"

def read_file_data(file_path):
    """주어진 파일에서 타임스탬프와 값을 읽어옵니다."""
    all_timestamps = []
    all_values = []
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line:
            try:
                start_time = float(first_line[0]) if zero_start else 0.0
            except (ValueError, IndexError):
                print(f"경고: {file_path}에서 start_time을 파싱할 수 없습니다. 0.0으로 기본 설정합니다.")
                start_time = 0.0
        else:
            start_time = 0.0

        for line in f:
            parts = list(map(float, line.split()))
            all_timestamps.append(parts[0] - start_time)
            all_values.append(parts[1])
    return all_timestamps, all_values

def plot_combined_waves_as_bars(nodef_dir, wtf_dir, output_base_path):
    nodef_files = set(os.listdir(nodef_dir))
    wtf_files = set(os.listdir(wtf_dir))

    common_files = natsorted(list(nodef_files.intersection(wtf_files)), alg=ns.IGNORECASE)

    # 막대 너비 설정. 데이터의 시간 해상도에 따라 조정해야 할 수 있습니다.
    bar_width = 0.05 

    for file_name in tqdm(common_files, desc="Processing Combined Files", unit="file"):
        nodef_file_path = os.path.join(nodef_dir, file_name)
        wtf_file_path = os.path.join(wtf_dir, file_name)

        # 두 파일의 데이터 읽기
        nodef_timestamps, nodef_values = read_file_data(nodef_file_path)
        wtf_timestamps, wtf_values = read_file_data(wtf_file_path)

        # 두 개의 서브플롯을 가진 Figure 생성
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='white') # 1행, 2열

        # nodef_path 데이터를 막대 그래프로 그리기
        axes[0].bar(nodef_timestamps, nodef_values, width=bar_width, label=f"nodef: {file_name}", color='black')
        axes[0].set_title(f"nodef: {file_name}")
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8) # y=0에 수평선 추가

        # wtf_path 데이터를 막대 그래프로 그리기
        axes[1].bar(wtf_timestamps, wtf_values, width=bar_width, label=f"wtf: {file_name}", color='black')
        axes[1].set_title(f"wtf: {file_name}")
        axes[1].set_xlim(xlim)
        axes[1].set_ylim(ylim)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8) # y=0에 수평선 추가

        plt.tight_layout() # 레이아웃 조정하여 제목/레이블 겹침 방지

        save_folder = os.path.join(output_base_path, file_name.split('-')[0])
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, f"combined_bar_{file_name.split('-')[1]}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white', transparent=False)
        plt.close(fig) # 메모리 확보를 위해 Figure 닫기


def main(nodef_dir, wtf_dir, output_dir):
    plot_combined_waves_as_bars(nodef_dir, wtf_dir, output_dir)


if __name__ == "__main__":
    nodef_dir = f"/scratch4/starlink/WFdata75x80/firefox_fiber"
    wtf_dir = f"/home/kwakrhkr59/XAI_WF/defense/wtfpad/results/default_250626_163519"
    if zero_start:
        output_dir = f"{base_path}/data/combined_firefox_fiber_{ylim[0]}_{ylim[1]}_{xlim[1]}_zero_bar_subplots/"
    else:
        output_dir = f"{base_path}/data/combined_firefox_fiber_{ylim[0]}_{ylim[1]}_{xlim[1]}_bar_subplots/"
    main(nodef_dir, wtf_dir, output_dir)