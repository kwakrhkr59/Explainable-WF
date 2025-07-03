import os
import matplotlib.pyplot as plt
from natsort import natsorted, ns
from tqdm import tqdm

ylim = (-1.5, 1.5)
zero_start = True
base_path = "/home/kwakrhkr59/XAI_WF/data/bigenough_tor_fiber"

nodef_dir = f"/scratch2/TrafficSliver/DeepCoAST/BE-original/mon_standard"
wtf_dir = f"/home/kwakrhkr59/XAI_WF/defense/wtfpad/results/bigenough_tor_fiber"
if zero_start:
    output_dir = f"{base_path}/data/combined_tor_fiber_{ylim[0]}_{ylim[1]}_adv_zero_bar_subplots/"
else:
    output_dir = f"{base_path}/data/combined_tor_fiber_{ylim[0]}_{ylim[1]}_adv_bar_subplots/"

def sign(value):
    """값이 0보다 크면 1, 0보다 작으면 -1, 0이면 0을 반환합니다."""
    return (value > 0) - (value < 0)

def read_file_data(file_path):
    """주어진 파일에서 타임스탬프와 값을 읽어옵니다. 첫 번째 타임스탬프를 0으로 정규화할 수 있습니다."""
    all_timestamps = []
    all_values = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        if not lines: # 파일이 비어있는 경우 처리
            print(f"경고: {file_path} 파일이 비어있습니다. 건너뜝니다.")
            return [], [], None, None # 빈 리스트와 None 반환

        # 파일 내의 첫 번째 실제 데이터 포인트를 찾아 start_time_offset을 결정합니다.
        first_data_point_time = None
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2: # 최소한 타임스탬프와 값이 있어야 함
                try:
                    first_data_point_time = float(parts[0])
                    break # 첫 번째 유효한 데이터 포인트를 찾으면 반복 중단
                except ValueError:
                    continue # 숫자가 아니면 다음 줄로
        
        if first_data_point_time is None:
            print(f"경고: {file_path}에서 유효한 첫 번째 타임스탬프를 찾을 수 없습니다. 0.0으로 기본 설정합니다.")
            start_time_offset = 0.0
        else:
            # zero_start가 True면 첫 번째 데이터 포인트를 0으로 맞추고, 아니면 0 오프셋
            start_time_offset = first_data_point_time if zero_start else 0.0

        # 데이터 파싱 및 정규화된 타임스탬프 저장
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2: # 타임스탬프와 값이 모두 존재하는지 확인
                try:
                    timestamp = float(parts[0])
                    value = float(parts[1])
                    all_timestamps.append(timestamp - start_time_offset)
                    all_values.append(sign(value))
                except ValueError:
                    continue # 숫자로 변환할 수 없으면 건너뛰기
    
    # xlim에 사용될 min/max 타임스탬프 계산 (정규화된 값)
    min_timestamp = min(all_timestamps) if all_timestamps else 0
    max_timestamp = max(all_timestamps) if all_timestamps else 0

    return all_timestamps, all_values, min_timestamp, max_timestamp

def plot_combined_waves_as_bars(nodef_dir, wtf_dir, output_dir):
    nodef_files = set(os.listdir(nodef_dir))
    wtf_files = set(os.listdir(wtf_dir))

    common_files = natsorted(list(nodef_files.intersection(wtf_files)), alg=ns.IGNORECASE)

    nodef_bar_width = 0.0005
    wtf_bar_width = 0.05

    for file_name in tqdm(common_files, desc="Processing Combined Files", unit="file"):
        nodef_file_path = os.path.join(nodef_dir, file_name)
        wtf_file_path = os.path.join(wtf_dir, file_name)

        # 두 파일의 데이터 및 각자의 타임스탬프 범위 가져오기
        nodef_timestamps, nodef_values, nodef_min_t, nodef_max_t = read_file_data(nodef_file_path)
        wtf_timestamps, wtf_values, wtf_min_t, wtf_max_t = read_file_data(wtf_file_path)

        # 두 파일 모두 유효한 데이터가 없으면 건너뛰기
        if not nodef_timestamps and not wtf_timestamps:
            print(f"경고: {file_name} (nodef 및 wtf) 파일에 유효한 데이터가 없습니다. 건너뜝니다.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')

        # nodef_path 데이터 그리기
        if nodef_timestamps: # 데이터가 있는 경우에만 그리기
            axes[0].bar(nodef_timestamps, nodef_values, width=nodef_bar_width, label=f"nodef: {file_name}", color='red')
            # nodef 데이터의 실제 범위로 xlim 설정
            axes[0].set_xlim((nodef_min_t, nodef_max_t))
        else:
            # 데이터가 없는 경우 기본 xlim 설정 (예: 0부터 1까지)
            axes[0].set_xlim((0, 1)) 
        axes[0].set_title(f"nodef: {file_name}")
        axes[0].set_ylim(ylim)
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

        # wtf_path 데이터 그리기
        if wtf_timestamps: # 데이터가 있는 경우에만 그리기
            axes[1].bar(wtf_timestamps, wtf_values, width=wtf_bar_width, label=f"wtf: {file_name}", color='blue')
            # wtf 데이터의 실제 범위로 xlim 설정
            axes[1].set_xlim((wtf_min_t, wtf_max_t))
        else:
            # 데이터가 없는 경우 기본 xlim 설정
            axes[1].set_xlim((0, 1))
        axes[1].set_title(f"wtf: {file_name}")
        axes[1].set_ylim(ylim)
        axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

        plt.tight_layout()

        save_folder = os.path.join(output_dir, file_name.split('-')[0])
        os.makedirs(save_folder, exist_ok=True)

        # 파일 이름에 'individual_xlim'을 추가하여 구분
        save_path = os.path.join(save_folder, f"combined_bar_individual_xlim_{file_name.split('-')[1]}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white', transparent=False)
        # plt.show()

        # if input("Press Enter to continue or type 'exit' to quit: ").strip().lower() == 'exit':
        #     break
        plt.close(fig)

plot_combined_waves_as_bars(nodef_dir, wtf_dir, output_dir)