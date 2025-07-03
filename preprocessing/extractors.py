def get_burst(traces):
    result = []
    for trace in traces:
        first = True
        feature = []
        for t, s in trace:
            dir = int(s/abs(s))
            if first:
                direction = dir
                burst = dir
                first = False
            elif dir == 0:
                break
            elif dir == direction:
                burst += dir
            else:
                feature.append(burst)
                burst = dir
                direction *= -1
        result.append(feature)
    return result

def get_direction(traces):
    # 각 trace의 두 번째 요소인 size 값을 양수/음수 방향으로 변환
    return [[int(s / abs(s)) for (t, s) in trace] for trace in traces]

def get_ipd(traces_list, clip_value=None, filter_last_10=None):
    ipds = []
    for traces in traces_list:
        ipd_values = [traces[i][0] - traces[i-1][0] for i in range(1, len(traces))]

        if clip_value is not None:
            ipd_values = [min(ipd, clip_value) for ipd in ipd_values]
        elif filter_last_10 is not None:
            sorted_ipds = sorted(ipd_values)
            threshold_index = int(len(sorted_ipds) * 0.1)
            ipd_values = sorted_ipds[threshold_index:]

        ipds.append(ipd_values)
    return ipds

def get_ipd_filtered(traces, size_threshold=80):
    ipd_features = []
    for trace in traces:
        filtered_trace = [pkt for pkt in trace if pkt[1] >= size_threshold]
        if len(filtered_trace) > 1:
            ipd = [filtered_trace[i][0] - filtered_trace[i - 1][0] for i in range(1, len(filtered_trace))]
        else:
            ipd = []
        ipd_features.append(ipd)
    
    return ipd_features

def get_cumulative_size(traces):
    result = []
    for trace in traces:
        cumulative = []
        total = 0
        for _, size in trace:
            total += int(size)
            cumulative.append(total)
        result.append(cumulative)
    return result

def get_size(traces, filter_small=None):
    # 각 trace의 두 번째 요소인 size 값만 추출
    sizes = [[int(s) for (t, s) in trace] for trace in traces]

    if filter_small:
        sorted_sizes = sorted(sizes)
        threshold_index = int(len(sorted_sizes) * 0.3)
        filtered_sizes = sorted_sizes[threshold_index:]
        return filtered_sizes
    return sizes

def get_1dtam(traces):
    max_matrix_len = 1800
    maximum_load_time = 10
    result = []
    for trace in traces:
        timestamps = [t for (t, s) in trace]
        dirs = [int(s/abs(s)) for (t, s) in trace]
        if timestamps:
            feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
            for i in range(0, len(dirs)):
                if dirs[i] > 0:
                    if timestamps[i] >= maximum_load_time:
                        feature[0][-1] += 1
                    else:
                        idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                        feature[0][idx] += 1
                if dirs[i] < 0:
                    if timestamps[i] >= maximum_load_time:
                        feature[1][-1] += 1
                    else:
                        idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                        feature[1][idx] += 1
            feature1 = feature[0]
            feature2 = feature[1]
        result.append(feature1 + feature2)
    return result

def get_upload_tam(traces):
    max_matrix_len = 1800
    maximum_load_time = 10
    result = []
    for trace in traces:
        timestamps = [t for (t, s) in trace]
        dirs = [int(s / abs(s)) for (t, s) in trace]
        if timestamps:
            feature = [0 for _ in range(max_matrix_len)]
            for i in range(len(dirs)):
                if dirs[i] > 0:  # 업로드 데이터만 처리
                    if timestamps[i] >= maximum_load_time:
                        feature[-1] += 1
                    else:
                        idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                        feature[idx] += 1
            result.append(feature)
    return result

def get_download_tam(traces):
    max_matrix_len = 1800
    maximum_load_time = 10
    result = []
    for trace in traces:
        timestamps = [t for (t, s) in trace]
        dirs = [int(s / abs(s)) for (t, s) in trace]
        if timestamps:
            feature = [0 for _ in range(max_matrix_len)]
            for i in range(len(dirs)):
                if dirs[i] < 0:  # 다운로드 데이터만 처리
                    if timestamps[i] >= maximum_load_time:
                        feature[-1] += 1
                    else:
                        idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                        feature[idx] += 1
            result.append(feature)
    return result

def get_tiktok_cumulative(traces):
    cumulative_tiktok = []
    for trace in traces:
        cumulative_time = np.cumsum([abs(t) for (t, s) in trace])  # time은 절대값으로 누적
        cumulative_tiktok.append([t_cum * (s / abs(s)) for t_cum, (t, s) in zip(cumulative_time, trace)])  # 누적된 time에 direction을 다시 붙임
    return cumulative_tiktok

def get_tiktok(traces):
    return [[t * int(s / abs(s)) for (t, s) in trace] for trace in traces]