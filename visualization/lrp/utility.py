def getData(line):
    timestamp, size = map(float, line.split('\t'))
    dir = int(abs(size) / size)
    size = abs(size)
    return timestamp, dir, size

def getCount(instance):
    return len(instance)

def rebase(instance):
    dirs = []
    start_time = getData(instance[0])[0]    
    for line in map(str.strip, instance):
        timestamp, dir, size = getData(line)
        dirs.append([timestamp - start_time, dir * size])
    return dirs

def getSize(instance):    
    dirs = []
    for line in map(str.strip, instance):
        _, _, size = getData(line)
        dirs.append(size)
    return dirs

def getDirection(instance):
    dirs = []
    for line in map(str.strip, instance):
        _, dir, _ = getData(line)
        dirs.append(dir)
    return dirs

def getTiktok(instance):
    feature = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = getData(line)
        feature.append(timestamp * dir)
    return feature

def get1DTAM(instance):
    max_matrix_len = 1800
    maximum_load_time = 80
    timestamps = []
    dirs = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = getData(line)
        timestamps.append(timestamp)
        dirs.append(dir)
    if timestamps:
        data = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
        for i in range(0, len(dirs)):
            if dirs[i] > 0:
                if timestamps[i] >= maximum_load_time:
                    data[0][-1] += 1
                else:
                    idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                    data[0][idx] += 1
            if dirs[i] < 0:
                if timestamps[i] >= maximum_load_time:
                    data[1][-1] += 1
                else:
                    idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                    data[1][idx] += 1
        return data[0] + data[1]
    return []

def getIPD(instance):
    timestamps = []
    dirs = []
    first = True
    for line in map(str.strip, instance):
        timestamp, dir, _ = getData(line)
        if first == True:
            first = False
            timestamps.append(0)
        else:
            timestamps.append(timestamp-beforetimestamp)
        dirs.append(dir)
        beforetimestamp = timestamp
    return timestamps