import os
from natsort import natsorted, ns
from tqdm import tqdm
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
# from plot_result import preprocess_data # 더 이상 필요하지 않다면 제거
from utility import getDirection
import logging

logger = logging.getLogger(__name__)

def load_raw(dataset_dir, feature=None, n=75, length=5000):
    """
    지정된 디렉토리에서 원시 데이터를 로드하고 전처리합니다.

    Args:
        dataset_dir (str): 데이터셋이 있는 디렉토리 경로.
        feature (str, optional): 사용할 특징. 현재 코드에서는 'direction'을 가정합니다. 기본값은 None.
        n (int, optional): 로드할 클래스의 총 개수. (현재는 단순히 레이블 인코딩에 사용)
                           실제로 클래스를 필터링하는 로직은 추가되지 않음.
        length (int, optional): 모든 트레이스(시퀀스)의 길이를 통일할 길이.
                                이 길이보다 길면 잘리고, 짧으면 0으로 패딩됩니다.

    Returns:
        tuple: X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    all_files = [file for file in os.listdir(dataset_dir)]
    sorted_files = natsorted(all_files, alg=ns.IGNORECASE)

    X_data = []
    y_labels = []

    for file in tqdm(sorted_files, desc="Processing Files", unit="file"):
        file_path = os.path.join(dataset_dir, file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = getDirection(f) # utility.getDirection 함수 호출
                label = file.split('-')[0] # 파일 이름에서 레이블 추출

            # 데이터 길이 통일: 길면 자르고, 짧으면 0으로 패딩
            if len(data) > length:
                data = data[:length]
            elif len(data) < length:
                data = data + [0] * (length - len(data))

            X_data.append(np.array(data, dtype=np.float32))
            y_labels.append(label)
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            continue

    if not X_data:
        raise ValueError("No data loaded. Check dataset_dir and file contents.")

    X_data = np.array(X_data, dtype=np.float32) # 이제 모든 데이터의 길이가 통일되어 float32로 바로 변환 가능
    y_labels = np.array(y_labels, dtype=str) # 레이블은 문자열로 유지

    # 데이터 분할
    # train:test = 8:2, 그 다음 test를 valid:test = 1:1 로 다시 분할
    X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 레이블 인코딩 및 원-핫 인코딩
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    y_test_encoded = label_encoder.transform(y_test)

    # NB_CLASSES를 main.py에서 넘겨주는 n 값과 일치시키거나, len(label_encoder.classes_)를 사용
    # 여기서는 동적으로 클래스 수를 결정합니다.
    num_classes = len(label_encoder.classes_)
    if n != num_classes:
        logger.warning(f"Configured NB_CLASSES ({n}) does not match actual number of classes ({num_classes}). Using {num_classes}.")

    y_train = to_categorical(y_train_encoded, num_classes=num_classes)
    y_valid = to_categorical(y_valid_encoded, num_classes=num_classes)
    y_test = to_categorical(y_test_encoded, num_classes=num_classes)

    logger.info("Data dimensions:")
    logger.info(f"X: Training data's shape : {X_train.shape}")
    logger.info(f"y: Training data's shape : {y_train.shape}")
    logger.info(f"X: Validation data's shape : {X_valid.shape}")
    logger.info(f"y: Validation data's shape : {y_valid.shape}")
    logger.info(f"X: Testing data's shape : {X_test.shape}")
    logger.info(f"y: Testing data's shape : {y_test.shape}")

    # 가비지 컬렉션
    gc.collect()

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# Load data for non-defended dataset for CW setting
def load_pkl(feature, n=75):

    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = f'/scratch4/starlink/baseline/feature/{feature}/'

    # X represents a sequence of traffic directions
    
    # Load training datapip install scikit-learn.
    with open(dataset_dir + f'FS_X_{feature}.pkl', 'rb') as handle:
        X = pickle.load(handle, encoding='latin1')
        X = np.array(X, dtype=object)
        
    with open(dataset_dir + f'FS_Y_{feature}.pkl', 'rb') as handle:
        y = pickle.load(handle, encoding='latin1')
        y = np.array(y, dtype=object)

    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
    X_valid, X_test, y_valid, y_test=train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    # Convert data as float32 type
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_valid = y_valid.astype('float32')
    y_test = y_test.astype('float32')

    # Initialize the scaler
    scaler = StandardScaler()

    # # Flatten the data for scaling (scalers expect 2D input)
    X_train_flat = X_train.reshape(-1, X_train.shape[1])
    X_valid_flat = X_valid.reshape(-1, X_valid.shape[1])
    X_test_flat = X_test.reshape(-1, X_test.shape[1])

    # # Fit the scaler on training data and transform all sets
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_valid_scaled = scaler.transform(X_valid_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Reshape back to the original dimensions
    X_train = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_valid = X_valid_scaled.reshape(X_valid.shape[0], X_valid.shape[1], 1)
    X_test = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Convert class vectors to categorical classes matrices
    y_train = to_categorical(y_train, n)   # y_train = np_utils.to_categorical(y_train, n)
    y_valid = to_categorical(y_valid, n)
    y_test = to_categorical(y_test, n)

    gc.collect()
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_npz(dataset_npz, feature=None, scenario="ff_sl", n=75, apply_scaler=False):
    data = np.load(dataset_npz)
    X = data['data']
    y = data['labels']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 레이블 인코딩 및 원-핫 인코딩
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_valid_encoded = label_encoder.transform(y_valid)
    y_test_encoded = label_encoder.transform(y_test)

    # NB_CLASSES를 main.py에서 넘겨주는 n 값과 일치시키거나, len(label_encoder.classes_)를 사용
    # 여기서는 동적으로 클래스 수를 결정합니다.
    num_classes = len(label_encoder.classes_)
    if n != num_classes:
        logger.warning(f"Configured NB_CLASSES ({n}) does not match actual number of classes ({num_classes}). Using {num_classes}.")

    y_train = to_categorical(y_train_encoded, num_classes=num_classes)
    y_valid = to_categorical(y_valid_encoded, num_classes=num_classes)
    y_test = to_categorical(y_test_encoded, num_classes=num_classes)

    logger.info("Data dimensions:")
    logger.info(f"X: Training data's shape : {X_train.shape}")
    logger.info(f"y: Training data's shape : {y_train.shape}")
    logger.info(f"X: Validation data's shape : {X_valid.shape}")
    logger.info(f"y: Validation data's shape : {y_valid.shape}")
    logger.info(f"X: Testing data's shape : {X_test.shape}")
    logger.info(f"y: Testing data's shape : {y_test.shape}")

    # 가비지 컬렉션
    gc.collect()

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_npz_split(feature, dataset_dir, scenario="ff_sl", n=75, apply_scaler=False):
    tr_path = os.path.join(dataset_dir, f"{scenario}_{feature}_training_56inst.npz")
    val_path = os.path.join(dataset_dir, f"{scenario}_{feature}_valid_12inst.npz")
    te_path = os.path.join(dataset_dir, f"{scenario}_{feature}_testing_12inst.npz")
    
    train = np.load(tr_path)
    valid = np.load(val_path)
    test = np.load(te_path)

    X_train, y_train = train['data'], train['labels']
    X_valid, y_valid = valid['data'], valid['labels']
    X_test, y_test = test['data'], test['labels']

    print(np.unique(y_test))
    print(len(np.unique(y_test)))

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_valid = encoder.transform(y_valid)
    y_test = encoder.transform(y_test)

    # y_train = np_utils.to_categorical(y_train, len(encoder.classes_))
    # y_valid = np_utils.to_categorical(y_valid, len(encoder.classes_))
    # y_test = np_utils.to_categorical(y_test, len(encoder.classes_))

    if apply_scaler:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
    
    X_train, y_train = preprocess_data(X_train, y_train, n)
    X_valid, y_valid = preprocess_data(X_valid, y_valid, n)
    X_test, y_test = preprocess_data(X_test, y_test, n)

    print(y_train.shape, y_valid.shape, y_test.shape)
    return X_train, y_train, X_valid, y_valid, X_test, y_test