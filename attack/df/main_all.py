import os
import numpy as np
import pandas as pd
import random
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import classification_report, top_k_accuracy_score
from keras import backend as K
from utility import load_npz, plot_corr, plot_history
from Model_Focal import DFNet

random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 설정
INPUT_PATH = f'/scratch4/starlink/filtered'
NB_EPOCH = 100
BATCH_SIZE = 128
VERBOSE = 1
LENGTH = 5000
OPTIMIZER = Adamax(learning_rate=0.002)
INPUT_SHAPE = (LENGTH, 1)


for class_dir in sorted(os.listdir(INPUT_PATH)):
    results = []
    class_path = os.path.join(INPUT_PATH, class_dir)
    if not os.path.isdir(class_path): continue
    NB_CLASSES = int(class_dir.replace("top", ""))

    for feature in sorted(os.listdir(class_path)):
        feature_path = os.path.join(class_path, feature)
        if not os.path.isdir(feature_path): continue

        print(f"\n=== Processing: {class_dir} / {feature} ===")

        # 데이터 불러오기
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_npz(feature, feature_path, scenario="ff_sl", n=NB_CLASSES, apply_scaler=True)
        K.set_image_data_format('channels_first')

        # 모델 빌드 및 학습
        model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
        model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=NB_EPOCH,
            verbose=VERBOSE,
            validation_data=(X_valid, y_valid)
        )

        # 평가
        score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
        y_pred = model.predict(X_test)
        y_true_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        # 메트릭 계산
        top2_acc = top_k_accuracy_score(y_true_labels, y_pred, k=2)
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True)

        metrics_dict = {
            "ClassDir": class_dir,
            "Feature": feature,
            "FinalTrainAcc": history.history["accuracy"][-1],
            "FinalValidAcc": history.history["val_accuracy"][-1],
            "TestAcc": score_test[1],
            "Top2Acc": top2_acc,
            "MacroPrecision": report["macro avg"]["precision"],
            "MacroRecall": report["macro avg"]["recall"],
            "MacroF1": report["macro avg"]["f1-score"],
            "MicroPrecision": report["weighted avg"]["precision"],
            "MicroRecall": report["weighted avg"]["recall"],
            "MicroF1": report["weighted avg"]["f1-score"],
        }

        # Epoch별 정확도도 저장
        for epoch in range(NB_EPOCH):
            metrics_dict[f"TrainAcc_E{epoch+1}"] = history.history["accuracy"][epoch]
            metrics_dict[f"ValidAcc_E{epoch+1}"] = history.history["val_accuracy"][epoch]

        results.append(metrics_dict)

        # 평가
        score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
        y_pred = model.predict(X_test)

        # 시각화
        plot_dir = f"plots/{class_dir}/{feature}"
        os.makedirs(plot_dir, exist_ok=True)

        heatmap_path = os.path.join(plot_dir, f"heatmap_class{NB_CLASSES}_{feature}.png")
        history_path = os.path.join(plot_dir, f"history_class{NB_CLASSES}_{feature}.png")

        plot_corr(y_test, y_pred, save_path=heatmap_path)
        plot_history(history, score_test, save_path=history_path)

    # 결과 저장
    results_df = pd.DataFrame(results)
    os.makedirs("summary_results", exist_ok=True)
    results_df.to_csv(f"summary_results/df_top{NB_CLASSES}_results.csv", index=False)

print("\n✅ 모든 조합에 대한 결과 저장 완료: summary_results/df_all_results.csv")
