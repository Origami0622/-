import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

from conv import *
from KAN import *

def plot_instrument_metrics(y_true, y_pred, instrument_names):
    # 计算每个乐器的precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=instrument_names)

    # 创建数据框以便于绘制
    metrics_df = pd.DataFrame({
        'Instrument': instrument_names,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    # 设置图形大小
    plt.figure(figsize=(14, 10))

    # 绘制柱状图
    for i, metric in enumerate(['Precision', 'Recall', 'F1 Score']):
        plt.subplot(3, 1, i + 1)
        sns.barplot(x='Instrument', y=metric, data=metrics_df)
        plt.title(f'{metric} per Instrument')
        plt.ylim(0, 1)  # 设置y轴范围从0到1
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# =============== 数据准备 ===============
wavelet_types = ['mexican_hat']
all_results = {}
instrument_names = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

# 类别样本数（用于类别不平衡）
sample_counts = np.array([776, 1008, 902, 1274, 1520, 1364, 1442, 1252, 1154, 1160, 1556])
sample_kinds_counts = 11
total_samples = sample_counts.sum()
frequencies = sample_counts / total_samples
weights = 1 / frequencies
weights_normalized = weights / weights.sum()

# 加载两组特征（CNN / KAN）(这俩的标签是完全一致的，这里没必要分别提出了)
features_file1 = 'npy_data/X/all_trainX.npy'   # 给 KAN
labels_file1 = 'npy_data/y/all_trainY.npy'     # 给 KAN
features1 = np.load(features_file1, allow_pickle=True)
labels1 = np.load(labels_file1, allow_pickle=True)

features_file2 = 'npy_data/X/mfccX_1s.npy'     # 给 CNN
labels_file2 = 'npy_data/y/mfccY_1s.npy'       # 给 CNN
features2 = np.load(features_file2, allow_pickle=True)
labels2 = np.load(labels_file2, allow_pickle=True)

# 获取输入维度
width_cnn, height_cnn = features2.shape[1], features2.shape[2]
D_kan = features1.shape[1] * features1.shape[2]

print(f"CNN input shape: {width_cnn} x {height_cnn}")
print(f"KAN input dimension: {D_kan}")
print(f"mfcc length: {len(features2)}")
print(f"all length: {len(features1)}")

assert len(features1) == len(features2), "Error: Two datasets have different number of samples!"
assert len(labels1) == len(labels2), "Error: Two label sets have different number of samples!"

# =============== 统一 Dataset（CNN + KAN） ===============
class DualNPYDataset(Dataset):
    def __init__(self, features_cnn, features_kan, labels, transform=None):
        self.features_cnn = features_cnn
        self.features_kan = features_kan
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature_cnn = self.features_cnn[idx]
        feature_kan = self.features_kan[idx]
        label = self.labels[idx]

        feature_cnn = feature_cnn.astype(float)
        feature_kan = feature_kan.astype(float)
        feature_cnn_tensor = torch.tensor(feature_cnn, dtype=torch.float)
        feature_kan_tensor = torch.tensor(feature_kan, dtype=torch.float)

        label = label.astype(float)
        label = torch.tensor(label, dtype=torch.float)

        return feature_cnn_tensor, feature_kan_tensor, label


# =============== KFold 划分 ===============
n_splits = 2
epochs_per_trial = 50
batch_size = 32
skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

X_train_cnn, X_test_cnn = [], []
y_train_cnn, y_test_cnn = [], []
X_train_kan, X_test_kan = [], []
y_train_kan, y_test_kan = [], []

for train_index, test_index in skf.split(features2, labels2):
    X_train_cnn.append(features2[train_index])
    X_test_cnn.append(features2[test_index])
    y_train_cnn.append(labels2[train_index])
    y_test_cnn.append(labels2[test_index])

    X_train_kan.append(features1[train_index])
    X_test_kan.append(features1[test_index])
    y_train_kan.append(labels1[train_index])
    y_test_kan.append(labels1[test_index])


# =============== 主训练循环 ===============
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

for wavelet in wavelet_types:
    all_train_losses, all_train_accuracies = [], []
    all_val_losses, all_val_accuracies = [], []
    all_val_f1_micro, all_val_f1_macro = [], []
    all_lraps = []
    all_recall_macro, all_precision_macro = [], []
    all_recall_micro, all_precision_micro = [], []

    print(f'Wavelet is {wavelet}')

    for i in range(n_splits):
        print(f'KFold {i+1}/{n_splits}')

        # Dataset & DataLoader(这俩的标签是完全一致的，用哪个都行)
        trainset = DualNPYDataset(X_train_cnn[i], X_train_kan[i], y_train_cnn[i])
        testset = DualNPYDataset(X_test_cnn[i], X_test_kan[i], y_test_cnn[i])

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)

        # 模型
        cnn_model = ConvNet(sample_kinds_counts)
        kan_model = KAN([D_kan, 1024, 512, 256, 64, 32, sample_kinds_counts], wavelet_type=wavelet)
        cnn_model.to(device)
        kan_model.to(device)

        # 优化器
        optimizer_cnn = optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer_kan = optim.AdamW(kan_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler_cnn = optim.lr_scheduler.ExponentialLR(optimizer_cnn, gamma=0.9)
        scheduler_kan = optim.lr_scheduler.ExponentialLR(optimizer_kan, gamma=0.9)

        # 损失函数
        class_weights = torch.tensor(weights_normalized, device=device)
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        trial_train_losses, trial_val_losses = [], []
        trial_train_accuracies, trial_val_accuracies = [], []
        trial_f1_micro, trial_f1_macro = [], []
        trial_lrap, trial_recall_macro, trial_precision_macro = [], [], []
        trial_recall_micro, trial_precision_micro = [], []

        # ========== 训练循环 ==========
        for epoch in range(epochs_per_trial):
            cnn_model.train()
            kan_model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for images_cnn_raw, images_kan_raw, labels in tqdm(trainloader):
                images_cnn = images_cnn_raw.view(-1, width_cnn, height_cnn).unsqueeze(1).to(device)
                images_kan = images_kan_raw.view(-1, D_kan).to(device)
                labels = labels.to(device)

                # CNN forward
                outputs_cnn = cnn_model(images_cnn)
                loss_cnn = criterion(outputs_cnn, labels)
                weighted_loss_cnn = (loss_cnn * class_weights[None, :]).sum(dim=1).mean()

                # KAN forward
                outputs_kan = kan_model(images_kan)
                loss_kan = criterion(outputs_kan, labels)
                weighted_loss_kan = (loss_kan * class_weights[None, :]).sum(dim=1).mean()

                # 总损失
                total_loss = 0.4 * weighted_loss_cnn + 0.6 * weighted_loss_kan

                optimizer_cnn.zero_grad()
                optimizer_kan.zero_grad()
                total_loss.backward()
                optimizer_cnn.step()
                optimizer_kan.step()

                train_loss += total_loss.item()

                # 融合预测
                probs_cnn = torch.sigmoid(outputs_cnn)
                probs_kan = torch.sigmoid(outputs_kan)
                probs_final = 0.4 * probs_cnn + 0.6 * probs_kan
                predicted = (probs_final >= 0.5).float()

                train_total += labels.numel()
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(trainloader)
            train_acc = 100 * train_correct / train_total
            trial_train_losses.append(train_loss)
            trial_train_accuracies.append(train_acc)

            # ========== 验证 ==========
            cnn_model.eval()
            kan_model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_labels, all_preds_ori = [], [], []

            with torch.no_grad():
                for images_cnn_raw, images_kan_raw, labels in tqdm(valloader, leave=False):
                    images_cnn = images_cnn_raw.view(-1, width_cnn, height_cnn).unsqueeze(1).to(device)
                    images_kan = images_kan_raw.view(-1, D_kan).to(device)
                    labels = labels.to(device)

                    outputs_cnn = cnn_model(images_cnn)
                    outputs_kan = kan_model(images_kan)

                    loss_cnn = criterion(outputs_cnn, labels)
                    loss_kan = criterion(outputs_kan, labels)
                    weighted_loss_cnn = (loss_cnn * class_weights[None, :]).sum(dim=1).mean()
                    weighted_loss_kan = (loss_kan * class_weights[None, :]).sum(dim=1).mean()
                    total_loss = 0.4 * weighted_loss_cnn + 0.6 * weighted_loss_kan

                    val_loss += total_loss.item()

                    probs_cnn = torch.sigmoid(outputs_cnn)
                    probs_kan = torch.sigmoid(outputs_kan)
                    probs_final = 0.4 * probs_cnn + 0.6 * probs_kan
                    predicted = (probs_final >= 0.5).float()

                    val_total += labels.numel()
                    val_correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_preds_ori.extend(probs_final.cpu().numpy())

            val_loss /= len(valloader)
            val_acc = 100 * val_correct / val_total
            trial_val_losses.append(val_loss)
            trial_val_accuracies.append(val_acc)

            # 计算指标
            # all_labels_arr = np.array(all_labels)
            # all_preds_arr = np.array(all_preds)
            all_preds_ori_arr = np.array(all_preds_ori)

            all_labels_arr = np.vstack(all_labels)
            all_preds_arr = np.vstack(all_preds)

            f1_micro = f1_score(all_labels_arr, all_preds_arr, average='micro', zero_division=0)
            f1_macro = f1_score(all_labels_arr, all_preds_arr, average='macro', zero_division=0)
            recall_macro = recall_score(all_labels_arr, all_preds_arr, average='macro', zero_division=0)
            precision_macro = precision_score(all_labels_arr, all_preds_arr, average='macro', zero_division=0)
            recall_micro = recall_score(all_labels_arr, all_preds_arr, average='micro', zero_division=0)
            precision_micro = precision_score(all_labels_arr, all_preds_arr, average='micro', zero_division=0)
            lraps = label_ranking_average_precision_score(all_labels_arr, all_preds_ori_arr)

            # 绘制逐个乐器识别效果图像
            plot_instrument_metrics(all_labels_arr, all_preds_arr, instrument_names)


            trial_f1_micro.append(f1_micro)
            trial_f1_macro.append(f1_macro)
            trial_recall_macro.append(recall_macro)
            trial_precision_macro.append(precision_macro)
            trial_recall_micro.append(recall_micro)
            trial_precision_micro.append(precision_micro)
            trial_lrap.append(lraps)

            scheduler_cnn.step()
            scheduler_kan.step()

        # 收集结果
        all_train_losses.append(trial_train_losses)
        all_train_accuracies.append(trial_train_accuracies)
        all_val_losses.append(trial_val_losses)
        all_val_accuracies.append(trial_val_accuracies)
        all_val_f1_micro.append(trial_f1_micro)
        all_val_f1_macro.append(trial_f1_macro)
        all_recall_macro.append(trial_recall_macro)
        all_precision_macro.append(trial_precision_macro)
        all_recall_micro.append(trial_recall_micro)
        all_precision_micro.append(trial_precision_micro)
        all_lraps.append(trial_lrap)

    # 保存模型（最后一个 fold 的模型）
    torch.save({'cnn': cnn_model.state_dict(),
                'kan': kan_model.state_dict()},
               f'ensemble_all_cnn_kan_fixed.pth')

    # 保存结果
    results_df = pd.DataFrame({
        'Epoch': range(1, epochs_per_trial + 1),
        'Train Loss': pd.DataFrame(all_train_losses).mean().tolist(),
        'Train Accuracy': pd.DataFrame(all_train_accuracies).mean().tolist(),
        'Test Loss': pd.DataFrame(all_val_losses).mean().tolist(),
        'Test Accuracy': pd.DataFrame(all_val_accuracies).mean().tolist(),
        'Test F1_micro': pd.DataFrame(all_val_f1_micro).mean().tolist(),
        'Test F1_macro': pd.DataFrame(all_val_f1_macro).mean().tolist(),
        'Test Recall_micro': pd.DataFrame(all_recall_micro).mean().tolist(),
        'Test Precision_micro': pd.DataFrame(all_precision_micro).mean().tolist(),
        'Test Recall_macro': pd.DataFrame(all_recall_macro).mean().tolist(),
        'Test Precision_macro': pd.DataFrame(all_precision_macro).mean().tolist(),
        'Test LRAP': pd.DataFrame(all_lraps).mean().tolist()
    })
    outname = f'ensemble_all_results_fixed.xlsx'
    results_df.to_excel(outname, index=False)
    print("Results saved to", outname)


# =============== 可视化 ===============
plt.figure(figsize=(12, 6))
plt.plot(results_df['Train Loss'], label='Train Loss')
plt.plot(results_df['Test Loss'], label='Test Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(results_df['Train Accuracy'], label='Train Acc')
plt.plot(results_df['Test Accuracy'], label='Test Acc')
plt.legend()
plt.show()
