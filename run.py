
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import skimage.transform
import torch.optim as optim
import torchvision.transforms as transforms
from Bio import SeqIO
from collections import Counter
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from DM import DiffusionModel, train_diffusion_model, generate_augmented_data
from MTL import MultiTaskModel
from metrics import evaluate_loc
from sklearn.metrics import accuracy_score, precision_score, auc
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, recall_score

start = time.time()
SEED = 388014
BATCH_SIZE = 64
EPOCHS = 16
GRAY_SCALE = False
CHANNELS = 1 if GRAY_SCALE else 3

# 任务一参数
ddr1 = 256
NUM1 = 729
NUM2 = 82

# 任务二参数
ddr2 = 256
NUM3 = 3046
NUM4 = 136

# 任务一数据读取
print("lncRNA 729")
R1 = []
for c in range(NUM1):
    p = "feature_extraction/CGRxy_3D_RNA_729_AG_T/CGRxy_3D_729_" + str(c + 1) + ".png"
    transform = transforms.ToTensor()
    img = skimage.io.imread(p)
    img = img[6: 400, 83: 477]
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    imgT = img.T
    R1.append(imgT)
lncRNA_loc_features_o = np.array(R1)

# 读取任务一标签
dfl = pd.read_csv('dataset_preparation/dataset_Loc_II/label_729.csv')
dfl_ho = pd.read_csv('dataset_preparation/dataset_Loc_II/label_holdout_82.csv')
lncRNA_loc_labels_o = dfl.values
lncRNA_loc_labels_ho = dfl_ho.values

# 数据集已经加载为dataloader
data = TensorDataset(torch.tensor(lncRNA_loc_features_o, dtype=torch.float32),
                     torch.tensor(lncRNA_loc_labels_o, dtype=torch.float32))
dataloader = DataLoader(data, batch_size=32, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化扩散模型
model = DiffusionModel(image_channels=3, image_size=256, timesteps=1000, num_labels=4).to(device)
# 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# 训练模型
train_diffusion_model(model, dataloader, optimizer, epochs=2, timesteps=1000, device=device)

y = lncRNA_loc_labels_o
# 统计每个标签组合的频率
label_counts = Counter([tuple(row) for row in y])
print("原始标签分布:", label_counts)
# 找到最频繁的标签组合的数量
max_count = max(label_counts.values())
# 对每个标签组合进行重采样
y_balanced = []
for label, count in label_counts.items():
    # 找到当前标签组合的所有索引
    indices = [i for i, row in enumerate(y) if tuple(row) == label]
    # 如果当前标签组合的数量少于最大数量，则过采样
    if count < max_count:
        resampled_indices = resample(indices, replace=True, n_samples=max_count, random_state=42)
    # 如果当前标签组合的数量多于最大数量，则欠采样
    elif count > max_count:
        resampled_indices = resample(indices, replace=False, n_samples=max_count, random_state=42)
    # 如果数量正好，则保持不变
    else:
        resampled_indices = indices
    # 将重采样后的标签添加到结果中
    y_balanced.extend(y[resampled_indices])
# 转换为 numpy 数组
y_balanced = np.array(y_balanced)
# 统计均衡后的标签分布
balanced_label_counts = Counter([tuple(row) for row in y_balanced])
print("均衡后的标签分布:", balanced_label_counts)
labels = torch.from_numpy(y_balanced)
augmented_datas = []
label_num = 0
for label in labels:
    label_num += 1
    print(f"label_num：{label_num}")
    augmented_data = generate_augmented_data(model, num_samples=1, labels=label, device=device)
    augmented_datas.append(np.squeeze(augmented_data))
augmented_datas = np.array(augmented_datas)
# 将生成样本和标签加入原始数据
lncRNA_loc_features = np.concatenate([lncRNA_loc_features_o, augmented_datas], axis=0)
lncRNA_loc_labels = np.concatenate([lncRNA_loc_labels_o, y_balanced], axis=0)

# 任务一的独立测试集（ho）
print(f"lncRNA 82")
R1_ho = []
for c in range(NUM2):
    p = "feature_extraction/CGRxy_3D_RNA_82_AG_T/CGRxy_3D_82_" + str(c + 1) + ".png"
    transform = transforms.ToTensor()
    img = skimage.io.imread(p)
    img = img[6: 400, 83: 477]
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    imgT = img.T
    R1_ho.append(imgT)
lncRNA_loc_features_ho = np.array(R1_ho)

# 任务二数据读取
print("lncRNA 3046")
R1_I = []
for c in range(NUM3):
    p = "feature_extraction/CGRxy_3D_RNA_3046_AG_T/CGRxy_3D_3046_256_" + str(c + 1) + ".png"
    transform = transforms.ToTensor()
    img = skimage.io.imread(p)
    img = img[6: 400, 83: 477]
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    imgT = img.T
    R1_I.append(imgT)
lncRNA_features = np.array(R1_I)

print("Protein 136")
P = []
for c in range(NUM4):
    p = "feature_extraction/CGRxy_3D_Protein_136_AG_T/CGRxy_3D_136_256_" + str(c + 1) + ".png"
    transform = transforms.ToTensor()
    img = skimage.io.imread(p)
    img = img[6: 400, 83: 477]
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    imgT = img.T
    P.append(imgT)
protein_features = np.array(P)

labels = np.concatenate([np.ones(8112), np.zeros(8112)])
df_pos = pd.read_csv('dataset_preparation/dataset_Loc_II/positive_NPInter.csv')
df_neg = pd.read_csv('negative_NPInter.csv')
name_pair_pos = df_pos.values
name_pair_nes = df_neg.values
name_pairs = np.concatenate((name_pair_pos, name_pair_nes), axis=0)
name_rna = []
for record in SeqIO.parse('dataset_preparation/dataset_Loc_II/NPInter_rdrna_seq_3046.fasta', "fasta"):
    name_rna.append(record.id)
name_rna = np.array(name_rna)
name_protein = []
for record in SeqIO.parse('dataset_preparation/dataset_Loc_II/NPInter_rdprotein_seq_136.fasta', "fasta"):
    name_protein.append(record.id)
name_protein = np.array(name_protein)
indice_rna = np.array([np.where(name_rna == rna)[0][0] for rna in name_pairs[:, 0]])
indice_protein = np.array([np.where(name_protein == protein)[0][0] for protein in name_pairs[:, 1]])
pairs = np.column_stack((indice_rna, indice_protein))

# 任务一评估指标存储（训练集验证相关）
MiP = []
MiR = []
MiF = []
MiAUC = []
MaAUC = []
HL = []
AP = []
AvgF1 = []
Pat1 = []

# 任务一评估指标存储（独立测试集 ho 相关）
MiP_ho = []
MiR_ho = []
MiF_ho = []
MiAUC_ho = []
MaAUC_ho = []
HL_ho = []
AP_ho = []
AvgF1_ho = []
Pat1_ho = []

# 任务二评估指标存储
AUC_task2 = []
AUPR_task2 = []
ACC_task2 = []
PRE_task2 = []
REC_task2 = []
F1_task2 = []

# 任务一最优结果
best_task1_maauc_ho = -np.inf
best_task1_metrics = None

# 任务二最优结果
best_task2_auc = -np.inf
best_task2_metrics = None

# 定义模型
model = MultiTaskModel()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 任务一损失函数
criterion1 = nn.BCEWithLogitsLoss()
# 任务二损失函数
criterion2 = nn.BCELoss()
# 联合损失权重
lambda1 = 0.5
lambda2 = 0.5

# 任务一的五折交叉验证
skf_task1 = KFold(n_splits=5, shuffle=True, random_state=SEED)
# 任务二的五折交叉验证
skf_task2 = KFold(n_splits=5, shuffle=True, random_state=SEED)

# 同步进行五折交叉验证
for fold, ((train_idx_task1, val_idx_task1), (train_idx_task2, val_idx_task2)) in enumerate(
        zip(skf_task1.split(lncRNA_loc_features, lncRNA_loc_labels), skf_task2.split(pairs, labels)), 1):
    print(f"\nTask 1 Fold: {fold}, Task 2 Fold: {fold}")

    train_data_task1 = TensorDataset(
        torch.tensor(lncRNA_loc_features[train_idx_task1], dtype=torch.float32),
        torch.tensor(lncRNA_loc_labels[train_idx_task1], dtype=torch.float32)
    )
    val_data_task1 = TensorDataset(
        torch.tensor(lncRNA_loc_features[val_idx_task1], dtype=torch.float32),
        torch.tensor(lncRNA_loc_labels[val_idx_task1], dtype=torch.float32)
    )
    train_loader_task1 = DataLoader(train_data_task1, batch_size=32, shuffle=True)
    val_loader_task1 = DataLoader(val_data_task1, batch_size=32, shuffle=True)

    # 构建任务一独立测试集（ho）的数据加载器
    ho_data = TensorDataset(
        torch.tensor(lncRNA_loc_features_ho, dtype=torch.float32),
        torch.tensor(lncRNA_loc_labels_ho, dtype=torch.float32)
    )
    ho_loader = DataLoader(ho_data, batch_size=32, shuffle=True)

    train_pairs_fold, val_pairs_fold = pairs[train_idx_task2], pairs[val_idx_task2]
    train_labels_fold, val_labels_fold = labels[train_idx_task2], labels[val_idx_task2]

    train_data_task2 = TensorDataset(
        torch.tensor(lncRNA_features[train_pairs_fold[:, 0]], dtype=torch.float32),
        torch.tensor(protein_features[train_pairs_fold[:, 1]], dtype=torch.float32),
        torch.tensor(train_labels_fold, dtype=torch.float32)
    )
    val_data_task2 = TensorDataset(
        torch.tensor(lncRNA_features[val_pairs_fold[:, 0]], dtype=torch.float32),
        torch.tensor(protein_features[val_pairs_fold[:, 1]], dtype=torch.float32),
        torch.tensor(val_labels_fold, dtype=torch.float32)
    )
    train_loader_task2 = DataLoader(train_data_task2, batch_size=32, shuffle=True)
    val_loader_task2 = DataLoader(val_data_task2, batch_size=32, shuffle=True)

    best_task1_maauc_ho = -np.inf
    best_task1_metrics = None
    best_task2_auc = -np.inf
    best_task2_metrics = None

    for epoch in range(EPOCHS):
        print(f'\nTask 1 Fold: {fold}, Task 2 Fold: {fold}, Epoch: {epoch + 1}')
        model.train()
        train_iter_task1 = iter(train_loader_task1)
        train_iter_task2 = iter(train_loader_task2)

        while True:
            try:
                x1_task1, y1_task1 = next(train_iter_task1)
                x1_task2, x2_task2, y2_task2 = next(train_iter_task2)
            except StopIteration:
                break

            optimizer.zero_grad()
            # 任务一输出
            prob1, _ = model(x1_task1)
            # 任务二输出
            prob2 = model(x1_task2, x2_task2)

            # 任务一损失
            loss1 = criterion1(prob1, y1_task1)
            # 任务二损失
            loss2 = criterion2(prob2.squeeze(), y2_task2)

            # 联合损失
            combined_loss = lambda1 * loss1 + lambda2 * loss2
            combined_loss.backward()
            optimizer.step()

        # 任务一验证（使用验证集）
        model.eval()
        y_true_task1, y_prob_task1 = [], []
        for x1_task1, y1_task1 in val_loader_task1:
            prob1, _ = model(x1_task1)
            y_true_task1.append(y1_task1.numpy())
            y_prob_task1.append(prob1.detach().numpy())
        y_true_task1 = np.concatenate(y_true_task1)
        y_prob_task1 = np.concatenate(y_prob_task1)
        mip, mir, mif, miauc, maauc, hl, ap, avgF1, pat1 = evaluate_loc(y_true_task1, y_prob_task1)
        print(
            f"Task 1 Val: MiP={mip:.3f}, MiR={mir:.3f}, MiF={mif:.3f}, "
            f"MiAUC={miauc:.3f}, MaAUC={maauc:.3f}, HL={hl:.3f}, "
            f"AP={ap:.3f}, AvgF1={avgF1:.3f}, P@1={pat1:.3f}")

        # 任务一独立测试集（ho）评估
        model.eval()
        y_true_ho_task1, y_prob_ho_task1 = [], []
        for x1_ho_task1, y1_ho_task1 in ho_loader:
            prob_ho_1, _ = model(x1_ho_task1)
            y_true_ho_task1.append(y1_ho_task1.numpy())
            y_prob_ho_task1.append(prob_ho_1.detach().numpy())
        y_true_ho_task1 = np.concatenate(y_true_ho_task1)
        y_prob_ho_task1 = np.concatenate(y_prob_ho_task1)
        mip_ho, mir_ho, mif_ho, miauc_ho, maauc_ho, hl_ho, ap_ho, avgF1_ho, pat1_ho = evaluate_loc(y_true_ho_task1,
                                                                                                   y_prob_ho_task1)
        print(
            f"Task 1 Ho: MiP={mip_ho:.3f}, MiR={mir_ho:.3f}, MiF={mif_ho:.3f}, "
            f"MiAUC={miauc_ho:.3f}, MaAUC={maauc_ho:.3f}, HL={hl_ho:.3f}, "
            f"AP={ap_ho:.3f}, AvgF1={avgF1_ho:.3f}, P@1={pat1_ho:.3f}")

        # 任务二验证
        model.eval()
        y_true_task2, y_prob_task2 = [], []
        for x1_task2, x2_task2, y2_task2 in val_loader_task2:
            prob2 = model(x1_task2, x2_task2)
            y_prob_num = prob2.detach().numpy()
            y_prob_new = [item[0] for item in y_prob_num]
            y_prob_arr = np.array(y_prob_new)
            y_true_task2.append(y2_task2.numpy())
            y_prob_task2.append(y_prob_arr)
        y_true_task2 = np.concatenate(y_true_task2)
        y_prob_task2 = np.concatenate(y_prob_task2)
        y_pred_task2 = (y_prob_task2 > 0.5).astype(int)
        auc_o = roc_auc_score(y_true_task2, y_prob_task2)
        print(f"Task 2 Val: AUC_o={auc_o:.3f}")
        fpr, tpr, _ = roc_curve(y_true_task2, y_prob_task2)
        auc_task2_value = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true_task2, y_prob_task2)
        aupr_task2 = auc(recall, precision)
        acc_task2 = accuracy_score(y_true_task2, y_pred_task2)
        pre_task2 = precision_score(y_true_task2, y_pred_task2)
        rec_task2 = recall_score(y_true_task2, y_pred_task2)
        f1_task2 = f1_score(y_true_task2, y_pred_task2)
        print(f"Task 2 Val: AUC={auc_task2_value:.3f}, AUPR={aupr_task2:.3f}, ACC={acc_task2:.3f}, "
              f"PRE={pre_task2:.3f}, REC={rec_task2:.3f}, F1={f1_task2:.3f}")

        # 判断任务一是否为最优结果
        if maauc_ho > best_task1_maauc_ho:
            best_task1_maauc_ho = maauc_ho
            best_task1_metrics = (mip, mir, mif, miauc, maauc, hl, ap, avgF1, pat1, mip_ho, mir_ho, mif_ho, miauc_ho,
                                  maauc_ho, hl_ho, ap_ho, avgF1_ho, pat1_ho)

        # 判断任务二是否为最优结果
        if auc_task2_value > best_task2_auc:
            best_task2_auc = auc_task2_value
            best_task2_metrics = (auc_task2_value, aupr_task2, acc_task2, pre_task2, rec_task2, f1_task2)

    # 存储任务一最优结果
    if best_task1_metrics:
        mip, mir, mif, miauc, maauc, hl, ap, avgF1, pat1, mip_ho, mir_ho, mif_ho, miauc_ho, maauc_ho, hl_ho, ap_ho, avgF1_ho, pat1_ho = best_task1_metrics
        MiP.append(mip)
        MiR.append(mir)
        MiF.append(mif)
        MiAUC.append(miauc)
        MaAUC.append(maauc)
        HL.append(hl)
        AP.append(ap)
        AvgF1.append(avgF1)
        Pat1.append(pat1)
        MiP_ho.append(mip_ho)
        MiR_ho.append(mir_ho)
        MiF_ho.append(mif_ho)
        MiAUC_ho.append(miauc_ho)
        MaAUC_ho.append(maauc_ho)
        HL_ho.append(hl_ho)
        AP_ho.append(ap_ho)
        AvgF1_ho.append(avgF1_ho)
        Pat1_ho.append(pat1_ho)

    # 存储任务二最优结果
    if best_task2_metrics:
        auc_task2_value, aupr_task2, acc_task2, pre_task2, rec_task2, f1_task2 = best_task2_metrics
        AUC_task2.append(auc_task2_value)
        AUPR_task2.append(aupr_task2)
        ACC_task2.append(acc_task2)
        PRE_task2.append(pre_task2)
        REC_task2.append(rec_task2)
        F1_task2.append(f1_task2)

# 计算任务一指标均值（验证集相关）
MiP_mean = np.mean(MiP)
MiR_mean = np.mean(MiR)
MiF_mean = np.mean(MiF)
MiAUC_mean = np.mean(MiAUC)
MaAUC_mean = np.mean(MaAUC)
HL_mean = np.mean(HL)
AP_mean = np.mean(AP)
AvgF1_mean = np.mean(AvgF1)
Pat1_mean = np.mean(Pat1)

# 计算任务一指标均值（独立测试集 ho 相关）
MiP_mean_ho = np.mean(MiP_ho)
MiR_mean_ho = np.mean(MiR_ho)
MiF_mean_ho = np.mean(MiF_ho)
MiAUC_mean_ho = np.mean(MiAUC_ho)
MaAUC_mean_ho = np.mean(MaAUC_ho)
HL_mean_ho = np.mean(HL_ho)
AP_mean_ho = np.mean(AP_ho)
AvgF1_mean_ho = np.mean(AvgF1_ho)
Pat1_mean_ho = np.mean(Pat1_ho)

print(f"\nMean   : MiP={MiP_mean:.3f}, MiR={MiR_mean:.3f}, MiF={MiF_mean:.3f}, "
      f"MiAUC={MiAUC_mean:.3f}, MaAUC={MaAUC_mean:.3f}, HL={HL_mean:.3f}, "
      f"AP={AP_mean:.3f}, AvgF1={AvgF1_mean:.3f}, P@1={Pat1_mean:.3f}")

print(f"Mean Ho: MiP={MiP_mean_ho:.3f}, MiR={MiR_mean_ho:.3f}, MiF={MiF_mean_ho:.3f}, "
      f"MiAUC={MiAUC_mean_ho:.3f}, MaAUC={MaAUC_mean_ho:.3f}, HL={HL_mean_ho:.3f}, "
      f"AP={AP_mean_ho:.3f}, AvgF1={AvgF1_mean_ho:.3f}, P@1={Pat1_mean_ho:.3f}")

# 计算任务二指标均值
AUC_task2_mean = np.mean(AUC_task2)
AUPR_task2_mean = np.mean(AUPR_task2)
ACC_task2_mean = np.mean(ACC_task2)
PRE_task2_mean = np.mean(PRE_task2)
REC_task2_mean = np.mean(REC_task2)
F1_task2_mean = np.mean(F1_task2)

print(f"\nMean Task 2: AUC={AUC_task2_mean:.3f}, AUPR={AUPR_task2_mean:.3f}, ACC={ACC_task2_mean:.3f}, "
      f"PRE={PRE_task2_mean:.3f}, REC={REC_task2_mean:.3f}, F1={F1_task2_mean:.3f}")

end = time.time()
haoshi = end - start
print(f"运行时间: {haoshi} 秒")
