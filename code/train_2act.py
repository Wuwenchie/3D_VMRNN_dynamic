from 3D_VMRNN_2act import (
    create_supervised_geo_vmrnn, get_ablation_configs, get_enso_phase_label_tensor,
    PhysicalLatentGating, ENSO_PHASE_TO_ACTIVATION
)
from dynamic_loss_weighter import DynamicLossWeighter
from DynamicActivationSelector import DynamicActivationLoss, detect_climate_phenomenon
from myconfig import mypara
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import math
from LoadData import make_dataset2, make_testdataset
from progressive_teacher_forcing import TeacherForcingScheduler
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import gc  # 添加垃圾回收模块

# 設置 PyTorch 內存分配器配置
# 這有助於減少內存碎片化
torch.cuda.empty_cache()
if hasattr(torch.cuda, 'memory_stats'):
    print(f"初始 CUDA 內存狀態: {torch.cuda.memory_allocated() / 1024**2:.2f} MB 已分配")

# 尝试设置 PyTorch CUDA 内存分配器配置
try:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("已設置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
except Exception as e:
    print(f"設置 CUDA 內存分配器配置時出錯: {e}")


# =============================================================================
# Linear Probing 模組（整合自 linear_probing.py，文件三原始保留）
# =============================================================================

def get_enso_phase_label(nino_value):
    """
    根據 Niño 3.4 值標註 ENSO 相位
    0 = La Niña  (< -0.5°C)
    1 = Neutral  (-0.5°C ~ 0.5°C)
    2 = El Niño  (> 0.5°C)
    參考：NOAA Climate Prediction Center 的 ONI 定義
    """
    if nino_value < -0.5:
        return 0  # La Niña
    elif nino_value > 0.5:
        return 2  # El Niño
    else:
        return 1  # Neutral


def batch_label_enso_phase(nino_values):
    """批次標註 ENSO 相位"""
    return torch.tensor(
        [get_enso_phase_label(n.item()) for n in nino_values],
        dtype=torch.long
    )


class LinearPhaseProbe(nn.Module):
    """
    線性探針：驗證 latent 是否線性可分地編碼了 ENSO 相位
    參考: Tenney et al. (ICLR 2019) - "Probing for sentence structure"
    設計原則：故意使用最簡單的線性模型，避免探針本身學習複雜模式
    """
    def __init__(self, latent_dim=256, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)
        print(f"[LinearPhaseProbe] 初始化完成")
        print(f"  輸入維度: {latent_dim}")
        print(f"  輸出類別: {num_classes} (La Niña / Neutral / El Niño)")
        print(f"  總參數量: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, latent):
        """
        Args:
            latent: [B, latent_dim]
        Returns:
            logits: [B, num_classes]
        """
        return self.linear(latent)


# ★ 新增：針對 Gate Logits 的線性探針 ─────────────────────────────────────
class GateLogitPhaseProbe(nn.Module):
    """
    ★ 方案 F — Gate Logit 線性探針

    驗證 PhysicalLatentGating 的 gate logits（維度=2）是否線性可分地
    編碼了 ENSO 相位（3 類別）。

    這是「最嚴格、最直接」的 linear probing：
      - 輸入維度僅有 2（gate logits），沒有藏在高維空間的餘地
      - 若準確率 >> 33.3%，代表 gate 真的學到了 ENSO 相位映射
      - 若準確率接近 66.7%，代表 gate 至少學會了 El Niño vs 其他

    設計原則：使用最簡單的線性層，不加任何隱層。

    Args:
        gate_dim  : gate logits 維度，通常為 num_activations=2
        num_classes: 分類數，通常為 3（La Niña / Neutral / El Niño）
    """
    def __init__(self, gate_dim=2, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(gate_dim, num_classes)
        print(f"[GateLogitPhaseProbe] 初始化完成")
        print(f"  Gate logit 維度: {gate_dim}")
        print(f"  輸出類別: {num_classes} (La Niña / Neutral / El Niño)")
        print(f"  總參數量: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, gate_logits):
        """
        Args:
            gate_logits: [B, gate_dim]（從 PhysicalLatentGating._last_logits 取得）
        Returns:
            logits: [B, num_classes]
        """
        return self.linear(gate_logits)


def extract_latents_and_labels(model, dataloader, device, mypara):
    """
    從資料集提取所有樣本的 latent vectors 和 ENSO 相位標籤
    """
    model.eval()
    all_latents = []
    all_phase_labels = []
    all_nino_values = []

    sstlevel = 2 if mypara.needtauxy else 0

    print("\n" + "="*60)
    print("開始提取 Latent Vectors 和 ENSO 相位標籤")
    print("="*60)

    with torch.no_grad():
        for i, (input_var, var_true) in enumerate(dataloader):
            input_var = input_var.float().to(device)

            # 使用修改後的 encode() 提取 latent（支援新版三回傳值格式）
            try:
                _, _, latent_for_probe = model.encode(input_var)
            except ValueError:
                print("Warning: 模型 encode() 回傳格式舊版，請更新 GeoVMRNN_enhance.py")
                encoder_output, encoder_hidden = model.encode(input_var)
                latent_for_probe = encoder_hidden[0].mean(dim=1)

            # 計算當前時間步的 Niño 3.4（對應輸入序列最後一步）
            SST = var_true[:, -1, sstlevel]  # [B, H, W]
            nino_current = SST[
                :,
                mypara.lat_nino_relative[0]:mypara.lat_nino_relative[1],
                mypara.lon_nino_relative[0]:mypara.lon_nino_relative[1],
            ].mean(dim=[1, 2])  # [B]

            phase_labels = batch_label_enso_phase(nino_current)

            all_latents.append(latent_for_probe.cpu())
            all_phase_labels.append(phase_labels)
            all_nino_values.append(nino_current.cpu())

            if (i + 1) % 50 == 0:
                print(f"  已處理 {(i+1) * input_var.size(0)} 個樣本...")

    latents = torch.cat(all_latents, dim=0)
    phase_labels = torch.cat(all_phase_labels, dim=0)
    nino_values = torch.cat(all_nino_values, dim=0)

    print(f"\n✅ 資料提取完成！")
    print(f"  總樣本數: {latents.size(0)}")
    print(f"  Latent 維度: {latents.size(1)}")
    print(f"  相位分布:")
    for phase, name in enumerate(['La Niña', 'Neutral', 'El Niño']):
        count = (phase_labels == phase).sum().item()
        pct = 100 * count / len(phase_labels)
        print(f"    {name}: {count} ({pct:.1f}%)")
    print(f"  Niño 3.4 範圍: {nino_values.min():.2f} ~ {nino_values.max():.2f} °C")

    return latents, phase_labels, nino_values


# ★ 新增：提取 Gate Logits 供 GateLogitPhaseProbe 使用 ─────────────────────
def extract_gate_logits_and_labels(model, dataloader, device, mypara):
    """
    ★ 方案 F — 從 PhysicalLatentGating 提取 gate logits 和 ENSO 相位標籤。

    這是驗證「門控是否真的學到 ENSO 相位選擇能力」的核心函數。

    流程：
      1. 對每個 batch 執行一次 model.forward(train=False)
      2. 從 model.fusion_activation._last_logits 取得 gate logits [B, 2]
      3. 同步計算對應的 ENSO 相位標籤

    Args:
        model      : 已訓練好的 GeoVMRNN_Enhance（fusion_activation 需為 PhysicalLatentGating）
        dataloader : 資料加載器
        device     : 設備
        mypara     : 參數配置

    Returns:
        gate_logits : [N, 2] Tensor，所有樣本的 gate logits（最後一個 decode step）
        phase_labels: [N]   LongTensor，ENSO 相位標籤
        nino_values : [N]   FloatTensor，Niño 3.4 原始值（供分析用）
    """
    # 確認模型有 PhysicalLatentGating
    if not isinstance(getattr(model, 'fusion_activation', None), PhysicalLatentGating):
        print("[Warning] model.fusion_activation 不是 PhysicalLatentGating，無法提取 gate logits")
        return None, None, None

    model.eval()
    all_gate_logits  = []
    all_phase_labels = []
    all_nino_values  = []

    sstlevel = 2 if mypara.needtauxy else 0

    print("\n" + "="*60)
    print("★ 提取 Gate Logits（方案 F：門控相位分類驗證）")
    print("="*60)
    print(f"  gate logit[0] 大 → 傾向選 ReLU          （El Niño 目標）")
    print(f"  gate logit[1] 大 → 傾向選 Snake_lowfreq  （Neutral 目標）")
    print(f"  gate logit[2] 大 → 傾向選 Snake_highfreq （La Niña 目標）")

    with torch.no_grad():
        for i, (input_var, var_true) in enumerate(dataloader):
            input_var = input_var.float().to(device)

            # 清空 _last_logits（確保取到的是本次 forward 的結果）
            model.fusion_activation._last_logits = None
            model.fusion_activation._last_logits_grad = None

            # 執行 forward（推理模式，只跑一個 decode step 即可取到 logits）
            # 為了效率，直接呼叫 encode + 一次 decode_step
            _, encoder_hidden, _ = model.encode(input_var)
            current_input = input_var[:, -1]

            # 執行一次 decode_step 以觸發 PhysicalLatentGating.forward()
            _, _, _, _, _ = model.decode_step(
                current_input,
                encoder_output=None,      # decode_step 中未實際使用 encoder_output
                decoder_hidden=encoder_hidden,
            )

            # 取得 gate logits（使用 get_gate_logits_for_probing() 正確處理累積格式）
            gate_logits = model.fusion_activation.get_gate_logits_for_probing()  # [B, num_act]
            if gate_logits is None:
                print(f"  [Warning] batch {i}: gate_logits 為 None，跳過")
                continue

            # 計算 ENSO 相位標籤
            SST = var_true[:, -1, sstlevel]  # [B, H, W]
            nino_current = SST[
                :,
                mypara.lat_nino_relative[0]:mypara.lat_nino_relative[1],
                mypara.lon_nino_relative[0]:mypara.lon_nino_relative[1],
            ].mean(dim=[1, 2])  # [B]

            # ★ 確保 gate_logits 與 nino_current batch size 一致
            # （最後一個不完整 batch 可能造成不一致）
            actual_b = nino_current.size(0)
            gate_logits = gate_logits[:actual_b]   # 沿 B 維裁切

            phase_labels = batch_label_enso_phase(nino_current)

            all_gate_logits.append(gate_logits.detach().cpu())
            all_phase_labels.append(phase_labels)
            all_nino_values.append(nino_current.cpu())

            if (i + 1) % 50 == 0:
                print(f"  已處理 {(i+1) * input_var.size(0)} 個樣本...")

    if not all_gate_logits:
        print("  [Error] 未能提取任何 gate logits")
        return None, None, None

    gate_logits_all  = torch.cat(all_gate_logits,  dim=0)  # [N, 2]
    phase_labels_all = torch.cat(all_phase_labels, dim=0)  # [N]
    nino_values_all  = torch.cat(all_nino_values,  dim=0)  # [N]

    print(f"\n✅ Gate Logits 提取完成！")
    print(f"  總樣本數: {gate_logits_all.size(0)}")
    print(f"  Gate logits 形狀: {gate_logits_all.shape}  (index 0=ReLU, 1=Snake)")
    print(f"  相位分布:")
    for phase, name in enumerate(['La Niña', 'Neutral', 'El Niño']):
        count = (phase_labels_all == phase).sum().item()
        pct = 100 * count / len(phase_labels_all)
        avg_logit_0 = gate_logits_all[phase_labels_all == phase, 0].mean().item()
        avg_logit_1 = gate_logits_all[phase_labels_all == phase, 1].mean().item()
        print(f"    {name:10s}: {count:5d} ({pct:5.1f}%)  "
              f"avg logit[ReLU]={avg_logit_0:.3f}  avg logit[Snake]={avg_logit_1:.3f}")

    return gate_logits_all, phase_labels_all, nino_values_all


def train_linear_probe(train_loader, test_loader, latent_dim=256, num_classes=3,
                       device='cuda', num_epochs=50, lr=1e-3):
    """
    訓練線性探針
    設計原則：
    1. 只訓練探針，不訓練主模型（主模型已凍結在資料提取階段）
    2. 使用簡單的優化器（Adam）和適度的學習率
    3. 訓練 epoch 不宜過多（避免過擬合）
    """
    print("\n" + "="*60)
    print("開始訓練 Linear Probe")
    print("="*60)

    probe = LinearPhaseProbe(latent_dim=latent_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # 訓練階段
        probe.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for latent, label in train_loader:
            latent, label = latent.to(device), label.to(device)
            optimizer.zero_grad()
            logits = probe(latent)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * latent.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == label).sum().item()
            train_total += label.size(0)

        train_loss /= train_total
        train_acc = 100 * train_correct / train_total

        # 測試階段
        probe.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for latent, label in test_loader:
                latent, label = latent.to(device), label.to(device)
                logits = probe(latent)
                pred = logits.argmax(dim=1)
                test_correct += (pred == label).sum().item()
                test_total += label.size(0)

        test_acc = 100 * test_correct / test_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}  |  "
                  f"Train Loss: {train_loss:.4f}  |  "
                  f"Train Acc: {train_acc:.2f}%  |  "
                  f"Test Acc: {test_acc:.2f}%")

    print(f"\n✅ 訓練完成！")
    print(f"  最佳測試準確率: {best_test_acc:.2f}% (Epoch {best_epoch})")

    return probe, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch
    }


# ★ 新增：訓練 GateLogitPhaseProbe ────────────────────────────────────────
def train_gate_logit_probe(train_loader, test_loader, gate_dim=2, num_classes=3,
                           device='cuda', num_epochs=50, lr=1e-3):
    """
    ★ 方案 F — 訓練 Gate Logit 線性探針

    這個探針只有 gate_dim=2 → num_classes=3 的線性映射，
    是驗證門控學習能力最簡潔的方式。

    預期結果：
      - 若訓練成功（方案 F 損失有效），準確率應 >> 33.3%（隨機基線）
      - El Niño vs 非 El Niño 的二分類準確率應接近 66.7%+（La Niña+Neutral 合併）
      - 完整 3 類別準確率若 > 50% 即代表門控確實編碼了 ENSO 相位信息
    """
    print("\n" + "="*60)
    print("★ 訓練 Gate Logit 線性探針（方案 F 驗證）")
    print(f"  輸入維度: {gate_dim}（僅 gate logits）")
    print(f"  分類目標: {num_classes} 類")
    print("="*60)

    probe = GateLogitPhaseProbe(gate_dim=gate_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # 訓練
        probe.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for gate_logit, label in train_loader:
            gate_logit, label = gate_logit.to(device), label.to(device)
            optimizer.zero_grad()
            logits = probe(gate_logit)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * gate_logit.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == label).sum().item()
            train_total += label.size(0)

        train_loss /= train_total
        train_acc = 100 * train_correct / train_total

        # 測試
        probe.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for gate_logit, label in test_loader:
                gate_logit, label = gate_logit.to(device), label.to(device)
                logits = probe(gate_logit)
                pred = logits.argmax(dim=1)
                test_correct += (pred == label).sum().item()
                test_total += label.size(0)

        test_acc = 100 * test_correct / test_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}  |  "
                  f"Train Loss: {train_loss:.4f}  |  "
                  f"Train Acc: {train_acc:.2f}%  |  "
                  f"Test Acc: {test_acc:.2f}%")

    print(f"\n✅ Gate Logit 探針訓練完成！")
    print(f"  最佳測試準確率: {best_test_acc:.2f}% (Epoch {best_epoch})")
    print(f"  隨機基線準確率: 33.33%")

    if best_test_acc > 60:
        print(f"  🎉 結論：gate 成功學到 ENSO 相位映射（準確率 {best_test_acc:.1f}%）")
    elif best_test_acc > 45:
        print(f"  ✅ 結論：gate 有部分 ENSO 相位信息（準確率 {best_test_acc:.1f}%）")
    else:
        print(f"  ⚠️  結論：gate 相位信息不足（準確率 {best_test_acc:.1f}%），建議加大 phase_gate_weight")

    return probe, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch
    }


def evaluate_probe(probe, test_loader, device, save_dir='./linear_probe_results'):
    """詳細評估探針性能並繪製可視化圖表"""
    os.makedirs(save_dir, exist_ok=True)

    probe.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("\n" + "="*60)
    print("評估 Linear Probe 性能")
    print("="*60)

    with torch.no_grad():
        for latent, label in test_loader:
            latent = latent.to(device)
            logits = probe(latent)
            probs = F.softmax(logits, dim=1)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(label)
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # 混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['La Niña', 'Neutral', 'El Niño'],
                yticklabels=['La Niña', 'Neutral', 'El Niño'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Phase', fontsize=12, fontweight='bold')
    plt.ylabel('True Phase', fontsize=12, fontweight='bold')
    test_acc = 100 * accuracy_score(all_labels, all_preds)
    plt.title(f'Linear Probe Confusion Matrix\nTest Accuracy = {test_acc:.2f}%',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 混淆矩陣已儲存: {save_dir}/confusion_matrix.png")

    # 分類報告
    report = classification_report(
        all_labels, all_preds,
        target_names=['La Niña', 'Neutral', 'El Niño'],
        digits=3
    )
    print(f"\n分類報告:")
    print(report)

    with open(f'{save_dir}/classification_report.txt', 'w') as f:
        f.write("Linear Probe 分類報告\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write(f"\n整體準確率: {test_acc:.2f}%\n")
    print(f"  ✅ 分類報告已儲存: {save_dir}/classification_report.txt")

    return test_acc, cm, report


# ★ 新增：GateLogitProbe 視覺化分析 ─────────────────────────────────────
def evaluate_gate_logit_probe(gate_probe, test_loader, device,
                              save_dir='./linear_probe_results/gate_logit'):
    """
    ★ 方案 F — 評估並視覺化 Gate Logit 探針

    包含：
      1. 混淆矩陣
      2. Gate logit 散點圖（按相位著色）：直觀看 gate 是否分離 El Niño
      3. 分類報告
    """
    os.makedirs(save_dir, exist_ok=True)

    gate_probe.eval()
    all_preds    = []
    all_labels   = []
    all_logits   = []  # 原始 gate logits（用於散點圖）

    with torch.no_grad():
        for gate_logit, label in test_loader:
            gate_logit = gate_logit.to(device)
            pred_logits = gate_probe(gate_logit)
            all_preds.append(pred_logits.argmax(dim=1).cpu())
            all_labels.append(label)
            all_logits.append(gate_logit.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()   # [N, 2]

    test_acc = 100 * accuracy_score(all_labels, all_preds)

    # ── 1. 混淆矩陣 ──────────────────────────────────────────────────────── #
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['La Niña', 'Neutral', 'El Niño'],
                yticklabels=['La Niña', 'Neutral', 'El Niño'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Phase', fontsize=12, fontweight='bold')
    plt.ylabel('True Phase', fontsize=12, fontweight='bold')
    plt.title(f'Gate Logit Probe Confusion Matrix\nTest Accuracy = {test_acc:.2f}%\n'
              f'(Input: 2-dim gate logits only)',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/gate_logit_probe_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Gate logit 混淆矩陣已儲存")

    # ── 2. Gate Logit 散點圖（關鍵圖！）──────────────────────────────────── #
    phase_names  = ['La Niña', 'Neutral', 'El Niño']
    phase_colors = ['#2196F3', '#4CAF50', '#F44336']  # 藍、綠、紅

    plt.figure(figsize=(9, 7))
    for phase_idx, (name, color) in enumerate(zip(phase_names, phase_colors)):
        mask = all_labels == phase_idx
        plt.scatter(
            all_logits[mask, 0],   # gate logit for ReLU  (x axis)
            all_logits[mask, 1],   # gate logit for Snake (y axis)
            c=color, label=name, alpha=0.5, s=20, edgecolors='none'
        )

    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.xlabel('Gate Logit[0] — ReLU (El Niño target ↑)', fontsize=12, fontweight='bold')
    plt.ylabel('Gate Logit[1] — Snake (La Niña/Neutral target ↑)', fontsize=12, fontweight='bold')
    plt.title(f'Gate Logits by ENSO Phase\n'
              f'Test Acc = {test_acc:.1f}%  |  Random Baseline = 33.3%',
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/gate_logit_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Gate logit 散點圖已儲存（關鍵視覺化）")

    # ── 3. 分類報告 ──────────────────────────────────────────────────────── #
    report = classification_report(
        all_labels, all_preds,
        target_names=phase_names,
        digits=3
    )
    print(f"\n★ Gate Logit 探針分類報告:")
    print(report)

    with open(f'{save_dir}/gate_logit_probe_report.txt', 'w') as f:
        f.write("★ 方案 F — Gate Logit 線性探針分類報告\n")
        f.write("輸入：僅 2 維 gate logits（index 0=ReLU, index 1=Snake）\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write(f"\n整體準確率: {test_acc:.2f}%\n")
        f.write(f"隨機基線:   33.33%\n")
        f.write(f"超越基線:   +{test_acc - 33.33:.2f}%\n")
    print(f"  ✅ Gate logit 探針分類報告已儲存")

    return test_acc, cm, report


def plot_training_history(history, save_dir='./linear_probe_results'):
    """繪製訓練歷史曲線"""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_losses']) + 1)

    ax1.plot(epochs, history['train_losses'], 'b-', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(epochs, history['train_accs'], 'b-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, history['test_accs'], 'r-', linewidth=2, label='Test Acc')
    ax2.axhline(y=33.33, color='gray', linestyle='--', linewidth=1,
                alpha=0.7, label='Random Baseline (33.3%)')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Test Accuracy', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 訓練歷史曲線已儲存: {save_dir}/training_history.png")


def run_linear_probe_experiment(
    model,
    mypara,
    save_dir='./linear_probe_results',
    num_probe_epochs=50,
    probe_lr=1e-3,
    test_size=0.2,
    batch_size=64
):
    """
    完整的 Linear Probing 實驗流程（接受已載入的 model，直接使用，不重新載入）
    ★ 新增：在原始 latent probe 之後，額外執行 Gate Logit Probe 實驗

    Args:
        model: 已訓練好的 GeoVMRNN 模型（直接傳入，無需再次載入）
        mypara: 參數配置
        save_dir: 結果儲存目錄
        num_probe_epochs: 探針訓練 epoch 數
        probe_lr: 探針學習率
        test_size: 測試集比例
        batch_size: 批次大小
    """
    device = mypara.device
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Linear Probing 實驗：驗證 Latent 是否編碼 ENSO 相位")
    print("="*60)
    print(f"結果目錄: {save_dir}")

    # Step 1: 提取 Latent Vectors 和標籤
    print("\n[Step 1] 提取 Latent Vectors...")
    evaldataset = make_testdataset(mypara, ngroup=200)
    eval_loader = DataLoader(evaldataset, batch_size=16, shuffle=False)

    latents, phase_labels, nino_values = extract_latents_and_labels(
        model, eval_loader, device, mypara
    )

    # Step 2: 分割訓練/測試集
    print(f"\n[Step 2] 分割訓練/測試集 (測試集比例 = {test_size})...")
    train_latents, test_latents, train_labels, test_labels = train_test_split(
        latents, phase_labels,
        test_size=test_size,
        stratify=phase_labels,
        random_state=42
    )
    print(f"  訓練集: {train_latents.size(0)} 樣本")
    print(f"  測試集: {test_latents.size(0)} 樣本")

    train_dataset = TensorDataset(train_latents, train_labels)
    test_dataset = TensorDataset(test_latents, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 3: 訓練 Linear Probe（原始 latent 版）
    print(f"\n[Step 3] 訓練 Linear Probe ({num_probe_epochs} epochs)...")
    probe, history = train_linear_probe(
        train_loader, test_loader,
        latent_dim=latents.size(1),
        num_classes=3,
        device=device,
        num_epochs=num_probe_epochs,
        lr=probe_lr
    )

    # Step 4: 評估與可視化
    print("\n[Step 4] 評估探針性能...")
    test_acc, cm, report = evaluate_probe(probe, test_loader, device, save_dir)
    plot_training_history(history, save_dir)

    # Step 5: 儲存探針模型
    probe_save_path = f'{save_dir}/linear_probe.pth'
    torch.save(probe.state_dict(), probe_save_path)
    print(f"\n  ✅ 探針模型已儲存: {probe_save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # ★ Step 5.5：Gate Logit Probe 實驗（方案 F 驗證）
    # ─────────────────────────────────────────────────────────────────────────
    gate_probe_test_acc = None
    if isinstance(getattr(model, 'fusion_activation', None), PhysicalLatentGating):
        print(f"\n{'='*60}")
        print("★ [Step 5.5] Gate Logit 線性探針實驗（方案 F）")
        print(f"{'='*60}")

        gate_logits_all, gate_phase_labels, gate_nino_values = extract_gate_logits_and_labels(
            model, eval_loader, device, mypara
        )

        if gate_logits_all is not None:
            # 分割訓練/測試集
            # stratify 要求每個類別至少有 1/test_size 筆，若不足則改用隨機分割
            try:
                (train_gate, test_gate,
                 train_gate_labels, test_gate_labels) = train_test_split(
                    gate_logits_all, gate_phase_labels,
                    test_size=test_size,
                    stratify=gate_phase_labels,
                    random_state=42
                )
            except ValueError as e:
                print(f"  [Warning] stratify 分割失敗（{e}），改用隨機分割")
                (train_gate, test_gate,
                 train_gate_labels, test_gate_labels) = train_test_split(
                    gate_logits_all, gate_phase_labels,
                    test_size=test_size,
                    stratify=None,
                    random_state=42
                )

            gate_train_dataset = TensorDataset(train_gate, train_gate_labels)
            gate_test_dataset  = TensorDataset(test_gate,  test_gate_labels)
            gate_train_loader  = DataLoader(gate_train_dataset, batch_size=batch_size, shuffle=True)
            gate_test_loader   = DataLoader(gate_test_dataset,  batch_size=batch_size, shuffle=False)

            # 訓練 Gate Logit Probe
            gate_probe, gate_history = train_gate_logit_probe(
                gate_train_loader, gate_test_loader,
                gate_dim=gate_logits_all.size(1),
                num_classes=3,
                device=device,
                num_epochs=num_probe_epochs,
                lr=probe_lr
            )

            # 評估並視覺化
            gate_probe_save_dir = os.path.join(save_dir, 'gate_logit')
            gate_probe_test_acc, _, _ = evaluate_gate_logit_probe(
                gate_probe, gate_test_loader, device, gate_probe_save_dir
            )
            plot_training_history(gate_history, gate_probe_save_dir)

            # 儲存 Gate Probe 模型
            gate_probe_save_path = f'{gate_probe_save_dir}/gate_logit_probe.pth'
            torch.save(gate_probe.state_dict(), gate_probe_save_path)
            print(f"  ✅ Gate logit 探針模型已儲存: {gate_probe_save_path}")
    else:
        print("\n[Step 5.5] 跳過 Gate Logit Probe（fusion_activation 不是 PhysicalLatentGating）")

    # Step 6: 結果總結
    print("\n" + "="*60)
    print("實驗結果總結")
    print("="*60)
    print(f"  [Latent Probe] 最佳測試準確率: {history['best_test_acc']:.2f}%")
    print(f"  [Latent Probe] 隨機基準準確率: 33.33%")
    print(f"  [Latent Probe] 準確率提升    : +{history['best_test_acc'] - 33.33:.2f}%")

    if gate_probe_test_acc is not None:
        print(f"\n  [Gate Logit Probe] 最佳測試準確率: {gate_probe_test_acc:.2f}%")
        print(f"  [Gate Logit Probe] 隨機基準準確率: 33.33%")
        print(f"  [Gate Logit Probe] 準確率提升    : +{gate_probe_test_acc - 33.33:.2f}%")
        print(f"  （注意：Gate Logit Probe 僅用 2 維輸入，準確率高更具說服力）")

    if history['best_test_acc'] < 40:
        conclusion = "❌ Latent 幾乎不含 ENSO 信息"
        suggestion = "建議：大幅提高 probe_weight 或重新設計監督信號"
    elif history['best_test_acc'] < 60:
        conclusion = "⚠️  有部分信息，但編碼不夠清晰"
        suggestion = "建議：調整 probe_weight 或使用對比學習"
    elif history['best_test_acc'] < 75:
        conclusion = "✅ 成功編碼 ENSO 信息，但仍有改進空間"
        suggestion = "建議：檢查門控網路是否有效利用此信息（診斷 1）"
    else:
        conclusion = "🎉 強烈編碼 ENSO 信息！"
        suggestion = "可以撰寫論文了！重點分析門控權重與相位的關聯"

    print(f"\n  結論：{conclusion}")
    print(f"  {suggestion}")
    print(f"\n  所有結果已儲存至：{save_dir}/")
    print("="*60 + "\n")

    return probe, history, test_acc


# =============================================================================
# DynamicActivationTrainer — 整合版訓練器
# （保留文件三所有功能 + 整合文件四的激活函數記錄機制 + 方案 F 相位監督）
# =============================================================================

class DynamicActivationTrainer:
    """動態激活函數訓練器（整合版 + 方案 F ENSO 相位門控監督）"""
    def __init__(self, mypara):
        self.mypara = mypara
        self.device = mypara.device

        # ── 模型配置 ──────────────────────────────────────────────────────── #
        self.activation_configs = get_ablation_configs()
        
        # ── 創建模型：支援 physical_gating（文件四）或 dynamic_fusion（文件三）── #
        # 預設使用 physical_gating；如需 dynamic_fusion 可修改此處
        self.mymodel = create_supervised_geo_vmrnn(
            mypara,
            activation_config={
                'cnn'       : 'relu',
                'fusion'    : {
                    'type'  : 'physical_gating',
                    'params': {
                        'num_activations'   : 2,
                        'use_hard_selection': False,
                    }
                },
                'prediction': 'relu',
            }
        ).to(self.device)

        # 添加模型參數統計
        total_params = sum(p.numel() for p in self.mymodel.parameters())
        trainable_params = sum(p.numel() for p in self.mymodel.parameters() if p.requires_grad)
        print(f"模型總參數: {total_params:,}")
        print(f"可訓練參數: {trainable_params:,}")
        print(f"模型設備: {self.device}")
        print(f"使用動態激活函數配置")

        # ── 優化器（在模型創建後才初始化）────────────────────────────────── #
        self.opt = torch.optim.Adam(self.mymodel.parameters(), lr=1e-4)

        # ── 損失權重（保留作 fallback，平時由動態調度器覆蓋）──────────────── #
        # 注意：僅在 activation_weights 為 None（fallback 路徑）時使用
        # 實際訓練時由 DynamicLossWeighter 根據 EMA 動態計算
        self.diversity_weight    = 0.01
        self.entropy_weight      = 0.01
        self.probe_weight        = 10.0
        self.phase_gate_weight   = 0.0   # fallback 值（動態調度器目標：15% × loss_var）
        self.decorr_weight       = 0.5   # fallback 值（動態調度器目標：2%  × loss_var）

        # ── ★ 動態損失權重調度器開關 ─────────────────────────────────────── #
        # True  → 使用 DynamicLossWeighter（EMA 自動調節，對照組：Dyn）
        # False → 使用上方固定 fallback 權重（對照組：No-Dyn）
        # 切換後 model_name / experiment_name 會自動反映，方便 MLflow 區分兩組實驗
        self.use_dynamic_loss_weighter = False   # ← 在此切換

        # ── ★ 動態損失權重調度器 ──────────────────────────────────────────── #
        # target_ratios：各輔助損失加權後，希望佔 loss_var 的比例
        #   probe_loss      → 15%：強迫 L3 latent 編碼 ENSO 連續值 + 相位
        #   phase_gate_loss → 15%：強迫 gate 學習 ENSO 相位映射（核心目標）
        #   diversity_loss  →  3%：防止 gate 崩潰
        #   decorr_loss     →  2%：防止 gate logit 退化成 1D
        # curriculum_steps=5000：前 5000 步線性暖身，避免初期物理損失搶佔主任務
        self.loss_weighter = DynamicLossWeighter(
            target_ratios={
                'probe'     : 0.15,
                'phase_gate': 0.15,
                'diversity' : 0.03,
                'decorr'    : 0.02,
            },
            curriculum_steps = 5000,
            ema_decay        = 0.99,
            clip_max         = 2000.0,
            clip_min         = 0.1,
            ref_loss_key     = 'var',
            log_interval     = 200,
        )

        # ── 設置 SST 層級 ─────────────────────────────────────────────────── #
        self.sstlevel = 0
        if self.mypara.needtauxy:
            self.sstlevel = 2

        # ── 設置 Nino 指數權重（文件四：直接在目標設備上創建）──────────────── #
        ninoweight = torch.tensor(
            [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6,
            dtype=torch.float32,
            device=self.device
        ) * torch.log(torch.arange(24, dtype=torch.float32, device=self.device) + 1)
        self.ninoweight = ninoweight[:self.mypara.output_length]

        # ── 創建 Teacher Forcing 調度器 ───────────────────────────────────── #
        self.tf_scheduler = TeacherForcingScheduler(
            strategy='exponential',
            initial_ratio=1.0,
            final_ratio=0.0,
            decay_rate=0.9999
        )

        # ── 創建動態激活函數損失 ──────────────────────────────────────────── #
        self.dynamic_loss = DynamicActivationLoss(
            main_loss_fn=self.combine_loss,
            regularization_weight=0.1
        )

        # ── 記錄激活函數權重歷史 ──────────────────────────────────────────── #
        self.activation_weights_history = []

        # ── ✅ 新增：設備驗證（文件四）──────────────────────────────────────── #
        self._verify_device_placement()

    def _verify_device_placement(self):
        """驗證所有組件的設備配置（文件四新增）"""
        print(f"\n{'='*60}")
        print("設備配置檢查:")
        print(f"  目標設備: {self.device}")
        print(f"  模型設備: {next(self.mymodel.parameters()).device}")
        print(f"  ninoweight 設備: {self.ninoweight.device}")
        
        if hasattr(self.mymodel, 'fusion_activation'):
            fa = self.mymodel.fusion_activation
            if hasattr(fa, 'temperature'):
                print(f"  fusion_activation.temperature 設備: {fa.temperature.device}")
            if hasattr(fa, 'diversity_regularizer'):
                print(f"  diversity_regularizer.target_distribution 設備: "
                      f"{fa.diversity_regularizer.target_distribution.device}")
            if isinstance(fa, PhysicalLatentGating):
                print(f"  phase_to_activation_idx 設備: {fa.phase_to_activation_idx.device}")
                print(f"  ★ phase_gate_weight: {self.phase_gate_weight}")
                print(f"     La Niña→Snake_high[2], Neutral→Snake_low[1], El Niño→ReLU[0]")
        
        print(f"{'='*60}\n")

    def calscore(self, y_pred, y_true):
        """計算 Nino 評分"""
        with torch.no_grad():
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)
            true = y_true - y_true.mean(dim=0, keepdim=True)
            cor = (pred * true).sum(dim=0) / (
                torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0))
                + 1e-6
            )
            acc = (self.ninoweight * cor).sum()
            rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
            sc = 2 / 3.0 * acc - rmse
        return sc.item()

    def loss_var(self, y_pred, y_true, residual_losses=None, alpha=1.0):
        """計算變量損失"""
        min_len = min(y_pred.size(1), y_true.size(1))
        y_pred = y_pred[:, :min_len]
        y_true = y_true[:, :min_len]

        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
        rmse = rmse.mean(dim=0)
        rmse = rmse.sum()

        if residual_losses is not None:
            if isinstance(residual_losses, torch.Tensor):
                residual_term = alpha * residual_losses
            else:
                residual_term = alpha * torch.tensor(residual_losses, device=rmse.device)
        else:
            residual_term = torch.tensor(0.0, device=rmse.device)

        total_loss = rmse + residual_term
        return total_loss

    def loss_nino(self, y_pred, y_true):
        """計算 Nino 損失"""
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()

    def combine_loss(self, loss1, loss2):
        """組合損失函數"""
        combine_loss = loss1 + loss2
        return combine_loss

    def entropy_loss(self, weights):
        """計算熵損失：H(P) = -sum(p * log(p))（文件四新增）"""
        if weights is None:
            return 0.0
        eps = 1e-8
        entropy = -torch.sum(weights * torch.log(weights + eps), dim=1)
        return entropy.mean()

    def train_model(self, dataset_train, dataset_eval, num_epochs=50,
                    run_linear_probe=False, linear_probe_save_dir='./linear_probe_results',
                    linear_probe_epochs=50, linear_probe_lr=1e-3):
        """
        訓練模型

        Args:
            dataset_train: 訓練資料集
            dataset_eval: 驗證資料集
            num_epochs: 訓練 epoch 數
            run_linear_probe: 訓練結束後是否執行 Linear Probing 實驗（文件三功能）
            linear_probe_save_dir: Linear Probe 結果儲存目錄
            linear_probe_epochs: Linear Probe 訓練 epoch 數
            linear_probe_lr: Linear Probe 學習率
        """
        # 設置模型保存路徑
        model_name = "GeoVMRNN_DynamicActivation_Integrated.pkl"
        chk_path = self.mypara.model_savepath + model_name
        torch.manual_seed(self.mypara.seeds)

        # 創建數據加載器
        batch_size_train = max(1, self.mypara.batch_size_train // 2)
        batch_size_eval  = max(1, self.mypara.batch_size_eval  // 2)

        print(f"原始训练批次大小: {self.mypara.batch_size_train}, 調整後: {batch_size_train}")
        print(f"原始评估批次大小: {self.mypara.batch_size_eval}, 調整後: {batch_size_eval}")

        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size_train, shuffle=False
        )
        dataloader_eval = DataLoader(
            dataset_eval, batch_size=batch_size_eval, shuffle=False
        )

        count = 0
        best = -math.inf
        global_step = 0

        print(f"使用 Teacher Forcing 調度策略: {self.tf_scheduler.strategy}")
        print(f"使用動態激活函數配置（方案 E + F）")
        print(f"★ phase_gate_weight = {self.phase_gate_weight}（ENSO 相位門控監督，第五次校正）")
        print(f"  La Niña→Snake_highfreq[2], Neutral→Snake_lowfreq[1], El Niño→ReLU[0]")
        # ★ 開關狀態提示
        if self.use_dynamic_loss_weighter:
            print(f"★ 損失權重模式：DynamicLossWeighter（EMA 自動調節）")
            _run_tag   = "Dyn"
            _model_sfx = "Dyn"
        else:
            print(f"★ 損失權重模式：固定 fallback 權重（No-Dyn 對照組）")
            print(f"   probe={self.probe_weight}  phase_gate={self.phase_gate_weight}"
                  f"  decorr={self.decorr_weight}  diversity={self.diversity_weight}")
            _run_tag   = "NoDyn"
            _model_sfx = "NoDyn"
        print(f"模型將保存到: {chk_path}")

        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment("GeoVMRNN_DynamicActivation_2act")

        with mlflow.start_run(run_name=f"GeoVMRNN_2act_{_run_tag}"):
            mlflow.set_tag("model", "GeoVMRNN_DynamicActivation_Integrated")
            mlflow.set_tag("loss_weight_mode", _run_tag)
            mlflow.log_params({
                "activation_config"          : "physical_gating_scheme_EF",
                "use_dynamic_loss_weighter"  : self.use_dynamic_loss_weighter,  # ★ 開關記錄
                "lr"                 : 1e-4,
                "batch_size_train"   : batch_size_train,
                "batch_size_eval"    : batch_size_eval,
                "epochs"             : num_epochs,
                "tf_strategy"        : self.tf_scheduler.strategy,
                "scheme"             : "E_F_combined",
                "init_temperature"   : 10.0,
                "min_temperature"    : 1.0,
                "temperature_decay"  : 0.9998,
                "warmup_steps"       : 10000,
                "noise_scale"        : 1.0,
                "enforce_balance"    : True,
                "min_weight"         : 0.15,
                "balance_steps"      : 15000,
                "diversity_weight"   : self.diversity_weight,
                "probe_weight"       : self.probe_weight,
                "decorr_weight"      : self.decorr_weight,
                # ★ 動態損失權重調度器設定（use_dynamic_loss_weighter=True 時有效）
                "dyn_target_probe"   : 0.15,
                "dyn_target_gate"    : 0.15,
                "dyn_target_diversity": 0.03,
                "dyn_target_decorr"  : 0.02,
                "dyn_curriculum_steps": 5000,
                "dyn_ema_decay"      : 0.99,
                # ★ 方案 F
                "phase_gate_weight"  : self.phase_gate_weight,
                "ElNino_target_act"  : "ReLU(index=0)",
                "Neutral_target_act" : "LearnedSnake(index=1)",
                "LaNina_target_act"  : "LearnedSnake(index=1)",
                "num_activations"    : 2,
            })

            for i_epoch in range(num_epochs):
                print("=========="*8)
                print(f"\n-->epoch: {i_epoch}")

                # 訓練階段
                self.mymodel.train()
                epoch_activation_weights = []

                for j, (input_var, var_true) in enumerate(dataloader_train):
                    input_var = input_var.float().to(self.device)
                    var_true  = var_true.float().to(self.device)

                    # ── 提取真實 SST 和 Nino 指數 ──────────────────────────── #
                    SST = var_true[:, :, self.sstlevel]
                    nino_true = SST[
                        :, :,
                        self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                        self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1],
                    ].mean(dim=[2, 3])

                    # ── ✅ 新增（文件四）：取輸入序列最後一步的 Niño 3.4 → [B] ── #
                    input_SST    = var_true[:, -1, self.sstlevel]   # [B, H, W]
                    nino_current = input_SST[
                        :,
                        self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                        self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1],
                    ].mean(dim=[1, 2])                              # [B]

                    # ★ 轉換為相位標籤（方案 F 核心）
                    phase_labels = get_enso_phase_label_tensor(nino_current).to(self.device)  # [B]

                    # ── 檢測氣候現象類型（文件三原始功能）──────────────────── #
                    climate_labels = detect_climate_phenomenon(
                        var_true[:, :, self.sstlevel],
                        nino_region=(
                            self.mypara.lat_nino_relative[0],
                            self.mypara.lat_nino_relative[1],
                            self.mypara.lon_nino_relative[0],
                            self.mypara.lon_nino_relative[1]
                        )
                    )

                    # 設備確認（第一個 batch 打印）
                    if j == 0:
                        print(f"\nBatch 0 設備檢查:")
                        print(f"  input_var: {input_var.device}")
                        print(f"  var_true: {var_true.device}")
                        print(f"  nino_true: {nino_true.device}")
                        print(f"  ninoweight: {self.ninoweight.device}")
                        print(f"  phase_labels: {phase_labels.device}")

                    # ── 獲取當前的 Teacher Forcing 比例 ──────────────────── #
                    current_tf_ratio = self.tf_scheduler.get_ratio()

                    # ── ✅ 前向傳播（文件四：三回傳值格式）────────────────── #
                    var_pred, residual_loss, probe_loss = self.mymodel(
                        input_var,
                        var_true,
                        train=True,
                        sv_ratio=current_tf_ratio,
                        nino_true=nino_current,             # ✅ 文件四新增
                        nino_probe_weight=self.probe_weight,# ✅ 文件四新增
                    )

                    # ── 提取預測的 SST 和 Nino 指數 ──────────────────────── #
                    SST_pred = var_pred[:, :, self.sstlevel]
                    nino_pred = SST_pred[
                        :, :,
                        self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                        self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1],
                    ].mean(dim=[2, 3])

                    # ── ✅ 取得最新激活函數權重（文件四方式）──────────────── #
                    fa = self.mymodel.fusion_activation
                    activation_weights = None
                    if hasattr(fa, 'weights_history') and fa.weights_history:
                        activation_weights = fa.weights_history[-1]   # [B, num_activations]

                    # ── 計算損失 ─────────────────────────────────────────── #
                    loss_var  = self.loss_var(var_pred, var_true, residual_loss)
                    loss_nino = self.loss_nino(nino_pred, nino_true)
                    score     = self.calscore(nino_pred, nino_true)

                    # ── ✅ 組合損失（文件四：diversity + probe + ★ 方案 F）── #
                    combine_loss = self.combine_loss(loss_var, loss_nino)

                    # ── ★ 方案 F：ENSO 相位門控監督損失 ─────────────────── #
                    phase_gate_loss  = torch.tensor(0.0, device=self.device)
                    decorr_loss      = torch.tensor(0.0, device=self.device)
                    if isinstance(fa, PhysicalLatentGating):
                        try:
                            # ★ BUG FIX：使用 per-step 未來相位標籤 [T, B]
                            # model.forward() 已將其計算並存在 _phase_labels_per_step
                            # 若不存在（推理模式）則 fallback 到當前 phase_labels [B]
                            labels_for_gate = getattr(
                                self.mymodel, '_phase_labels_per_step', None
                            )
                            if labels_for_gate is None:
                                labels_for_gate = phase_labels  # fallback
                            phase_gate_loss = fa.compute_phase_gate_loss(labels_for_gate)
                        except Exception as e:
                            print(f"[Warning] compute_phase_gate_loss 失敗: {e}")
                        # ★ 方向 B：解耦正則化（使用 _logits_grad_list，更穩定）
                        try:
                            if fa._last_logits_grad is not None:
                                decorr_loss = fa.compute_decorrelation_loss()
                        except Exception as e:
                            print(f"[Warning] compute_decorrelation_loss 失敗: {e}")

                    if activation_weights is not None:
                        # ── 方案 B：多樣性損失（兩種模式都計算）────────────── #
                        diversity_loss = fa.compute_diversity_loss(activation_weights) \
                            if hasattr(fa, 'compute_diversity_loss') \
                            else torch.tensor(0.0, device=self.device)

                        if self.use_dynamic_loss_weighter:
                            # ── ★ 動態權重路徑（Dyn 模式）────────────────────── #
                            loss_dict_for_ema = {
                                'var'       : loss_var.item(),
                                'nino'      : loss_nino.item(),
                                'probe'     : probe_loss.item(),
                                'phase_gate': phase_gate_loss.item(),
                                'diversity' : diversity_loss.item(),
                                'decorr'    : decorr_loss.item(),
                            }
                            dyn_w = self.loss_weighter.get_weights(
                                loss_dict_for_ema, global_step
                            )
                        else:
                            # ── ★ 固定權重路徑（No-Dyn 模式）─────────────────── #
                            dyn_w = {
                                'probe'     : self.probe_weight,
                                'phase_gate': self.phase_gate_weight,
                                'diversity' : self.diversity_weight,
                                'decorr'    : self.decorr_weight,
                            }

                        combine_loss = (combine_loss
                                        + dyn_w['diversity']  * diversity_loss
                                        + dyn_w['probe']      * probe_loss
                                        + dyn_w['phase_gate'] * phase_gate_loss
                                        + dyn_w['decorr']     * decorr_loss)
                    else:
                        # ── fallback：無激活函數權重時使用固定權重 ────────── #
                        diversity_loss = torch.tensor(0.0, device=self.device)
                        dyn_w = {
                            'probe'     : self.probe_weight,
                            'phase_gate': self.phase_gate_weight,
                            'diversity' : self.diversity_weight,
                            'decorr'    : self.decorr_weight,
                        }
                        if hasattr(self.mymodel, 'fusion_activation') and \
                           hasattr(self.mymodel.fusion_activation, 'selector'):
                            combine_loss = self.dynamic_loss(
                                pred=(var_pred, nino_pred),
                                target=(var_true, nino_true),
                                activation_weights=activation_weights,
                                climate_labels=climate_labels.to(self.device)
                            )
                        else:
                            combine_loss = (combine_loss
                                            + dyn_w['probe']      * probe_loss
                                            + dyn_w['phase_gate'] * phase_gate_loss
                                            + dyn_w['decorr']     * decorr_loss)

                    # ── 反向傳播 ─────────────────────────────────────────── #
                    self.opt.zero_grad()
                    combine_loss.backward()
                    self.opt.step()

                    # ── ★ 更新動態損失 EMA（僅 Dyn 模式，在 optimizer.step() 之後）── #
                    if self.use_dynamic_loss_weighter:
                        self.loss_weighter.step({
                            'var'       : loss_var.item(),
                            'nino'      : loss_nino.item(),
                            'probe'     : probe_loss.item(),
                            'phase_gate': phase_gate_loss.item(),
                            'diversity' : diversity_loss.item()
                                          if isinstance(diversity_loss, torch.Tensor)
                                          else float(diversity_loss),
                            'decorr'    : decorr_loss.item(),
                        })

                    # ── ✅ 方案 1：每個 batch 後更新溫度（文件四新增）──────── #
                    if hasattr(fa, 'update_temperature'):
                        fa.update_temperature()

                    # 更新 Teacher Forcing 調度器
                    self.tf_scheduler.step()

                    # ── ✅ MLflow 記錄（整合版：文件三基礎 + 文件四詳細指標 + ★ 方案 F）── #
                    mlflow.log_metric("Train/Loss_Var",        loss_var.item(),        step=global_step)
                    mlflow.log_metric("Train/Loss_Nino",       loss_nino.item(),       step=global_step)
                    mlflow.log_metric("Train/Combine_Loss",    combine_loss.item(),    step=global_step)
                    mlflow.log_metric("Train/Score",           score,                  step=global_step)
                    mlflow.log_metric("Train/TF_Ratio",        current_tf_ratio,       step=global_step)
                    mlflow.log_metric("Train/Loss_Residual",   residual_loss.item(),   step=global_step)
                    mlflow.log_metric("Train/Diversity_Loss",  diversity_loss.item(),  step=global_step)
                    mlflow.log_metric("Train/Probe_Loss",      probe_loss.item(),      step=global_step)
                    # ★ 方案 F + 方向 B 損失記錄
                    mlflow.log_metric("Train/Phase_Gate_Loss", phase_gate_loss.item(), step=global_step)
                    mlflow.log_metric("Train/Decorr_Loss",     decorr_loss.item(),     step=global_step)

                    # ── ★ 動態權重記錄 ─────────────────────────────────────── #
                    if 'dyn_w' in locals():
                        if self.use_dynamic_loss_weighter:
                            # Dyn 模式：記錄 EMA 動態調節的權重和 curriculum 進度
                            self.loss_weighter.log_to_mlflow(dyn_w, global_step)
                            summary = self.loss_weighter.get_summary()
                            mlflow.log_metric(
                                "Train/Curriculum_Progress",
                                summary['curriculum_progress'],
                                step=global_step
                            )
                        # 兩種模式都記錄加權後貢獻（方便對照比較）
                        mlflow.log_metric(
                            "Train/WeightedContrib_probe",
                            dyn_w.get('probe', 0) * probe_loss.item(),
                            step=global_step
                        )
                        mlflow.log_metric(
                            "Train/WeightedContrib_phase_gate",
                            dyn_w.get('phase_gate', 0) * phase_gate_loss.item(),
                            step=global_step
                        )
                        mlflow.log_metric(
                            "Train/WeightedContrib_diversity",
                            dyn_w.get('diversity', 0) * diversity_loss.item()
                            if isinstance(diversity_loss, torch.Tensor) else 0.0,
                            step=global_step
                        )
                        mlflow.log_metric(
                            "Train/WeightedContrib_decorr",
                            dyn_w.get('decorr', 0) * decorr_loss.item(),
                            step=global_step
                        )

                    # ★ 方案 F：每 200 步打印相位分佈統計
                    if activation_weights is not None and global_step % 200 == 0:
                        self._print_phase_gate_stats(activation_weights, phase_labels)

                    # ✅ 溫度記錄（文件四新增）
                    if hasattr(fa, 'current_temperature'):
                        mlflow.log_metric("Train/Temperature", fa.current_temperature, step=global_step)

                    # ✅ 激活函數權重詳細記錄（文件四新增）
                    if activation_weights is not None:
                        avg_w = activation_weights.mean(dim=0)
                        mlflow.log_metric("Train/Weight_ReLU",  avg_w[0].item(),    step=global_step)
                        mlflow.log_metric("Train/Weight_Snake", avg_w[1].item(),    step=global_step)
                        mlflow.log_metric("Train/Weight_Std",   avg_w.std().item(), step=global_step)
                        mlflow.log_metric("Train/Min_Weight",   avg_w.min().item(), step=global_step)
                        # 文件三備用：逐索引記錄
                        for i, w in enumerate(avg_w):
                            mlflow.log_metric(f"Train/Activation_Weight_{i}", w.item(), step=global_step)

                        # ★ 方案 F：按相位分別記錄激活函數權重（關鍵監控指標）
                        # activation_weights 由 weights_history 取出，已 .cpu()，
                        # 故 phase_mask 也需移到 CPU 才能做 boolean indexing
                        for phase_idx, phase_name in enumerate(['LaNina', 'Neutral', 'ElNino']):
                            phase_mask = (phase_labels == phase_idx).cpu()
                            if phase_mask.sum() > 0:
                                phase_weights = activation_weights[phase_mask]
                                phase_avg_w = phase_weights.mean(dim=0)
                                mlflow.log_metric(
                                    f"Train/Weight_ReLU_{phase_name}",
                                    phase_avg_w[0].item(), step=global_step
                                )
                                mlflow.log_metric(
                                    f"Train/Weight_Snake_{phase_name}",
                                    phase_avg_w[1].item(), step=global_step
                                )

                    # ✅ Snake 參數監控（文件四新增：判斷 Snake 是否在學習）
                    if hasattr(fa, 'activations') and len(fa.activations) > 1 and \
                       hasattr(fa.activations[1], 'get_params_info'):
                        p = fa.activations[1].get_params_info()
                        mlflow.log_metric("Train/Snake_a",     p['a'],     step=global_step)
                        mlflow.log_metric("Train/Snake_scale", p['scale'], step=global_step)
                        mlflow.log_metric("Train/Snake_bias",  p['bias'],  step=global_step)

                    # ✅ 訓練階段進度（文件四新增）
                    if hasattr(fa, 'get_training_stats') and global_step % 100 == 0:
                        stats = fa.get_training_stats()
                        mlflow.log_metric("Train/Warmup_Progress",
                                          stats['warmup_progress'],  step=global_step)
                        mlflow.log_metric("Train/Balance_Progress",
                                          stats['balance_progress'], step=global_step)

                    # ── 保存權重歷史 ──────────────────────────────────────── #
                    if activation_weights is not None:
                        epoch_activation_weights.append(activation_weights.detach().cpu())

                    global_step += 1

                    # ── 打印訓練進度 ──────────────────────────────────────── #
                    if j % 100 == 0:
                        print(f"\n-->batch:{j}")
                        print(f"  loss_var={loss_var:.4f}  loss_nino={loss_nino:.4f}  "
                              f"diversity={diversity_loss:.4f}  probe={probe_loss:.4f}  "
                              f"phase_gate={phase_gate_loss:.4f}  decorr={decorr_loss:.4f}  score={score:.3f}")
                        if activation_weights is not None:
                            avg_w = activation_weights.mean(dim=0)
                            temp_str = f"  temp={fa.current_temperature:.3f}" \
                                if hasattr(fa, 'current_temperature') else ""
                            print(f"  weights: ReLU={avg_w[0]:.4f}  Snake={avg_w[1]:.4f}{temp_str}")

                        if torch.cuda.is_available():
                            print(f"CUDA 內存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB 已分配, "
                                  f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB 已保留")

                        torch.cuda.empty_cache()
                        gc.collect()

                    # ── 密集驗證 ─────────────────────────────────────────── #
                    if (i_epoch + 1 >= 4) and (j + 1) % 400 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                        eval_results = self.evaluate(dataloader_eval)

                        print(
                            f"-->Evaluation... \nloss_var:{eval_results['loss_var']:.3f} "
                            f"\nloss_nino:{eval_results['loss_nino']:.3f} "
                            f"\nloss_com:{eval_results['combine_loss']:.3f} "
                            f"\nscore:{eval_results['score']:.3f}"
                        )
                        mlflow.log_metric("Eval/Loss_Var",     eval_results['loss_var'],     step=global_step)
                        mlflow.log_metric("Eval/Loss_Nino",    eval_results['loss_nino'],    step=global_step)
                        mlflow.log_metric("Eval/Combine_Loss", eval_results['combine_loss'], step=global_step)
                        mlflow.log_metric("Eval/Score",        eval_results['score'],        step=global_step)

                        # ✅ 驗證集激活函數權重記錄（文件三原始 + 文件四格式統一）
                        if 'activation_weights' in eval_results and \
                           eval_results['activation_weights'] is not None:
                            weights_array = eval_results['activation_weights']
                            if isinstance(weights_array, torch.Tensor):
                                weights_array = weights_array.cpu().numpy()
                            if weights_array.ndim > 1:
                                weights_array = weights_array.mean(axis=0)
                            for i in range(len(weights_array)):
                                try:
                                    mlflow.log_metric(f"Eval/Activation_Weight_{i}",
                                                      float(weights_array[i]), step=global_step)
                                except Exception as e:
                                    print(f"Warning: 無法記錄評估權重 {i}: {e}")

                        if eval_results['score'] > best:
                            torch.save(self.mymodel.state_dict(), chk_path)
                            best = eval_results['score']
                            count = 0
                            print(f"\nsaving model with dynamic activation (E+F)...")

                        torch.cuda.empty_cache()
                        gc.collect()

                # ── 記錄每個 epoch 的激活函數權重 ──────────────────────── #
                if epoch_activation_weights:
                    epoch_weights = torch.stack(epoch_activation_weights).mean(dim=0)
                    self.activation_weights_history.append(epoch_weights)

                    if (i_epoch + 1) % 10 == 0:
                        self.plot_activation_weights(i_epoch)

                torch.cuda.empty_cache()
                gc.collect()

                # ── 每個 epoch 結束後的評估 ──────────────────────────────── #
                eval_results = self.evaluate(dataloader_eval)

                print(
                    f"\n-->epoch{i_epoch} end... \nloss_var:{eval_results['loss_var']:.3f} "
                    f"\nloss_nino:{eval_results['loss_nino']:.3f} "
                    f"\nloss_com:{eval_results['combine_loss']:.3f} "
                    f"\nscore: {eval_results['score']:.3f}"
                )

                mlflow.log_metric("Epoch/Loss_Var",     eval_results['loss_var'],     step=i_epoch)
                mlflow.log_metric("Epoch/Loss_Nino",    eval_results['loss_nino'],    step=i_epoch)
                mlflow.log_metric("Epoch/Combine_Loss", eval_results['combine_loss'], step=i_epoch)
                mlflow.log_metric("Epoch/Score",        eval_results['score'],        step=i_epoch)

                if 'activation_weights' in eval_results and \
                   eval_results['activation_weights'] is not None:
                    weights_array = eval_results['activation_weights']
                    if isinstance(weights_array, torch.Tensor):
                        weights_array = weights_array.cpu().numpy()
                    if weights_array.ndim > 1:
                        weights_array = weights_array.mean(axis=0)
                    for i in range(len(weights_array)):
                        mlflow.log_metric(f"Epoch/Activation_Weight_{i}",
                                          float(weights_array[i]), step=i_epoch)

                if eval_results['score'] <= best:
                    count += 1
                    print(f"\nsc is not increase for {count} epoch")
                else:
                    count = 0
                    print(
                        f"\nsc is increase from {best:.3f} to {eval_results['score']:.3f} "
                        f"with dynamic activation (E+F) \nsaving model...\n"
                    )
                    torch.save(self.mymodel.state_dict(), chk_path)
                    best = eval_results['score']

                if count == self.mypara.patience:
                    print(
                        f"\n-----!!!early stopping reached, max(sceval)= {best:.3f} "
                        f"with dynamic activation (E+F)!!!-----"
                    )
                    break

        # 繪製最終的激活函數權重變化圖
        self.plot_activation_weights(num_epochs, is_final=True)

        # ══════════════════════════════════════════════════════════════════════
        # ✅ Linear Probing 實驗（文件三功能 + ★ Gate Logit Probe，訓練結束後執行）
        # ══════════════════════════════════════════════════════════════════════
        if run_linear_probe:
            print(f"\n{'='*60}")
            print("訓練完成，開始執行 Linear Probing 實驗（含 ★ Gate Logit Probe）...")
            print(f"{'='*60}")

            try:
                best_state = torch.load(chk_path, map_location=self.device)
                self.mymodel.load_state_dict(best_state)
                print(f"  ✅ 已載入最佳模型權重: {chk_path}")
            except Exception as e:
                print(f"  ⚠️  載入最佳模型失敗，使用當前模型權重: {e}")

            torch.cuda.empty_cache()
            gc.collect()

            try:
                probe, probe_history, probe_test_acc = run_linear_probe_experiment(
                    model=self.mymodel,
                    mypara=self.mypara,
                    save_dir=linear_probe_save_dir,
                    num_probe_epochs=linear_probe_epochs,
                    probe_lr=linear_probe_lr,
                    test_size=0.2,
                    batch_size=64
                )
                print(f"\n✅ Linear Probing 實驗完成！最佳探針準確率: {probe_test_acc:.2f}%")
            except Exception as e:
                print(f"\n⚠️  Linear Probing 實驗發生錯誤: {e}")
            finally:
                torch.cuda.empty_cache()
                gc.collect()

        return best

    def _print_phase_gate_stats(self, activation_weights, phase_labels):
        """
        ★ 2激活函數版：ReLU(0) vs LearnedSnake(1)
        El Niño → ReLU(0)
        La Niña / Neutral → LearnedSnake(1)
        """
        # 相位設定
        phase_names    = ['La Niña', 'Neutral', 'El Niño']
        target_indices = [1,          1,          0       ]   # ★ 2act版映射
        target_names   = ['Snake(1)', 'Snake(1)', 'ReLU(0)']
    
        phase_labels_cpu = phase_labels.cpu()
        n_act = activation_weights.shape[1]   # 應為 2
    
        print(f"\n  [2激活函數監控] ENSO 相位 → Gate 權重（{n_act}個激活函數）:")
        for phase_idx, (name, tgt_idx, tgt_name) in enumerate(
                zip(phase_names, target_indices, target_names)):
            mask = (phase_labels_cpu == phase_idx)
            if mask.sum() > 0:
                phase_w  = activation_weights[mask].mean(dim=0)
                pred_idx = phase_w.argmax().item()
                correct  = "✅" if pred_idx == tgt_idx else "❌"
                w_str = "  ".join([f"[{i}]={phase_w[i]:.3f}" for i in range(n_act)])
                print(f"    {name:10s} target={tgt_name:10s}: {w_str}  {correct}")
            else:
                print(f"    {name:10s}: 本批次無樣本")

    def evaluate(self, dataloader, save_weights=False, epoch=None):
        """
        評估模型

        整合版：保留文件三基本功能 + 文件四的逐樣本權重收集與 save_weights 功能
        """
        self.mymodel.eval()
        nino_pred = []
        var_pred  = []
        nino_true = []
        var_true  = []
        activation_weights_list = []
        sample_indices = []          # ✅ 文件四新增：樣本索引記錄
        current_sample_idx = 0      # ✅ 文件四新增

        with torch.no_grad():
            for input_var, var_true1 in dataloader:
                # ✅ 文件四新增：每個 batch 前清空 weights_history，避免累積
                if hasattr(self.mymodel, 'fusion_activation') and \
                   hasattr(self.mymodel.fusion_activation, 'weights_history'):
                    self.mymodel.fusion_activation.weights_history = []

                SST = var_true1[:, :, self.sstlevel]
                nino_true1 = SST[
                    :, :,
                    self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])

                # ✅ 整合：接收三回傳值（文件四格式）
                out_var, residual_loss, probe_loss = self.mymodel(
                    input_var.float().to(self.device),
                    predictand=None,
                    train=False,
                )

                SST_out = out_var[:, :, self.sstlevel]
                out_nino = SST_out[
                    :, :,
                    self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])

                var_true.append(var_true1.cpu())
                nino_true.append(nino_true1.cpu())
                var_pred.append(out_var.cpu())
                nino_pred.append(out_nino.cpu())

                # ✅ 文件四方式：逐樣本收集激活函數權重
                if hasattr(self.mymodel, 'fusion_activation') and \
                   hasattr(self.mymodel.fusion_activation, 'weights_history'):
                    for weights in self.mymodel.fusion_activation.weights_history:
                        # weights 形狀: [current_batch_size, num_activations]
                        for b in range(weights.size(0)):
                            activation_weights_list.append(weights[b].detach().cpu())
                            sample_indices.append(current_sample_idx)
                            current_sample_idx += 1
                else:
                    # 文件三備用路徑
                    if hasattr(self.mymodel, 'fusion_activation') and \
                       hasattr(self.mymodel.fusion_activation, 'selector'):
                        w = self.mymodel.fusion_activation.selector.weights_history
                        if w:
                            activation_weights_list.append(w[-1].detach().cpu())

                if len(var_pred) % 5 == 0:
                    torch.cuda.empty_cache()

            var_pred  = torch.cat(var_pred,  dim=0)
            nino_pred = torch.cat(nino_pred, dim=0)
            nino_true = torch.cat(nino_true, dim=0)
            var_true  = torch.cat(var_true,  dim=0)

            var_pred_gpu  = var_pred.to(self.device)
            nino_pred_gpu = nino_pred.to(self.device)
            nino_true_gpu = nino_true.to(self.device)
            var_true_gpu  = var_true.to(self.device)

            ninosc     = self.calscore(nino_pred_gpu, nino_true_gpu)
            loss_var   = self.loss_var(var_pred_gpu, var_true_gpu, residual_losses=None).item()
            loss_nino  = self.loss_nino(nino_pred_gpu, nino_true_gpu).item()
            combine_loss = self.combine_loss(loss_var, loss_nino)

            del var_pred_gpu, nino_pred_gpu, nino_true_gpu, var_true_gpu
            torch.cuda.empty_cache()

            # ✅ 整合：計算平均激活函數權重 + 文件四的 save_weights 功能
            activation_weights = None
            if activation_weights_list:
                # 文件四：所有元素都是 [num_activations]，可直接 stack
                try:
                    activation_weights = torch.stack(activation_weights_list).mean(dim=0).numpy()
                except Exception:
                    # 文件三備用：直接 mean
                    activation_weights = torch.stack(activation_weights_list).mean(dim=0).numpy()

                # ✅ 文件四新增：保存詳細的權重記錄（用於物理分析）
                if save_weights and epoch is not None:
                    weights_array = torch.stack(activation_weights_list).numpy()
                    save_path = os.path.join(self.mypara.model_savepath, 'activation_weights')
                    os.makedirs(save_path, exist_ok=True)

                    np.save(
                        os.path.join(save_path, f'weights_epoch_{epoch}.npy'),
                        weights_array
                    )
                    np.save(
                        os.path.join(save_path, f'sample_indices_epoch_{epoch}.npy'),
                        np.array(sample_indices)
                    )
                    print(f"\n已保存激活函数权重: {save_path}/weights_epoch_{epoch}.npy")
                    print(f"权重数组形状: {weights_array.shape} [樣本數 x 激活函數數量]")

        return {
            'var_pred'          : var_pred,
            'nino_pred'         : nino_pred,
            'loss_var'          : loss_var,
            'loss_nino'         : loss_nino,
            'combine_loss'      : combine_loss,
            'score'             : ninosc,
            'activation_weights': activation_weights
        }

    def plot_activation_weights(self, epoch, is_final=False):
        """繪製激活函數權重變化圖"""
        if not self.activation_weights_history:
            return

        save_dir = os.path.join(self.mypara.model_savepath, 'activation_weights')
        os.makedirs(save_dir, exist_ok=True)

        activation_names = list(self.activation_configs.keys())

        plt.figure(figsize=(10, 6))
        weights = torch.stack(self.activation_weights_history).numpy()
        for i in range(weights.shape[1]):
            plt.plot(weights[:, i],
                     label=activation_names[i] if i < len(activation_names) else f"Activation {i}")

        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.title('Activation Function Weights over Training')
        plt.legend()
        plt.grid(True)

        if is_final:
            plt.savefig(os.path.join(save_dir, 'final_activation_weights.png'))
        else:
            plt.savefig(os.path.join(save_dir, f'activation_weights_epoch_{epoch}.png'))

        plt.close()


# =============================================================================
# 主訓練函數（整合版：保留文件三的 linear probe 參數）
# =============================================================================

def train_dynamic_activation(
    run_linear_probe_after_training=True,
    linear_probe_save_dir='./linear_probe_results',
    linear_probe_epochs=50,
    linear_probe_lr=1e-3
):
    """
    訓練動態激活函數模型，並可選擇性地在訓練後執行 Linear Probing 實驗

    Args:
        run_linear_probe_after_training: 是否在訓練結束後自動執行 Linear Probing 實驗
        linear_probe_save_dir: Linear Probe 結果儲存目錄
        linear_probe_epochs: Linear Probe 訓練 epoch 數
        linear_probe_lr: Linear Probe 學習率
    """
    print(f"\n{'='*60}")
    print(f"開始訓練動態激活函數模型（整合版 E+F）")
    print(f"  ★ 方案 F：ENSO 相位監督門控（第五次校正）")
    print(f"    La Niña  → Snake_highfreq (index 2)")
    print(f"    Neutral  → Snake_lowfreq  (index 1)")
    print(f"    El Niño  → ReLU           (index 0)")
    print(f"{'='*60}")

    torch.cuda.empty_cache()
    gc.collect()

    print(mypara.__dict__)
    print(f"\nloading pre-train dataset for dynamic activation model...")
    traindataset = make_dataset2(mypara)
    print(traindataset.selectregion())

    print(f"\nloading evaluation dataset for dynamic activation model...")
    evaldataset = make_testdataset(mypara, ngroup=100)
    print(evaldataset.selectregion())

    try:
        trainer = DynamicActivationTrainer(mypara)
        best_score = trainer.train_model(
            dataset_train=traindataset,
            dataset_eval=evaldataset,
            num_epochs=50,
            run_linear_probe=run_linear_probe_after_training,
            linear_probe_save_dir=linear_probe_save_dir,
            linear_probe_epochs=linear_probe_epochs,
            linear_probe_lr=linear_probe_lr
        )

        print(f"\n動態激活函數模型（E+F）訓練完成！最佳評分: {best_score:.3f}")
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"\n內存不足錯誤: {e}")
            print("\n嘗試減小批次大小或模型大小後重新運行")
        else:
            print(f"\n運行時錯誤: {e}")
    except Exception as e:
        print(f"\n發生錯誤: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # ★ 控制是否在訓練後自動執行 Linear Probing 實驗
    # 設為 True  → 訓練完成後自動執行（使用最佳模型權重）
    # 設為 False → 僅執行訓練，不執行探針實驗
    RUN_LINEAR_PROBE = True

    train_dynamic_activation(
        run_linear_probe_after_training=RUN_LINEAR_PROBE,
        linear_probe_save_dir='./linear_probe_results',
        linear_probe_epochs=50,
        linear_probe_lr=1e-3
    )
