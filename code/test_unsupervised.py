"""
無監督動態激活函數模型測試腳本
評估模型效能並分析 regime 和激活權重
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import xarray as xr
import os
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

from myconfig import mypara
from my_tools import cal_ninoskill2, runmean
from UnsupervisedDynamicActivation import RegimeAnalyzer
from LoadData import make_testdataset

# Matplotlib 設置
mpl.use("Agg")
plt.rc("font", family="sans-serif")
mpl.rc("image", cmap="RdYlBu_r")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


class UnsupervisedModelTester:
    """無監督模型測試器"""
    
    def __init__(self, mypara, model_path, adr_oridata):
        self.mypara = mypara
        self.model_path = model_path
        self.adr_oridata = adr_oridata
        self.device = mypara.device
        
        # 載入原始數據用於計算標準差
        data_ori = xr.open_dataset(adr_oridata)
        self.nino34_true = data_ori["nino34"].values
        self.stdtemp = data_ori["stdtemp"][
            mypara.lev_range[0]:mypara.lev_range[1]
        ].values
        self.stdtemp = np.nanmean(self.stdtemp, axis=(1, 2))
        
        if mypara.needtauxy:
            self.stdtaux = np.nanmean(data_ori["stdtaux"].values, axis=(0, 1))
            self.stdtauy = np.nanmean(data_ori["stdtauy"].values, axis=(0, 1))
            self.stds = np.concatenate(
                (self.stdtaux[None], self.stdtauy[None], self.stdtemp), axis=0
            )
            self.sst_lev = 2
        else:
            self.stds = self.stdtemp
            self.sst_lev = 0
        
        # 創建 regime 分析器
        self.regime_analyzer = RegimeAnalyzer()
        
        # 用於存儲測試過程中的信息
        self.activation_weights_history = []
        self.regime_labels_history = []
        self.regime_probs_history = []
        
    def load_model(self):
        """載入無監督模型"""
        from train_unsupervised import GeoVMRNN_Unsupervised
        from GeoVMRNN_resnet import create_resnet_geo_vmrnn
        
        # 創建基礎模型
        base_model = create_resnet_geo_vmrnn(self.mypara, activation_config='relu')
        
        # 包裝為無監督版本
        self.model = GeoVMRNN_Unsupervised(
            base_model=base_model,
            latent_dim=256,
            num_regimes=3
        ).to(self.device)
        
        # 載入權重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"模型已載入: {self.model_path}")
        print(f"模型參數量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def test_model(self, test_dataset):
        """測試模型並收集預測結果"""
        dataloader = DataLoader(
            test_dataset, 
            batch_size=self.mypara.batch_size_eval, 
            shuffle=False
        )
        
        lead_max = self.mypara.output_length
        test_group = len(test_dataset)
        
        # 初始化預測結果
        if self.mypara.needtauxy:
            n_lev = self.mypara.lev_range[1] - self.mypara.lev_range[0] + 2
        else:
            n_lev = self.mypara.lev_range[1] - self.mypara.lev_range[0]
        
        var_pred = np.zeros([
            test_group, lead_max, n_lev,
            self.mypara.lat_range[1] - self.mypara.lat_range[0],
            self.mypara.lon_range[1] - self.mypara.lon_range[0]
        ])
        
        print("\n開始測試...")
        ii = 0
        iii = 0
        
        with torch.no_grad():
            for batch_idx, input_var in enumerate(dataloader):
                input_tensor = input_var.float().to(self.device)
                
                # 前向傳播
                out_var, _, activation_info_list = self.model(
                    input_tensor,
                    predictand=None,
                    train=False,
                )
                
                # 保存預測結果
                ii += out_var.shape[0]
                var_pred[iii:ii] = out_var.cpu().detach().numpy()
                iii = ii
                
                # 收集激活信息（使用最後一個時間步）
                if activation_info_list and len(activation_info_list) > 0:
                    last_info = activation_info_list[-1]
                    
                    self.activation_weights_history.append(
                        last_info['activation_weights'].cpu().numpy()
                    )
                    self.regime_labels_history.append(
                        last_info['regime_labels'].cpu().numpy()
                    )
                    self.regime_probs_history.append(
                        last_info['regime_probs'].cpu().numpy()
                    )
                    
                    # 添加到 regime 分析器
                    SST = input_tensor[:, -1, self.sst_lev]  # 最後時間步的 SST
                    self.regime_analyzer.add_data(
                        last_info['regime_labels'],
                        SST,
                        (
                            self.mypara.lat_nino_relative[0],
                            self.mypara.lat_nino_relative[1],
                            self.mypara.lon_nino_relative[0],
                            self.mypara.lon_nino_relative[1]
                        )
                    )
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  已處理 {ii}/{test_group} 個樣本")
        
        print(f"測試完成！共處理 {test_group} 個樣本\n")
        
        # 轉換為 numpy arrays
        self.activation_weights_history = np.concatenate(
            self.activation_weights_history, axis=0
        )
        self.regime_labels_history = np.concatenate(
            self.regime_labels_history, axis=0
        )
        self.regime_probs_history = np.concatenate(
            self.regime_probs_history, axis=0
        )
        
        return var_pred
    
    def compute_metrics(self, var_pred):
        """計算評估指標"""
        lead_max = self.mypara.output_length
        test_group = var_pred.shape[0]
        len_data = test_group - lead_max
        
        # 截取真實數據
        start_idx = 12 + lead_max - 1
        cut_nino_true = self.nino34_true[start_idx:start_idx + len_data]
        
        # 計算預測的 Nino 指數
        cut_nino_pred = np.zeros([lead_max, len_data])
        for i in range(lead_max):
            l = i + 1
            cut_var_pred_i = var_pred[lead_max - l:test_group - l, i] * \
                            self.stds[None, :, None, None]
            
            cut_nino_pred[i] = np.nanmean(
                cut_var_pred_i[
                    :, self.sst_lev,
                    self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]
                ],
                axis=(1, 2)
            )
        
        # 應用3個月移動平均
        cut_nino_true_jx = deepcopy(cut_nino_true[(24 - lead_max + 1):])
        cut_nino_pred_jx = deepcopy(cut_nino_pred[:, (24 - lead_max + 1):])
        
        cut_nino_true_jx = runmean(cut_nino_true_jx, 3)
        for l in range(lead_max):
            cut_nino_pred_jx[l] = runmean(cut_nino_pred_jx[l], 3)
        
        # 計算評估指標
        corr = np.zeros(lead_max)
        rmse = np.zeros(lead_max)
        mae = np.zeros(lead_max)
        
        for l in range(lead_max):
            aa = cut_nino_pred_jx[l]
            bb = cut_nino_true_jx
            corr[l] = np.corrcoef(aa, bb)[0, 1]
            rmse[l] = np.sqrt(mean_squared_error(aa, bb))
            mae[l] = mean_absolute_error(aa, bb)
        
        # 計算技巧分數
        nino_skill = self._compute_skill_score(cut_nino_pred_jx, cut_nino_true_jx)
        
        return {
            'corr': corr,
            'rmse': rmse,
            'mae': mae,
            'nino_skill': nino_skill,
            'cut_nino_pred': cut_nino_pred_jx,
            'cut_nino_true': cut_nino_true_jx
        }
    
    def _compute_skill_score(self, pre_nino, real_nino):
        """計算技巧分數（按月份和提前期）"""
        lead_max = pre_nino.shape[0]
        long_eval_yr = len(real_nino) // 12
        
        # 重塑數據
        pre_nino_tg = np.zeros([long_eval_yr, 12, lead_max])
        for l in range(lead_max):
            for i in range(long_eval_yr):
                pre_nino_tg[i, :, l] = pre_nino[l, 12 * i:12 * (i + 1)]
        
        real_nino_reshaped = np.zeros([long_eval_yr, 12])
        for i in range(long_eval_yr):
            real_nino_reshaped[i, :] = real_nino[12 * i:12 * (i + 1)]
        
        # 轉換為起始月份格式
        pre_nino_st = np.zeros(pre_nino_tg.shape)
        for y in range(long_eval_yr):
            for t in range(12):
                target = t + 1
                for l in range(lead_max):
                    lead = l + 1
                    start_mon = target - lead
                    if -12 < start_mon <= 0:
                        start_mon += 12
                    elif start_mon <= -12:
                        start_mon += 24
                    pre_nino_st[y, start_mon - 1, l] = pre_nino_tg[y, t, l]
        
        # 計算技巧分數
        nino_skill = cal_ninoskill2(pre_nino_st, real_nino_reshaped)
        
        return nino_skill
    
    def plot_results(self, metrics, save_dir):
        """繪製評估結果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 傳統評估指標圖
        self._plot_traditional_metrics(metrics, save_dir)
        
        # 2. 時間序列對比圖
        self._plot_time_series(metrics, save_dir)
        
        # 3. 激活權重分析圖
        self._plot_activation_weights(save_dir)
        
        # 4. Regime 分析圖
        self._plot_regime_analysis(save_dir)
        
        # 5. Regime vs ENSO 對應圖
        self._plot_regime_enso_correspondence(save_dir)
        
        print(f"\n所有圖表已保存至: {save_dir}")
    
    def _plot_traditional_metrics(self, metrics, save_dir):
        """繪製傳統評估指標"""
        lead_max = self.mypara.output_length
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
        
        # 左圖: Corr, RMSE, MAE
        ax1.plot(metrics['corr'], 'o-', color='C0', label='Correlation', linewidth=2)
        ax1.plot(metrics['rmse'], 's-', color='C2', label='RMSE', linewidth=2)
        ax1.plot(metrics['mae'], '^-', color='C3', label='MAE', linewidth=2)
        ax1.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='0.5 threshold')
        
        ax1.set_xlim(0, lead_max - 1)
        ax1.set_xticks(np.array([1, 5, 10, 15, 20]) - 1)
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        ax1.set_xticklabels(np.array([1, 5, 10, 15, 20]), fontsize=10)
        ax1.set_xlabel('Prediction Lead (months)', fontsize=11)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(linestyle=':', alpha=0.5)
        ax1.set_title('(a) Unsupervised Model - Metrics vs Lead Time', fontsize=11)
        
        # 右圖: 技巧分數熱圖
        im = ax2.contourf(
            metrics['nino_skill'], 
            levels=np.arange(0, 1.01, 0.1), 
            extend='both', 
            cmap='RdBu_r'
        )
        ct = ax2.contour(
            metrics['nino_skill'], 
            [0.5, 0.6, 0.7, 0.8, 0.9], 
            colors='k', 
            linewidths=1
        )
        ax2.clabel(ct, fontsize=8, colors='k', fmt='%.1f')
        
        ax2.set_xlim(0, lead_max - 1)
        ax2.set_xticks(np.array([1, 5, 10, 15, 20]) - 1)
        ax2.xaxis.set_minor_locator(MultipleLocator(1))
        ax2.set_xticklabels(np.array([1, 5, 10, 15, 20]), fontsize=10)
        ax2.set_xlabel('Prediction Lead (months)', fontsize=11)
        ax2.set_yticks(np.arange(0, 12, 1))
        ax2.set_yticklabels([
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ], fontsize=9)
        ax2.set_ylabel('Start Month', fontsize=11)
        ax2.set_title('(b) Skill Score by Month and Lead', fontsize=11)
        
        plt.colorbar(im, ax=ax2, label='Correlation Skill')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/test_skill_unsupervised.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存指標
        np.savez(
            f"{save_dir}/metrics_unsupervised.npz",
            corr=metrics['corr'],
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            skill_score=metrics['nino_skill']
        )
        print(f"✓ 傳統評估指標圖已保存")
    
    def _plot_time_series(self, metrics, save_dir):
        """繪製時間序列對比圖"""
        cut_nino_true = metrics['cut_nino_true']
        cut_nino_pred = metrics['cut_nino_pred']
        
        # 創建年份軸
        start_year = 1983
        n_data_points = len(cut_nino_true)
        years = start_year + np.arange(n_data_points) / 12
        
        # 選擇幾個提前期繪製
        lead_to_plot = [5, 11, 19]  # 6, 12, 20個月
        
        fig, axes = plt.subplots(len(lead_to_plot), 1, figsize=(14, 10), dpi=150)
        
        for i, lead in enumerate(lead_to_plot):
            ax = axes[i]
            
            # 繪製真實值和預測值
            ax.plot(years, cut_nino_true, label='Analyzed', 
                   color='black', linewidth=2, alpha=0.8)
            ax.plot(years, cut_nino_pred[lead], label=f'Predicted (Lead {lead+1})', 
                   color='red', linewidth=1.5, alpha=0.7)
            
            # 添加 El Niño / La Niña 閾值線
            ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=-0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # 計算該提前期的相關係數
            corr = np.corrcoef(cut_nino_true, cut_nino_pred[lead])[0, 1]
            rmse = np.sqrt(mean_squared_error(cut_nino_true, cut_nino_pred[lead]))
            
            ax.set_title(
                f'Lead {lead+1} months (Corr={corr:.3f}, RMSE={rmse:.3f})',
                fontsize=12, fontweight='bold'
            )
            ax.set_ylabel('Niño3.4 Index (°C)', fontsize=11)
            ax.set_xlim(years[0], years[-1])
            ax.set_xticks(np.arange(1985, 2025, 5))
            ax.grid(linestyle=':', alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
        
        axes[-1].set_xlabel('Year', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/time_series_comparison_unsupervised.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 時間序列對比圖已保存")
    
    def _plot_activation_weights(self, save_dir):
        """繪製激活權重分析圖"""
        if len(self.activation_weights_history) == 0:
            print("  警告: 沒有激活權重數據")
            return
        
        weights = self.activation_weights_history
        n_activations = weights.shape[1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
        
        # 1. 激活權重時間序列
        ax = axes[0, 0]
        for i in range(n_activations):
            ax.plot(weights[:, i], label=f'Activation {i}', alpha=0.7, linewidth=1)
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title('(a) Activation Weights Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(linestyle=':', alpha=0.3)
        
        # 2. 激活權重分布（箱線圖）
        ax = axes[0, 1]
        ax.boxplot([weights[:, i] for i in range(n_activations)],
                   labels=[f'Act {i}' for i in range(n_activations)])
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title('(b) Distribution of Activation Weights', fontsize=12, fontweight='bold')
        ax.grid(linestyle=':', alpha=0.3, axis='y')
        
        # 3. 激活權重相關性矩陣
        ax = axes[1, 0]
        corr_matrix = np.corrcoef(weights.T)
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n_activations))
        ax.set_yticks(range(n_activations))
        ax.set_xticklabels([f'Act {i}' for i in range(n_activations)], fontsize=9)
        ax.set_yticklabels([f'Act {i}' for i in range(n_activations)], fontsize=9)
        ax.set_title('(c) Correlation Between Activations', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Correlation')
        
        # 4. Regime 與激活權重的關係
        ax = axes[1, 1]
        regime_labels = self.regime_labels_history
        unique_regimes = np.unique(regime_labels)
        
        # 為每個 regime 計算平均權重
        regime_avg_weights = []
        for regime in unique_regimes:
            mask = regime_labels == regime
            avg_weights = weights[mask].mean(axis=0)
            regime_avg_weights.append(avg_weights)
        
        regime_avg_weights = np.array(regime_avg_weights)
        
        x = np.arange(n_activations)
        width = 0.25
        for i, regime in enumerate(unique_regimes):
            offset = (i - len(unique_regimes)/2) * width
            ax.bar(x + offset, regime_avg_weights[i], width, 
                  label=f'Regime {regime}', alpha=0.8)
        
        ax.set_xlabel('Activation Function', fontsize=11)
        ax.set_ylabel('Average Weight', fontsize=11)
        ax.set_title('(d) Activation Weights by Regime', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Act {i}' for i in range(n_activations)], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(linestyle=':', alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/activation_weights_analysis.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存權重數據
        np.save(f"{save_dir}/activation_weights.npy", weights)
        print(f"✓ 激活權重分析圖已保存")
    
    def _plot_regime_analysis(self, save_dir):
        """繪製 Regime 分析圖"""
        self.regime_analyzer.plot_regime_analysis(
            save_path=f"{save_dir}/regime_analysis_test.png"
        )
        print(f"✓ Regime 分析圖已保存")
    
    def _plot_regime_enso_correspondence(self, save_dir):
        """繪製 Regime 與 ENSO 對應關係"""
        analysis = self.regime_analyzer.analyze_regime_enso_correspondence()
        
        if not analysis:
            print("  警告: 無法生成 Regime-ENSO 對應分析")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        
        # 1. 各 Regime 的平均 Niño 指數
        ax = axes[0]
        regimes = sorted(analysis['regime_nino_means'].keys())
        nino_means = [analysis['regime_nino_means'][r] for r in regimes]
        
        colors = ['blue' if m < -0.5 else 'orange' if m > 0.5 else 'gray' 
                 for m in nino_means]
        bars = ax.bar(regimes, nino_means, color=colors, alpha=0.7, edgecolor='black')
        
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, 
                  label='El Niño threshold')
        ax.axhline(y=-0.5, color='blue', linestyle='--', linewidth=1.5, 
                  label='La Niña threshold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Regime', fontsize=12)
        ax.set_ylabel('Mean Niño3.4 Index (°C)', fontsize=12)
        ax.set_title('(a) Average Niño Index per Regime', fontsize=13, fontweight='bold')
        ax.set_xticks(regimes)
        ax.legend(fontsize=10)
        ax.grid(linestyle=':', alpha=0.3, axis='y')
        
        # 2. ENSO Phase 分布（堆疊柱狀圖）
        ax = axes[1]
        el_nino_pct = [analysis['regime_enso_distribution'][r]['el_nino'] * 100 
                      for r in regimes]
        neutral_pct = [analysis['regime_enso_distribution'][r]['neutral'] * 100 
                      for r in regimes]
        la_nina_pct = [analysis['regime_enso_distribution'][r]['la_nina'] * 100 
                      for r in regimes]
        
        width = 0.6
        ax.bar(regimes, el_nino_pct, width, label='El Niño', 
              color='orange', alpha=0.8)
        ax.bar(regimes, neutral_pct, width, bottom=el_nino_pct, 
              label='Neutral', color='gray', alpha=0.8)
        ax.bar(regimes, la_nina_pct, width, 
              bottom=np.array(el_nino_pct) + np.array(neutral_pct),
              label='La Niña', color='blue', alpha=0.8)
        
        ax.set_xlabel('Regime', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('(b) ENSO Phase Distribution per Regime', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(regimes)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=10)
        ax.grid(linestyle=':', alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/regime_enso_correspondence.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Regime-ENSO 對應圖已保存")
        
        # 打印分析報告
        self.regime_analyzer.print_analysis_report()
    
    def generate_report(self, metrics, save_dir):
        """生成文字報告"""
        report_path = f"{save_dir}/test_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("無監督動態激活函數模型測試報告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"模型路徑: {self.model_path}\n")
            f.write(f"測試樣本數: {len(self.regime_labels_history)}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("1. 預測效能指標\n")
            f.write("-"*80 + "\n")
            
            for lead in [0, 5, 11, 19]:
                if lead < len(metrics['corr']):
                    f.write(f"\nLead {lead+1} months:\n")
                    f.write(f"  Correlation: {metrics['corr'][lead]:.4f}\n")
                    f.write(f"  RMSE: {metrics['rmse'][lead]:.4f}\n")
                    f.write(f"  MAE: {metrics['mae'][lead]:.4f}\n")
            
            # 平均效能
            f.write(f"\n平均效能 (所有提前期):\n")
            f.write(f"  Mean Correlation: {np.mean(metrics['corr']):.4f}\n")
            f.write(f"  Mean RMSE: {np.mean(metrics['rmse']):.4f}\n")
            f.write(f"  Mean MAE: {np.mean(metrics['mae']):.4f}\n")
            
            # Regime 分析
            f.write("\n" + "-"*80 + "\n")
            f.write("2. Regime 分析\n")
            f.write("-"*80 + "\n\n")
            
            regime_labels = self.regime_labels_history
            unique_regimes, counts = np.unique(regime_labels, return_counts=True)
            
            f.write("Regime 分布:\n")
            for regime, count in zip(unique_regimes, counts):
                pct = count / len(regime_labels) * 100
                f.write(f"  Regime {regime}: {count} 樣本 ({pct:.2f}%)\n")
            
            # 激活權重統計
            f.write("\n" + "-"*80 + "\n")
            f.write("3. 激活權重統計\n")
            f.write("-"*80 + "\n\n")
            
            weights = self.activation_weights_history
            n_activations = weights.shape[1]
            
            for i in range(n_activations):
                f.write(f"\nActivation {i}:\n")
                f.write(f"  Mean: {np.mean(weights[:, i]):.4f}\n")
                f.write(f"  Std: {np.std(weights[:, i]):.4f}\n")
                f.write(f"  Min: {np.min(weights[:, i]):.4f}\n")
                f.write(f"  Max: {np.max(weights[:, i]):.4f}\n")
            
            # Regime-ENSO 對應分析
            f.write("\n" + "-"*80 + "\n")
            f.write("4. Regime-ENSO 對應分析\n")
            f.write("-"*80 + "\n\n")
            
            analysis = self.regime_analyzer.analyze_regime_enso_correspondence()
            if analysis:
                f.write("各 Regime 的平均 Niño3.4 指數:\n")
                for regime in sorted(analysis['regime_nino_means'].keys()):
                    mean_nino = analysis['regime_nino_means'][regime]
                    f.write(f"  Regime {regime}: {mean_nino:.4f}°C\n")
                
                f.write("\nENSO Phase 分布:\n")
                for regime in sorted(analysis['regime_enso_distribution'].keys()):
                    dist = analysis['regime_enso_distribution'][regime]
                    f.write(f"  Regime {regime}:\n")
                    f.write(f"    El Niño: {dist['el_nino']*100:.2f}%\n")
                    f.write(f"    Neutral: {dist['neutral']*100:.2f}%\n")
                    f.write(f"    La Niña: {dist['la_nina']*100:.2f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("報告生成完成\n")
            f.write("="*80 + "\n")
        
        print(f"✓ 測試報告已保存: {report_path}")


def main():
    """主測試流程"""
    # 設定參數
    print("="*80)
    print("無監督動態激活函數模型測試")
    print("="*80 + "\n")
    
    # 模型路徑
    model_path = "./saved_models/unsupervised_best_model.pth"
    
    # 檢查模型是否存在
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型檔案 {model_path}")
        return
    
    # 數據路徑
    adr_oridata = mypara.adr_oridata
    
    # 創建測試器
    tester = UnsupervisedModelTester(mypara, model_path, adr_oridata)
    
    # 載入模型
    print("\n[步驟 1/5] 載入模型...")
    tester.load_model()
    
    # 準備測試數據
    print("\n[步驟 2/5] 準備測試數據...")
    test_dataset = make_testdataset(mypara)
    print(f"測試集大小: {len(test_dataset)} 個樣本")
    
    # 執行測試
    print("\n[步驟 3/5] 執行模型測試...")
    var_pred = tester.test_model(test_dataset)
    
    # 計算指標
    print("\n[步驟 4/5] 計算評估指標...")
    metrics = tester.compute_metrics(var_pred)
    
    # 輸出關鍵指標
    print("\n關鍵評估指標:")
    print(f"  6個月預測相關係數: {metrics['corr'][5]:.4f}")
    print(f"  12個月預測相關係數: {metrics['corr'][11]:.4f}")
    print(f"  20個月預測相關係數: {metrics['corr'][19]:.4f}")
    print(f"  平均相關係數: {np.mean(metrics['corr']):.4f}")
    
    # 繪製結果
    print("\n[步驟 5/5] 生成分析圖表...")
    save_dir = "./test_results_unsupervised"
    tester.plot_results(metrics, save_dir)
    
    # 生成報告
    tester.generate_report(metrics, save_dir)
    
    print("\n" + "="*80)
    print("測試完成！")
    print(f"所有結果已保存至: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
