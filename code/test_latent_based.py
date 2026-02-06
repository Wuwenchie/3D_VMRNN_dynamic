from myconfig import mypara
import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from my_tools import cal_ninoskill2, runmean
from func_for_prediction_latent_based import func_pre, classify_enso_phase
import xarray as xr
import seaborn as sns
from scipy import stats

mpl.use("Agg")
plt.rc("font", family="sans-serif")
mpl.rc("image", cmap="RdYlBu_r")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".pkl":
                L.append(os.path.join(root, file))
    return L


def plot_activation_heatmap_by_enso_phase(activation_weights_dict, save_path, 
                                          threshold=0.5, activation_names=None):
    """
    绘制激活函数权重的 ENSO 相位热图
    
    图表设计：
    - X 轴： 模型选用的激活函数类型（ReLU, Snake, GELU）
    - Y 轴： 非监督聚类分出的政权（Type 1: El Niño, Type 2: Neutral, Type 3: La Niña）
    - 数值： 平均权重
    
    Args:
        activation_weights_dict: 包含权重和 Niño 指数的字典
        save_path: 保存路径
        threshold: ENSO 分类阈值
        activation_names: 激活函数名称列表
    """
    if activation_weights_dict is None:
        print("警告：没有激活权重数据，跳过热图绘制")
        return
    
    weights = activation_weights_dict['weights']  # [num_samples, lead_max, num_activations]
    nino_indices = activation_weights_dict['nino_indices']  # [num_samples, lead_max]
    
    # 展平数据：将所有样本和时间步合并
    weights_flat = weights.reshape(-1, weights.shape[-1])  # [num_samples * lead_max, num_activations]
    nino_flat = nino_indices.flatten()  # [num_samples * lead_max]
    
    # 根据 Niño 指数分类 ENSO 相位
    enso_phases = classify_enso_phase(nino_flat, threshold=threshold)
    
    # 计算每个相位下每个激活函数的平均权重
    num_activations = weights_flat.shape[1]
    phase_names = ['La Niña', 'Neutral', 'El Niño']
    
    if activation_names is None:
        activation_names = ['ReLU', 'Learned Snake'][:num_activations]
    
    # 创建热图矩阵 [3 phases, num_activations]
    heatmap_data = np.zeros((3, num_activations))
    phase_counts = np.zeros(3)
    
    for phase_idx in range(3):
        mask = (enso_phases == phase_idx)
        if np.sum(mask) > 0:
            heatmap_data[phase_idx] = weights_flat[mask].mean(axis=0)
            phase_counts[phase_idx] = np.sum(mask)
    
    # 绘制热图
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(num_activations))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(activation_names, fontsize=12)
    ax.set_yticklabels(phase_names, fontsize=12)
    
    # 在每个单元格中显示数值
    for i in range(3):
        for j in range(num_activations):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xlabel('Activation Function Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('ENSO Phase', fontsize=14, fontweight='bold')
    ax.set_title('Mean Activation Weights by ENSO Phase', fontsize=16, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Weight', rotation=270, labelpad=20, fontsize=12)
    
    # 在右侧添加样本数量注释
    for i, count in enumerate(phase_counts):
        ax.text(num_activations + 0.3, i, f'n={int(count)}',
               ha="left", va="center", fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ENSO 相位热图已保存至: {save_path}")
    print("\n各相位平均权重:")
    for i, phase_name in enumerate(phase_names):
        print(f"{phase_name} (n={int(phase_counts[i])}): {heatmap_data[i]}")


def plot_time_series_alignment(activation_weights_dict, nino34_full, save_path,
                               lead_month=0, activation_idx=1, activation_name='Snake'):
    """
    绘制时间轴对齐图
    
    上图：历史 Niño 3.4 指数
    下图：模型产生的激活函数权重
    
    Args:
        activation_weights_dict: 包含权重和 Niño 指数的字典
        nino34_full: 完整的 Niño 3.4 时间序列
        save_path: 保存路径
        lead_month: 选择哪个预测月份的权重（0 表示第1个月）
        activation_idx: 选择哪个激活函数（1 表示 Snake）
        activation_name: 激活函数名称
    """
    if activation_weights_dict is None:
        print("警告：没有激活权重数据，跳过时间序列对齐图")
        return
    
    weights = activation_weights_dict['weights']  # [num_samples, lead_max, num_activations]
    nino_indices = activation_weights_dict['nino_indices']  # [num_samples, lead_max]
    sample_indices = activation_weights_dict['sample_indices']  # [num_samples]
    
    # 提取特定 lead_month 的权重
    weights_selected = weights[:, lead_month, activation_idx]  # [num_samples]
    nino_selected = nino_indices[:, lead_month]  # [num_samples]
    
    # 创建时间轴（假设从1980年开始）
    start_year = 1980
    num_samples = len(sample_indices)
    
    # 计算每个样本对应的年份和月份
    # 考虑到数据起始索引的偏移
    data_start_idx = 12 + mypara.output_length - 1
    time_points = []
    
    for i, sample_idx in enumerate(sample_indices):
        # 计算这个样本对应的实际时间索引
        actual_idx = data_start_idx + sample_idx + lead_month
        year = start_year + actual_idx // 12
        month = (actual_idx % 12) + 1
        time_points.append(year + (month - 1) / 12)
    
    time_points = np.array(time_points)
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=300, sharex=True)
    
    # 上图：Niño 3.4 指数
    ax1.plot(time_points, nino_selected, color='black', linewidth=1.5, label='Niño 3.4 Index')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='El Niño Threshold')
    ax1.axhline(y=-0.5, color='blue', linestyle=':', linewidth=1, alpha=0.5, label='La Niña Threshold')
    ax1.fill_between(time_points, 0, nino_selected, where=(nino_selected > 0.5), 
                     color='red', alpha=0.2, label='El Niño')
    ax1.fill_between(time_points, 0, nino_selected, where=(nino_selected < -0.5), 
                     color='blue', alpha=0.2, label='La Niña')
    
    ax1.set_ylabel('Niño 3.4 Index (°C)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Time-Series Alignment: Niño 3.4 vs {activation_name} Activation Weight (Lead {lead_month+1} month)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(linestyle=':', alpha=0.5)
    
    # 下图：激活函数权重
    ax2.plot(time_points, weights_selected, color='darkred', linewidth=1.5, 
            label=f'{activation_name} Weight')
    ax2.axhline(y=weights_selected.mean(), color='gray', linestyle='--', 
               linewidth=1, alpha=0.5, label=f'Mean = {weights_selected.mean():.3f}')
    ax2.fill_between(time_points, 0, weights_selected, alpha=0.3, color='darkred')
    
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'{activation_name} Weight', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(linestyle=':', alpha=0.5)
    
    # 设置 x 轴刻度
    year_ticks = np.arange(1985, 2025, 5)
    ax2.set_xticks(year_ticks)
    ax2.set_xlim(time_points[0], time_points[-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"时间序列对齐图已保存至: {save_path}")
    
    # 计算相关性
    correlation, p_value = stats.pearsonr(nino_selected, weights_selected)
    print(f"\nNiño 3.4 指数与 {activation_name} 权重的相关性:")
    print(f"  Pearson 相关系数: {correlation:.4f}")
    print(f"  P 值: {p_value:.4e}")


def plot_scatter_phase_space(activation_weights_dict, save_path,
                             activation_idx=1, activation_name='Snake',
                             lead_months=[0, 5, 11, 19]):
    """
    绘制散布图（相空间）
    
    X 轴：Niño 3.4 异常值
    Y 轴：激活函数的权重
    
    Args:
        activation_weights_dict: 包含权重和 Niño 指数的字典
        save_path: 保存路径
        activation_idx: 选择哪个激活函数
        activation_name: 激活函数名称
        lead_months: 要绘制的预测月份列表
    """
    if activation_weights_dict is None:
        print("警告：没有激活权重数据，跳过散布图")
        return
    
    weights = activation_weights_dict['weights']  # [num_samples, lead_max, num_activations]
    nino_indices = activation_weights_dict['nino_indices']  # [num_samples, lead_max]
    
    # 创建子图
    n_leads = len(lead_months)
    fig, axes = plt.subplots(1, n_leads, figsize=(5*n_leads, 4), dpi=300)
    
    if n_leads == 1:
        axes = [axes]
    
    for idx, lead_month in enumerate(lead_months):
        ax = axes[idx]
        
        # 提取数据
        nino_vals = nino_indices[:, lead_month].flatten()
        weight_vals = weights[:, lead_month, activation_idx].flatten()
        
        # 根据 ENSO 相位着色
        enso_phases = classify_enso_phase(nino_vals, threshold=0.5)
        colors = ['blue', 'gray', 'red']  # La Niña, Neutral, El Niño
        phase_names = ['La Niña', 'Neutral', 'El Niño']
        
        # 绘制散点图
        for phase_idx in range(3):
            mask = (enso_phases == phase_idx)
            if np.sum(mask) > 0:
                ax.scatter(nino_vals[mask], weight_vals[mask], 
                          c=colors[phase_idx], label=phase_names[phase_idx],
                          alpha=0.6, s=20)
        
        # 添加趋势线
        z = np.polyfit(nino_vals, weight_vals, 2)  # 二次拟合
        p = np.poly1d(z)
        nino_range = np.linspace(nino_vals.min(), nino_vals.max(), 100)
        ax.plot(nino_range, p(nino_range), "k--", linewidth=2, 
               alpha=0.7, label='Quadratic Fit')
        
        # 计算相关性
        correlation, p_value = stats.pearsonr(nino_vals, weight_vals)
        
        # 设置标签和标题
        ax.set_xlabel('Niño 3.4 Anomaly (°C)', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel(f'{activation_name} Weight', fontsize=11, fontweight='bold')
        ax.set_title(f'Lead {lead_month+1} month\nr={correlation:.3f}, p={p_value:.2e}', 
                    fontsize=10)
        ax.grid(linestyle=':', alpha=0.5)
        
        # 添加参考线
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.5, color='red', linestyle=':', linewidth=1, alpha=0.3)
        ax.axvline(x=-0.5, color='blue', linestyle=':', linewidth=1, alpha=0.3)
        
        if idx == n_leads - 1:
            ax.legend(loc='best', fontsize=8)
    
    plt.suptitle(f'Phase Space: Niño 3.4 vs {activation_name} Activation Weight', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"散布图已保存至: {save_path}")


def analyze_activation_statistics(activation_weights_dict, save_path):
    """
    分析激活权重的统计特性并保存报告
    
    Args:
        activation_weights_dict: 包含权重和 Niño 指数的字典
        save_path: 保存路径（文本文件）
    """
    if activation_weights_dict is None:
        print("警告：没有激活权重数据，跳过统计分析")
        return
    
    weights = activation_weights_dict['weights']  # [num_samples, lead_max, num_activations]
    nino_indices = activation_weights_dict['nino_indices']  # [num_samples, lead_max]
    
    num_samples, lead_max, num_activations = weights.shape
    activation_names = ['ReLU', 'Snake', 'GELU'][:num_activations]
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("激活函数权重统计分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 整体统计
        f.write("1. 整体统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"样本数量: {num_samples}\n")
        f.write(f"预测时长: {lead_max} 个月\n")
        f.write(f"激活函数数量: {num_activations}\n\n")
        
        for i, name in enumerate(activation_names):
            weights_i = weights[:, :, i].flatten()
            f.write(f"{name} 激活函数:\n")
            f.write(f"  均值: {weights_i.mean():.4f}\n")
            f.write(f"  标准差: {weights_i.std():.4f}\n")
            f.write(f"  最小值: {weights_i.min():.4f}\n")
            f.write(f"  最大值: {weights_i.max():.4f}\n")
            f.write(f"  中位数: {np.median(weights_i):.4f}\n\n")
        
        # 2. ENSO 相位统计
        f.write("2. ENSO 相位统计\n")
        f.write("-" * 80 + "\n")
        
        nino_flat = nino_indices.flatten()
        weights_flat = weights.reshape(-1, num_activations)
        enso_phases = classify_enso_phase(nino_flat, threshold=0.5)
        
        phase_names = ['La Niña', 'Neutral', 'El Niño']
        for phase_idx, phase_name in enumerate(phase_names):
            mask = (enso_phases == phase_idx)
            count = np.sum(mask)
            f.write(f"\n{phase_name} 相位 (n={count}):\n")
            
            if count > 0:
                for i, name in enumerate(activation_names):
                    weights_phase = weights_flat[mask, i]
                    f.write(f"  {name}: 均值={weights_phase.mean():.4f}, "
                           f"标准差={weights_phase.std():.4f}\n")
        
        # 3. 相关性分析
        f.write("\n3. 激活权重与 Niño 3.4 指数的相关性\n")
        f.write("-" * 80 + "\n")
        
        for lead in [0, 5, 11, 19]:
            if lead < lead_max:
                f.write(f"\n预测第 {lead+1} 个月:\n")
                nino_lead = nino_indices[:, lead]
                
                for i, name in enumerate(activation_names):
                    weights_lead = weights[:, lead, i]
                    corr, p_val = stats.pearsonr(nino_lead, weights_lead)
                    f.write(f"  {name}: r={corr:.4f}, p={p_val:.4e}\n")
        
        # 4. 时间演变
        f.write("\n4. 权重随预测时长的演变\n")
        f.write("-" * 80 + "\n")
        
        for i, name in enumerate(activation_names):
            f.write(f"\n{name} 激活函数:\n")
            for lead in range(0, lead_max, 6):
                weights_lead = weights[:, lead, i]
                f.write(f"  第 {lead+1} 月: 均值={weights_lead.mean():.4f}, "
                       f"標準差={weights_lead.std():.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"统計分析報告已保存至: {save_path}")


# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
file_dir = './model_latent_based_1/'
files = file_name(file_dir)
file_num = len(files)
lead_max = mypara.output_length
adr_datain = "./data/GODAS_group_up150_temp_tauxy_8021_kb.nc"
adr_oridata = "./data/GODAS_up150m_temp_nino_tauxy_kb.nc"

# 加载完整的 Niño 3.4 数据用于时间序列对齐
data_ori = xr.open_dataset(adr_oridata)
nino34_full = data_ori["nino34"].values

# ---------------------------------------------------------
for i_file in files[: file_num + 1]:
    print(f"\n{'='*80}")
    print(f"處理模型: {i_file}")
    print(f"{'='*80}\n")
    
    # 创建保存目录
    analysis_dir = os.path.join(file_dir, 'physical_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # ========================================
    # 传统评估指标
    # ========================================
    fig1 = plt.figure(figsize=(5, 2.5), dpi=300)
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    
    # 运行预测并获取激活权重
    (cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true, 
     activation_weights_dict) = func_pre(
        mypara=mypara,
        adr_model=i_file,
        adr_datain=adr_datain,
        adr_oridata=adr_oridata,
        needtauxy=mypara.needtauxy,
        save_activation_weights=True
    )
    
    # -----------
    cut_nino_true_jx = deepcopy(cut_nino_true[(24 - lead_max + 1) :])
    cut_nino_pred_jx = deepcopy(cut_nino_pred[:, (24 - lead_max + 1) :])
    assert np.mod(cut_nino_true_jx.shape[0], 12) == 0
    corr = np.zeros([lead_max])
    mse = np.zeros([lead_max])
    mae = np.zeros([lead_max])
    bb = runmean(cut_nino_true_jx, 3)
    for l in range(lead_max):
        aa = runmean(cut_nino_pred_jx[l], 3)
        corr[l] = np.corrcoef(aa, bb)[0, 1]
        mse[l] = mean_squared_error(aa, bb)
        mae[l] = mean_absolute_error(aa, bb)
    
    np.savez(f"{file_dir}/metrics.npz",
        corr=corr,
        rmse=np.sqrt(mse),
        mae=mae)
    print(f"傳統指標已保存至: {file_dir}/metrics.npz")
    
    # -------------figure---------------
    ax1.plot(corr, color="C0", linestyle="-", linewidth=1, label="Corr")
    ax1.plot(mse ** 0.5, color="C2", linestyle="-", linewidth=1, label="RMSE")
    ax1.plot(mae, color="C3", linestyle="-", linewidth=1, label="MAE")
    ax1.plot(np.ones(lead_max) * 0.5, color="k", linestyle="--", linewidth=1)
    ax1.set_xlim(0, lead_max - 1)
    ax1.set_xticks(np.array([1, 5, 10, 15, 20]) - 1)
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xticklabels(np.array([1, 5, 10, 15, 20]), fontsize=9)
    ax1.set_xlabel("Prediction lead (months)", fontsize=9)

    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.31, 0.1))
    ax1.set_yticklabels(np.around(np.arange(0, 1.31, 0.1), 1), fontsize=9)
    ax1.grid(linestyle=":")
    
    # ---------skill contourf
    # 1983.1~2021.12
    long_eval_yr = 2021 - 1983 + 1
    cut_nino_true_jx = runmean(cut_nino_true_jx, 3)
    for l in range(lead_max):
        cut_nino_pred_jx[l] = runmean(cut_nino_pred_jx[l], 3)  # [lead_max,len]
    pre_nino_tg = np.zeros([long_eval_yr, 12, lead_max])
    for l in range(lead_max):
        for i in range(long_eval_yr):
            pre_nino_tg[i, :, l] = cut_nino_pred_jx[l, 12 * i : 12 * (i + 1)]
    real_nino = np.zeros([long_eval_yr, 12])
    for i in range(long_eval_yr):
        real_nino[i, :] = cut_nino_true_jx[12 * i : 12 * (i + 1)]
    tem1 = deepcopy(pre_nino_tg)
    pre_nino_st = np.zeros(pre_nino_tg.shape)
    for y in range(long_eval_yr):
        for t in range(12):
            terget = t + 1
            for l in range(lead_max):
                lead = l + 1
                start_mon = terget - lead
                if -12 < start_mon <= 0:
                    start_mon += 12
                elif start_mon <= -12:
                    start_mon += 24
                pre_nino_st[y, start_mon - 1, l] = tem1[y, t, l]
    del y, t, l, start_mon, terget, lead, tem1
    tem1 = deepcopy(pre_nino_st)
    tem2 = deepcopy(real_nino)
    nino_skill = cal_ninoskill2(tem1, tem2)
    
    # ---------------figure
    ax2.contourf(
        nino_skill, levels=np.arange(0, 1.01, 0.1), extend="both", cmap="RdBu_r"
    )
    ct1 = ax2.contour(nino_skill, [0.5, 0.6, 0.7, 0.8, 0.9], colors="k", linewidths=1)
    ax2.clabel(ct1, fontsize=8, colors="k", fmt="%.1f")
    ax2.set_xlim(0, lead_max - 1)
    ax2.set_xticks(np.array([1, 5, 10, 15, 20]) - 1)
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_xticklabels(np.array([1, 5, 10, 15, 20]), fontsize=9)
    ax2.set_xlabel("Prediction lead (months)", fontsize=9)
    ax2.set_yticks(np.arange(0, 12, 1))
    y_ticklabel = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    ax2.set_yticklabels(y_ticklabel, fontsize=9)
    ax2.set_ylabel("Month", fontsize=9)
    del tem1, tem2
    legend = ax1.legend(loc="lower left", ncol=3, fontsize=5)

    _ = ax1.text(x=0.02, y=1.32, s="(a)", fontsize=9)
    _ = ax2.text(x=0.02, y=11.24, s="(b)", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{file_dir}/test_skill.png")
    plt.close(fig1)
    
    print(f"傳統技能圖已保存至: {file_dir}/test_skill.png")

    # -------------【时间序列对比图】-------------
    fig_debug, ax_debug = plt.subplots(3, 1, figsize=(8, 10), dpi=150)

    # 创建精确的年份轴
    start_year = 1980
    start_month = 1
    n_data_points = len(cut_nino_true_jx)

    years = []
    for i in range(n_data_points):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        years.append(year + (month - 1) / 12)

    years = np.array(years)

    lead_to_plot = [5, 11, 19]

    for i, lead in enumerate(lead_to_plot):
        ax_debug[i].plot(years, cut_nino_true_jx, label="Analyzed", 
                        color="black", linewidth=1.5)
        ax_debug[i].plot(years, cut_nino_pred_jx[lead], label=f"Pred", 
                        color="red", alpha=0.7)
        
        ax_debug[i].set_title(f"Analyzed data vs Model predicted (lead {lead+1} month)", 
                             fontsize=16)
        ax_debug[i].set_xlabel("Year", fontsize=14)
        ax_debug[i].set_ylabel("Niño3.4 index (°C)", fontsize=14)
        
        year_ticks = np.arange(1980, 2025, 5)
        ax_debug[i].set_xticks(year_ticks)
        ax_debug[i].set_xlim(years[0], years[-1])
        ax_debug[i].grid(linestyle=":")

    handles, labels = ax_debug[0].get_legend_handles_labels()
    fig_debug.legend(handles, labels, loc='lower center', ncol=3, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"{file_dir}/debug_pred_vs_true.png")
    plt.close(fig_debug)
    
    print(f"时间序列对比图已保存至: {file_dir}/debug_pred_vs_true.png")
    
    # ========================================
    # 物理分析可视化（新增部分）
    # ========================================
    if activation_weights_dict is not None:
        print(f"\n{'='*80}")
        print("开始物理分析可视化")
        print(f"{'='*80}\n")
        
        # 1. ENSO 相位热图
        print("1. 绘制 ENSO 相位热图...")
        plot_activation_heatmap_by_enso_phase(
            activation_weights_dict,
            save_path=os.path.join(analysis_dir, 'enso_phase_heatmap.png'),
            threshold=0.5,
            activation_names=['ReLU', 'Snake', 'GELU']
        )
        
        # 2. 时间序列对齐图（多个预测时长）
        print("\n2. 绘制时间序列对齐图...")
        for lead_month in [0, 5, 11, 19]:
            if lead_month < lead_max:
                plot_time_series_alignment(
                    activation_weights_dict,
                    nino34_full,
                    save_path=os.path.join(analysis_dir, 
                                          f'time_alignment_lead{lead_month+1}.png'),
                    lead_month=lead_month,
                    activation_idx=1,  # Snake
                    activation_name='Snake'
                )
        
        # 3. 散布图（相空间）
        print("\n3. 绘制散布图（相空间）...")
        plot_scatter_phase_space(
            activation_weights_dict,
            save_path=os.path.join(analysis_dir, 'phase_space_scatter.png'),
            activation_idx=1,  # Snake
            activation_name='Snake',
            lead_months=[0, 5, 11, 19]
        )
        
        # 4. 统计分析报告
        print("\n4. 生成统计分析报告...")
        analyze_activation_statistics(
            activation_weights_dict,
            save_path=os.path.join(analysis_dir, 'activation_statistics.txt')
        )
        
        print(f"\n{'='*80}")
        print("物理分析可视化完成！")
        print(f"{'='*80}\n")
    else:
        print("\n警告：未获取到激活权重数据，跳过物理分析可视化")
    
    print("*************" * 8)

print("\n" + "="*80)
print("评估完成！请检查以下文件:")
print("="*80)
print(f"传统评估:")
print(f"  - {file_dir}/test_skill.png (传统技能指标)")
print(f"  - {file_dir}/debug_pred_vs_true.png (时间序列对比)")
print(f"  - {file_dir}/metrics.npz (传统指标数据)")
print(f"\n物理分析:")
print(f"  - {analysis_dir}/enso_phase_heatmap.png (ENSO 相位热图)")
print(f"  - {analysis_dir}/time_alignment_lead*.png (时间序列对齐)")
print(f"  - {analysis_dir}/phase_space_scatter.png (相空间散布图)")
print(f"  - {analysis_dir}/activation_statistics.txt (统计分析报告)")
print(f"  - {file_dir}/activation_weights_test.npz (激活权重数据)")
print("="*80)
