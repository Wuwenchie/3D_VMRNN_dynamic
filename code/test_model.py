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
from func_for_prediction import func_pre
import xarray as xr

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


# --------------------------------------------------------
file_dir = './model/'
files = file_name(file_dir)
file_num = len(files)
lead_max = mypara.output_length
adr_datain = (
    "./data/GODAS_group_up150_temp_tauxy_8021_kb.nc"
)
adr_oridata = "./data/GODAS_up150m_temp_nino_tauxy_kb.nc"
# ---------------------------------------------------------
for i_file in files[: file_num + 1]:
    fig1 = plt.figure(figsize=(5, 2.5), dpi=300)
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    (cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true,) = func_pre(
        mypara=mypara,
        adr_model=i_file,
        adr_datain=adr_datain,
        adr_oridata=adr_oridata,
        needtauxy=mypara.needtauxy,
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
    print(f"Saved metrics to metrics.npz")
    # del aa, bb
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
    ax2.clabel(
        ct1,
        fontsize=8,
        colors="k",
        fmt="%.1f",
    )
    ax2.set_xlim(0, lead_max - 1)
    ax2.set_xticks(np.array([1, 5, 10, 15, 20]) - 1)
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_xticklabels(np.array([1, 5, 10, 15, 20]), fontsize=9)
    ax2.set_xlabel("Prediction lead (months)", fontsize=9)
    ax2.set_yticks(np.arange(0, 12, 1))
    y_ticklabel = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ax2.set_yticklabels(y_ticklabel, fontsize=9)
    ax2.set_ylabel("Month", fontsize=9)
    del tem1, tem2
    legend = ax1.legend(
        loc="lower left",
        ncol=3,
        fontsize=5,
    )

    _ = ax1.text(x=0.02, y=1.32, s="(a)", fontsize=9)
    _ = ax2.text(x=0.02, y=11.24, s="(b)", fontsize=9)

    plt.tight_layout()
    # plt.savefig("./model/residual_adaptive/test_skill_residual_relu.png")
    plt.savefig(f"{file_dir}/test_skill.png")
    # plt.show()
    print("*************" * 8)

   # -------------【新增：畫 cut_nino_true vs cut_nino_pred】-------------
    fig_debug, ax_debug = plt.subplots(3, 1, figsize=(8, 10), dpi=150)

    # 創建精確的年份軸（每個月份對應正確的年份位置）
    start_year = 1980
    start_month = 1  # 假設從1月開始，你可以根據實際情況調整
    n_data_points = len(cut_nino_true_jx)

    # 創建每個數據點對應的年份
    years = []
    for i in range(n_data_points):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        years.append(year + (month - 1) / 12)  # 年份 + 月份的小數部分

    years = np.array(years)

    lead_to_plot = [5, 11, 19]
    all_preds = {
        "true": cut_nino_true_jx,
        "pred": cut_nino_pred_jx,
    }

    for i, lead in enumerate(lead_to_plot):
        # 使用年份作為x軸
        ax_debug[i].plot(years, cut_nino_true_jx, label="Analyzed", color="black", linewidth=1.5)
        ax_debug[i].plot(years, cut_nino_pred_jx[lead], label=f"Pred", color="red", alpha=0.7)
        
        ax_debug[i].set_title(f"Analyzed data vs Model predicted (lead {lead+1} month)", fontsize=16)
        ax_debug[i].set_xlabel("Year", fontsize=14)
        ax_debug[i].set_ylabel("Niño3.4 index (°C)", fontsize=14)
        
        # 設置x軸刻度，每5年顯示一次
        year_ticks = np.arange(1980, 2025, 5)  # 每5年一個刻度
        ax_debug[i].set_xticks(year_ticks)
        ax_debug[i].set_xlim(years[0], years[-1])
        
        ax_debug[i].grid(linestyle=":")

    # 將圖例放在最下面
    handles, labels = ax_debug[0].get_legend_handles_labels()
    fig_debug.legend(handles, labels, loc='lower center', ncol=3, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"{file_dir}/debug_pred_vs_true.png")
    plt.close(fig_debug)
    
    print("*************" * 8)

print("\nEvaluation complete! Check the following files:")
print(f"- {file_dir}/test_skill.png (Traditional skill metrics)")
print(f"- {file_dir}/debug_pred_vs_true.png (Time series comparison)")
print(f"- {file_dir}/metrics.npz (Traditional metrics)")
