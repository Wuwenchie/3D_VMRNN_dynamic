from 3D_VMRNN import GeoVMRNN_Enhance, get_ablation_configs
import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import os
from scipy import signal
from copy import deepcopy
from my_tools import runmean 


class make_dataset_test(Dataset):
    def __init__(
        self,
        address,
        needtauxy,
        lev_range=(0, 1),
        lon_range=(0, 1),
        lat_range=(0, 1),
    ):
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = lev_range
        self.lon_range = lon_range
        self.lat_range = lat_range

        temp = data_in["temperatureNor"][
            :,
            :,
            lev_range[0] : lev_range[1],
            lat_range[0] : lat_range[1],
            lon_range[0] : lon_range[1],
        ].values
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        if needtauxy:
            taux = data_in["tauxNor"][
                :,
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            # --------------
            self.dataX = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )
            del temp, taux, tauy
        else:
            self.dataX = temp
            del temp

    def getdatashape(self):
        return {
            "dataX.shape": self.dataX.shape,
        }

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.lon[self.lon_range[0]],
                self.lon[self.lon_range[1] - 1],
            ),
            "lat: {}S to {}N".format(
                self.lat[self.lat_range[0]],
                self.lat[self.lat_range[1] - 1],
            ),
            "lev: {}m to {}m".format(
                self.lev[self.lev_range[0]], self.lev[self.lev_range[1] - 1]
            ),
        }

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx]


def get_model_class_from_path(model_path):
    """
    根據模型路徑自動選擇對應的模型類別
    
    Args:
        model_path: 模型檔案路徑
    
    Returns:
        model_class: 對應的模型類別
    """
    # 將路徑轉換為小寫以便比較
    path_lower = model_path.lower()
    
    # 根據路徑中的關鍵字判斷模型類型
    if 'dynamic' in path_lower or 'enhance' in path_lower:
        return GeoVMRNN_Enhance
    else:
        # 默認返回 ConvLSTM，你可以根據需要修改
        print(f"Warning: Cannot determine model type from path {model_path}, using TraditionalConvLSTM as default")
        return TraditionalConvLSTM


def get_model_class_from_name(model_name):
    """
    根據模型名稱選擇對應的模型類別
    
    Args:
        model_name: 模型名稱字符串
    
    Returns:
        model_class: 對應的模型類別
    """
    model_name_lower = model_name.lower()
    
    if 'dynamic' in model_name_lower or 'enhance' in model_name_lower:
        return GeoVMRNN_Enhance
    else:
        print(f"Warning: Cannot determine model type from name {model_name}, using TraditionalConvLSTM as default")
        return TraditionalConvLSTM


def load_model_with_flexible_activation(model, checkpoint_path, strict=False):
    """
    靈活載入模型，處理激活函數參數不匹配的問題
    
    Args:
        model: 已初始化的模型實例
        checkpoint_path: 檢查點文件路徑
        strict: 是否嚴格載入（False允許忽略缺失的鍵）
    """
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 處理不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    if not strict:
        # 獲取當前模型的狀態字典
        model_state = model.state_dict()
        
        # 過濾檢查點中與當前模型匹配的參數
        filtered_checkpoint = {}
        missing_keys = []
        unexpected_keys = []
        
        for key, value in state_dict.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_checkpoint[key] = value
                else:
                    print(f"形狀不匹配，跳過參數: {key}")
                    print(f"  模型形狀: {model_state[key].shape}")
                    print(f"  檢查點形狀: {value.shape}")
                    missing_keys.append(key)
            else:
                unexpected_keys.append(key)
        
        # 檢查缺失的鍵
        for key in model_state:
            if key not in filtered_checkpoint:
                missing_keys.append(key)
        
        # 載入過濾後的參數
        model.load_state_dict(filtered_checkpoint, strict=False)
        
        if missing_keys:
            activation_keys = [k for k in missing_keys if 'activation' in k.lower()]
            if activation_keys:
                print(f"警告：以下激活函數參數將使用隨機初始化: {activation_keys}")
            other_keys = [k for k in missing_keys if 'activation' not in k.lower()]
            if other_keys:
                print(f"警告：以下參數將使用隨機初始化: {other_keys}")
        if unexpected_keys:
            print(f"警告：以下參數在檢查點中但不在模型中: {unexpected_keys}")
            
        return model, missing_keys, unexpected_keys
    else:
        model.load_state_dict(state_dict, strict=True)
        return model, [], []



def func_pre(mypara, adr_model, adr_datain, adr_oridata, needtauxy, 
             model_type=None, activation_config=None, strict_loading=False,
             save_activation_weights=True, analyze_frequency=True):
    """
    改進的預測函數，支持靈活的模型載入、激活權重分析和頻率分析
    
    Args:
        mypara: 模型參數
        adr_model: 模型檢查點路徑
        adr_datain: 輸入數據地址
        adr_oridata: 原始數據地址
        needtauxy: 是否需要tau數據
        activation_config: 激活函數配置
        strict_loading: 是否嚴格載入模型參數
        save_activation_weights: 是否保存激活權重
        analyze_frequency: 是否進行頻率分析
    """
    lead_max = mypara.output_length
    
    # 數據載入部分
    data_ori = xr.open_dataset(adr_oridata)
    temp_ori_region = data_ori["temperatureNor"][
        :,
        mypara.lev_range[0] : mypara.lev_range[1],
        mypara.lat_range[0] : mypara.lat_range[1],
        mypara.lon_range[0] : mypara.lon_range[1],
    ].values
    nino34 = data_ori["nino34"].values
    stdtemp = data_ori["stdtemp"][mypara.lev_range[0] : mypara.lev_range[1]].values
    stdtemp = np.nanmean(stdtemp, axis=(1, 2))
    
    if needtauxy:
        taux_ori_region = data_ori["tauxNor"][
            :,
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        tauy_ori_region = data_ori["tauyNor"][
            :,
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        stdtaux = data_ori["stdtaux"].values
        stdtaux = np.nanmean(stdtaux, axis=(0, 1))
        stdtauy = data_ori["stdtauy"].values
        stdtauy = np.nanmean(stdtauy, axis=(0, 1))

        var_ori_region = np.concatenate(
            (taux_ori_region[:, None], tauy_ori_region[:, None], temp_ori_region),
            axis=1,
        )
        del taux_ori_region, tauy_ori_region, temp_ori_region
        stds = np.concatenate((stdtaux[None], stdtauy[None], stdtemp), axis=0)
        del stdtemp, stdtauy, stdtaux
    else:
        var_ori_region = temp_ori_region
        del temp_ori_region
        stds = stdtemp
        del stdtemp

    # 數據集載入
    dataCS = make_dataset_test(
        address=adr_datain,
        needtauxy=needtauxy,
        lev_range=mypara.lev_range,
        lon_range=mypara.lon_range,
        lat_range=mypara.lat_range,
    )
    test_group = len(dataCS)
    print(dataCS.getdatashape())
    print(dataCS.selectregion())
    dataloader_test = DataLoader(
        dataCS, batch_size=mypara.batch_size_eval, shuffle=False
    )

    # 動態選擇模型類別
    if model_type is None:
        # 自動從路徑檢測模型類型
        model_class = get_model_class_from_path(adr_model)
    else:
        # 根據指定的模型類型選擇
        model_class = get_model_class_from_name(model_type)
    
    print(f"Using model class: {model_class.__name__}")
    
    if model_class == GeoVMRNN_Enhance:
        if activation_config is None:
            # 使用動態激活函數配置
            try:
                activation_config = get_ablation_configs()['dynamic_fusion']
                print(f"使用動態激活函數配置: {activation_config}")
            except:
                activation_config = {'fusion': 'dynamic', 'cnn': 'relu', 'prediction': 'relu'}
                print(f"使用默認動態激活函數配置: {activation_config}")
        else:
            print(f"使用自定義激活函數配置: {activation_config}")

        mymodel = model_class(mypara, activation_config=activation_config)
    else:
        # 其他模型不需要激活函數配置
        mymodel = model_class(mypara)

    # 設置設備
    if torch.cuda.is_available() and hasattr(mypara, 'device'):
        device = mypara.device
    else:
        device = 'cpu'
    
    mymodel = mymodel.to(device)

    # 載入模型權重
    try:
        if model_class in [GeoVMRNN_Supervised, GeoVMRNN_Enhance] and not strict_loading:
            # 對於VMRNN模型，使用靈活載入來處理激活函數參數不匹配
            mymodel, missing_keys, unexpected_keys = load_model_with_flexible_activation(
                mymodel, adr_model, strict=False
            )
            print("VMRNN模型靈活模式載入完成")
        else:
            # 其他模型或嚴格模式
            checkpoint = torch.load(adr_model, map_location=device)
            if 'model_state_dict' in checkpoint:
                mymodel.load_state_dict(checkpoint['model_state_dict'], strict=strict_loading)
            else:
                mymodel.load_state_dict(checkpoint, strict=strict_loading)
            print("模型載入成功")
    except Exception as e:
        print(f"模型載入失敗: {e}")
        raise e

    mymodel.eval()
    torch.set_grad_enabled(False)
    
    # 初始化變量
    if needtauxy:
        n_lev = mypara.lev_range[1] - mypara.lev_range[0] + 2
        sst_lev = 2
    else:
        n_lev = mypara.lev_range[1] - mypara.lev_range[0]
        sst_lev = 0
        
    var_pred = np.zeros(
        [
            test_group,
            lead_max,
            n_lev,
            mypara.lat_range[1] - mypara.lat_range[0],
            mypara.lon_range[1] - mypara.lon_range[0],
        ]
    )
    
    # 用於存儲激活函數權重（如果有）
    activation_weights = []
    
    ii = 0
    iii = 0
    with torch.no_grad():
        for input_var in dataloader_test:
            input_tensor = input_var.float().to(device)
            
            # 對於動態激活函數模型，提取氣候特徵
            climate_features = None
            if model_class == GeoVMRNN_Enhance and hasattr(mymodel, 'use_dynamic_activation'):
                try:
                    # 提取 Nino 區域的 SST 作為氣候特徵
                    climate_features = input_tensor[:, :, sst_lev, 
                        mypara.lat_nino_relative[0]:mypara.lat_nino_relative[1], 
                        mypara.lon_nino_relative[0]:mypara.lon_nino_relative[1]].mean(dim=[2, 3])
                except Exception as e:
                    print(f"提取氣候特徵時出錯: {e}")
                    climate_features = None
            
            # 前向傳播
            try:
                if climate_features is not None:
                    model_output = mymodel(
                        input_tensor,
                        predictand=None,
                        train=False,
                        climate_features=climate_features
                    )
                else:
                    model_output = mymodel(
                        input_tensor,
                        predictand=None,
                        train=False,
                    )
            except TypeError:
                # 如果模型不支持 climate_features 參數
                model_output = mymodel(
                    input_tensor,
                    predictand=None,
                    train=False,
                )
            
            # 處理模型輸出
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                out_var, weights = model_output[0], model_output[1]
                # 保存激活權重
                if weights is not None and save_activation_weights:
                    try:
                        if torch.cuda.is_available():
                            activation_weights.append(weights.cpu().detach().numpy())
                        else:
                            activation_weights.append(weights.detach().numpy())
                    except Exception as e:
                        print(f"保存激活權重時出錯: {e}")
            else:
                out_var = model_output
                
            ii += out_var.shape[0]
            if torch.cuda.is_available():
                var_pred[iii:ii] = out_var.cpu().detach().numpy()
            else:
                var_pred[iii:ii] = out_var.detach().numpy()
            iii = ii
            
    # 處理激活權重
    if activation_weights and len(activation_weights) > 0 and save_activation_weights:
        try:
            activation_weights = np.concatenate(activation_weights, axis=0)
            print(f"收集到激活權重: 形狀={activation_weights.shape}")
            # 計算平均權重
            mean_weights = np.mean(activation_weights, axis=0)
            print(f"平均激活權重: {mean_weights}")
            # 保存權重
            weight_save_path = f"{os.path.dirname(adr_model)}/activation_weights_test.npy"
            np.save(weight_save_path, activation_weights)
            print(f"激活權重已保存至: {weight_save_path}")
        except Exception as e:
            print(f"處理激活權重時出錯: {e}")
            
    del out_var, input_var
    del mymodel, dataCS, dataloader_test

    # 數據處理部分
    len_data = test_group - lead_max
    print("len_data:", len_data)
    
    start_idx = 12+lead_max-1
    cut_var_true = var_ori_region[start_idx : start_idx + len_data]
    cut_var_true = cut_var_true * stds[None, :, None, None]
    cut_nino_true = nino34[start_idx : start_idx + len_data]
    
    print('cut_nino_true:', cut_nino_true.shape[0])
    print('cut_var_true:', cut_var_true.shape[0])
    assert cut_nino_true.shape[0] == cut_var_true.shape[0] == len_data
    
    cut_var_pred = np.zeros(
        [lead_max, len_data, var_pred.shape[2], var_pred.shape[3], var_pred.shape[4]]
    )
    cut_nino_pred = np.zeros([lead_max, len_data])
    
    for i in range(lead_max):
        l = i + 1
        cut_var_pred[i] = (
            var_pred[lead_max - l : test_group - l, i] * stds[None, :, None, None]
        )
        cut_nino_pred[i] = np.nanmean(
            cut_var_pred[
                i,
                :,
                sst_lev,
                mypara.lat_nino_relative[0] : mypara.lat_nino_relative[1],
                mypara.lon_nino_relative[0] : mypara.lon_nino_relative[1],
            ],
            axis=(1, 2),
        )
    
    assert cut_var_pred.shape[1] == cut_var_true.shape[0]
    
    return (
        cut_var_pred,
        cut_var_true,
        cut_nino_pred,
        cut_nino_true
    )


# 用於主程序的模型加載函數
def load_model_predictions(
        model_path, 
        mypara, 
        adr_datain, 
        adr_oridata, 
        model_type=None, 
        activation_config=None,
        save_activation_weights=True
    ):
    """
    載入單一模型的預測結果，包含激活權重和頻率分析
    
    Args:
        model_path: 模型檔案路徑
        mypara: 模型參數
        adr_datain: 輸入數據地址
        adr_oridata: 原始數據地址
        model_type: 模型類型，如果為None則自動檢測
        activation_config: 激活函數配置 (僅用於VMRNN模型)
        save_activation_weights: 是否保存激活權重
    """
    
    result = func_pre(
        mypara=mypara,
        adr_model=model_path,
        adr_datain=adr_datain,
        adr_oridata=adr_oridata,
        needtauxy=mypara.needtauxy,
        model_type=model_type,
        activation_config=activation_config,
        save_activation_weights=save_activation_weights,
    )
    
    cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true = result
    
    lead_max = mypara.output_length
    cut_nino_true_jx = deepcopy(cut_nino_true[(24 - lead_max + 1):])
    cut_nino_pred_jx = deepcopy(cut_nino_pred[:, (24 - lead_max + 1):])
    
    # 3個月移動平均
    cut_nino_true_jx = runmean(cut_nino_true_jx, 3)
    for l in range(lead_max):
        cut_nino_pred_jx[l] = runmean(cut_nino_pred_jx[l], 3)
    
    return cut_nino_true_jx, cut_nino_pred_jx
