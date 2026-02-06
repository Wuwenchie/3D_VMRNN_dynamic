from train_latent_based import create_supervised_geo_vmrnn, get_ablation_configs
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


def load_model_with_flexible_activation(model, checkpoint_path, strict=False):
    """
    灵活载入模型，处理激活函数参数不匹配的问题
    
    Args:
        model: 已初始化的模型实例
        checkpoint_path: 检查点文件路径
        strict: 是否严格载入（False允许忽略缺失的键）
    """
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    if not strict:
        # 获取当前模型的状态字典
        model_state = model.state_dict()
        
        # 过滤检查点中与当前模型匹配的参数
        filtered_checkpoint = {}
        missing_keys = []
        unexpected_keys = []
        
        for key, value in state_dict.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_checkpoint[key] = value
                else:
                    print(f"形状不匹配，跳过参数: {key}")
                    print(f"  模型形状: {model_state[key].shape}")
                    print(f"  检查点形状: {value.shape}")
                    missing_keys.append(key)
            else:
                unexpected_keys.append(key)
        
        # 检查缺失的键
        for key in model_state:
            if key not in filtered_checkpoint:
                missing_keys.append(key)
        
        # 载入过滤后的参数
        model.load_state_dict(filtered_checkpoint, strict=False)
        
        if missing_keys:
            activation_keys = [k for k in missing_keys if 'activation' in k.lower()]
            if activation_keys:
                print(f"警告：以下激活函数参数将使用随机初始化: {activation_keys}")
            other_keys = [k for k in missing_keys if 'activation' not in k.lower()]
            if other_keys:
                print(f"警告：以下参数将使用随机初始化: {other_keys}")
        if unexpected_keys:
            print(f"警告：以下参数在检查点中但不在模型中: {unexpected_keys}")
            
        return model, missing_keys, unexpected_keys
    else:
        model.load_state_dict(state_dict, strict=True)
        return model, [], []


def classify_enso_phase(nino34_values, threshold=0.5):
    """
    根据 Niño 3.4 指数对气候状态进行分类
    
    Args:
        nino34_values: Niño 3.4 指数数组
        threshold: 分类阈值（默认0.5°C）
    
    Returns:
        phases: 气候相位标签数组 (0: La Niña, 1: Neutral, 2: El Niño)
    """
    phases = np.zeros_like(nino34_values, dtype=int)
    phases[nino34_values > threshold] = 2  # El Niño
    phases[nino34_values < -threshold] = 0  # La Niña
    phases[np.abs(nino34_values) <= threshold] = 1  # Neutral
    return phases


def func_pre(mypara, adr_model, adr_datain, adr_oridata, needtauxy, 
             activation_config=None, strict_loading=False,
             save_activation_weights=True):
    """
    改进的预测函数，支持灵活的模型载入和激活权重分析
    
    Args:
        mypara: 模型参数
        adr_model: 模型检查点路径
        adr_datain: 输入数据地址
        adr_oridata: 原始数据地址
        needtauxy: 是否需要tau数据
        activation_config: 激活函数配置
        strict_loading: 是否严格载入模型参数
        save_activation_weights: 是否保存激活权重
    
    Returns:
        cut_var_pred: 预测的变量
        cut_var_true: 真实的变量
        cut_nino_pred: 预测的 Niño 指数
        cut_nino_true: 真实的 Niño 指数
        activation_weights_dict: 包含激活权重和对应 Niño 指数的字典
    """
    lead_max = mypara.output_length
    
    # 数据载入部分
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

    # 数据集载入
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

    # 使用 GeoVMRNN_Enhance 模型
    if activation_config is None:
        # 使用物理门控动态激活函数配置
        try:
            activation_config = get_ablation_configs()['physical_gating']
            print(f"使用物理门控激活函数配置: {activation_config}")
        except:
            activation_config = {'fusion': 'dynamic', 'cnn': 'relu', 'prediction': 'relu'}
            print(f"使用默认动态激活函数配置: {activation_config}")
    else:
        print(f"使用自定义激活函数配置: {activation_config}")

    mymodel = create_supervised_geo_vmrnn(mypara, activation_config=activation_config)

    # 设置设备
    if torch.cuda.is_available() and hasattr(mypara, 'device'):
        device = mypara.device
    else:
        device = 'cpu'
    
    mymodel = mymodel.to(device)

    # 载入模型权重
    try:
        mymodel, missing_keys, unexpected_keys = load_model_with_flexible_activation(
            mymodel, adr_model, strict=False
        )
        print("模型灵活模式载入完成")
    except Exception as e:
        print(f"模型载入失败: {e}")
        raise e

    mymodel.eval()
    torch.set_grad_enabled(False)
    
    # 初始化变量
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
    
    # 用于存储激活函数权重和对应的 Niño 指数
    all_activation_weights = []
    all_nino_indices = []
    all_sample_indices = []
    
    ii = 0
    iii = 0
    
    # 清空模型内部的权重历史记录
    if hasattr(mymodel, 'fusion_activation') and hasattr(mymodel.fusion_activation, 'weights_history'):
        mymodel.fusion_activation.weights_history = []
        print("已清空模型内部权重历史记录")
    
    with torch.no_grad():
        for batch_idx, input_var in enumerate(dataloader_test):
            input_tensor = input_var.float().to(device)
            batch_size = input_tensor.size(0)
            
            # 计算当前批次对应的真实 Niño 指数
            start_idx = 12 + lead_max - 1
            batch_nino_indices = []
            
            for b in range(batch_size):
                sample_idx = ii + b
                # 确保不会超出 nino34 的范围
                end_idx = start_idx + sample_idx + lead_max
                if end_idx <= len(nino34):
                    nino_idx = nino34[start_idx + sample_idx : end_idx]
                    # 确保长度正确
                    if len(nino_idx) == lead_max:
                        batch_nino_indices.append(nino_idx)
                    else:
                        # 如果长度不对，填充或截断
                        print(f"警告：批次 {batch_idx}, 样本 {b}: nino 长度 {len(nino_idx)} != {lead_max}")
                        if len(nino_idx) < lead_max:
                            # 填充
                            nino_idx_padded = np.zeros(lead_max)
                            nino_idx_padded[:len(nino_idx)] = nino_idx
                            batch_nino_indices.append(nino_idx_padded)
                        else:
                            # 截断
                            batch_nino_indices.append(nino_idx[:lead_max])
                else:
                    # 超出范围，用零填充
                    print(f"警告：批次 {batch_idx}, 样本 {b}: 索引超出范围")
                    batch_nino_indices.append(np.zeros(lead_max))
            
            # 调试：检查第一个批次
            if batch_idx == 0:
                print(f"\n第一个批次调试信息:")
                print(f"  batch_size: {batch_size}")
                print(f"  lead_max: {lead_max}")
                print(f"  nino34 总长度: {len(nino34)}")
                print(f"  batch_nino_indices 数量: {len(batch_nino_indices)}")
                if len(batch_nino_indices) > 0:
                    print(f"  第一个 nino 长度: {len(batch_nino_indices[0])}")
            
            # 记录批次开始前的权重数量
            if hasattr(mymodel, 'fusion_activation') and hasattr(mymodel.fusion_activation, 'weights_history'):
                weights_count_before = len(mymodel.fusion_activation.weights_history)
            else:
                weights_count_before = 0
            
            # 前向传播
            model_output = mymodel(
                input_tensor,
                predictand=None,
                train=False,
            )
            
            # 处理模型输出
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                out_var = model_output[0]
            else:
                out_var = model_output
            
            # 记录批次结束后的权重数量
            if hasattr(mymodel, 'fusion_activation') and hasattr(mymodel.fusion_activation, 'weights_history'):
                weights_count_after = len(mymodel.fusion_activation.weights_history)
                new_weights_count = weights_count_after - weights_count_before
            else:
                weights_count_after = 0
                new_weights_count = 0
            
            # 收集激活函数权重
            if new_weights_count > 0 and hasattr(mymodel, 'fusion_activation'):
                # 提取这个批次新增的权重
                batch_weights_raw = mymodel.fusion_activation.weights_history[weights_count_before:weights_count_after]
                
                # 展平和规范化权重
                batch_weights_normalized = []
                
                for w in batch_weights_raw:
                    # 转换为 numpy 并确保是 2D
                    if isinstance(w, torch.Tensor):
                        w_cpu = w.cpu()
                    else:
                        w_cpu = w
                    
                    # 处理不同的形状
                    if w_cpu.dim() == 1:
                        # [num_activations] -> [1, num_activations]
                        batch_weights_normalized.append(w_cpu.unsqueeze(0).numpy())
                    elif w_cpu.dim() == 2:
                        # 可能是 [1, num_activations] 或 [B, num_activations]
                        if w_cpu.size(0) == 1:
                            # 已经是正确形状
                            batch_weights_normalized.append(w_cpu.numpy())
                        else:
                            # 需要拆分
                            for i in range(w_cpu.size(0)):
                                batch_weights_normalized.append(w_cpu[i:i+1].numpy())
                    else:
                        print(f"警告：遇到意外的权重维度: {w_cpu.dim()}, 形状: {w_cpu.shape}")
                        continue
                
                # 期望的权重数量
                expected_count = batch_size * lead_max
                actual_count = len(batch_weights_normalized)
                
                if actual_count == expected_count:
                    # 完美匹配，直接处理
                    try:
                        # 合并为单个数组 [batch_size*lead_max, num_activations]
                        all_batch_weights = np.concatenate(batch_weights_normalized, axis=0)
                        
                        # 重塑为 [batch_size, lead_max, num_activations]
                        weights_reshaped = all_batch_weights.reshape(batch_size, lead_max, -1)
                        
                        # 存储时验证长度
                        for b in range(batch_size):
                            # 确保对应的 nino 索引存在且长度正确
                            if b < len(batch_nino_indices):
                                nino_b = batch_nino_indices[b]
                                if len(nino_b) == lead_max:
                                    all_activation_weights.append(weights_reshaped[b])
                                    all_nino_indices.append(nino_b)
                                    all_sample_indices.append(ii + b)
                                else:
                                    print(f"警告：批次 {batch_idx}, 样本 {b}: nino 长度 {len(nino_b)} != {lead_max}")
                            else:
                                print(f"警告：批次 {batch_idx}, 样本 {b}: 没有对应的 nino 索引")
                            
                    except Exception as e:
                        print(f"批次 {batch_idx} 处理权重时出错: {e}")
                        print(f"  形状信息: {[w.shape for w in batch_weights_normalized[:5]]}")
                        
                elif actual_count > 0:
                    # 数量不匹配，尝试修复
                    print(f"批次 {batch_idx} 权重数量不匹配: 期望 {expected_count}, 实际 {actual_count}")
                    
                    # 检查是否可以整除
                    if actual_count % lead_max == 0:
                        actual_batch_size = actual_count // lead_max
                        print(f"  尝试使用实际批次大小 {actual_batch_size} 进行处理")
                        
                        try:
                            all_batch_weights = np.concatenate(batch_weights_normalized, axis=0)
                            weights_reshaped = all_batch_weights.reshape(actual_batch_size, lead_max, -1)
                            
                            # 验证并存储
                            for b in range(min(actual_batch_size, batch_size)):
                                if b < len(batch_nino_indices) and len(batch_nino_indices[b]) == lead_max:
                                    all_activation_weights.append(weights_reshaped[b])
                                    all_nino_indices.append(batch_nino_indices[b])
                                    all_sample_indices.append(ii + b)
                            
                            print(f"  成功处理 {min(actual_batch_size, len(batch_nino_indices))} 个样本")
                        except Exception as e:
                            print(f"  修复失败: {e}")
                    else:
                        print(f"  无法修复：权重数量 {actual_count} 不能被 lead_max={lead_max} 整除")
            
            # 保存预测结果
            ii += out_var.shape[0]
            if torch.cuda.is_available():
                var_pred[iii:ii] = out_var.cpu().detach().numpy()
            else:
                var_pred[iii:ii] = out_var.detach().numpy()
            iii = ii
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"处理批次 {batch_idx + 1}/{len(dataloader_test)}, 已收集权重样本: {len(all_activation_weights)}")
    
    # 打印最终统计
    print(f"\n权重收集完成:")
    print(f"  总样本数: {test_group}")
    print(f"  成功收集权重的样本数: {len(all_activation_weights)}")
    
    if len(all_activation_weights) < test_group * 0.9:
        print(f"  警告：收集的权重样本数量较少，可能影响分析质量")
    
    # 整理激活权重数据
    activation_weights_dict = None
    if all_activation_weights:
        # 检查并修复数据形状
        print(f"\n检查收集到的数据...")
        
        # 检查 weights 形状
        weights_shapes = [w.shape for w in all_activation_weights[:5]]
        print(f"  前5个权重形状: {weights_shapes}")
        
        # 检查 nino_indices 长度
        nino_lengths = [len(n) for n in all_nino_indices[:5]]
        print(f"  前5个 nino 长度: {nino_lengths}")
        
        # 转换为 numpy 数组，处理可能的长度不一致问题
        try:
            # 尝试直接转换
            weights_array = np.array(all_activation_weights)
            nino_array = np.array(all_nino_indices)
            
            activation_weights_dict = {
                'weights': weights_array,  # [num_samples, lead_max, num_activations]
                'nino_indices': nino_array,   # [num_samples, lead_max]
                'sample_indices': np.array(all_sample_indices)  # [num_samples]
            }
            
            print(f"\n成功收集到激活权重:")
            print(f"  权重形状: {activation_weights_dict['weights'].shape}")
            print(f"  Nino 形状: {activation_weights_dict['nino_indices'].shape}")
            print(f"  样本数: {len(all_sample_indices)}")
            
        except ValueError as e:
            # 如果直接转换失败，说明长度不一致，需要修复
            print(f"  数据形状不一致，尝试修复...")
            
            # 找出最常见的长度
            from collections import Counter
            nino_length_counts = Counter(nino_lengths)
            most_common_length = nino_length_counts.most_common(1)[0][0]
            print(f"  最常见的 lead_max: {most_common_length}")
            
            # 过滤掉长度不匹配的样本
            valid_indices = []
            valid_weights = []
            valid_ninos = []
            valid_sample_ids = []
            
            for i in range(len(all_activation_weights)):
                if len(all_nino_indices[i]) == most_common_length and \
                   all_activation_weights[i].shape[0] == most_common_length:
                    valid_indices.append(i)
                    valid_weights.append(all_activation_weights[i])
                    valid_ninos.append(all_nino_indices[i])
                    valid_sample_ids.append(all_sample_indices[i])
            
            print(f"  过滤后样本数: {len(valid_weights)}/{len(all_activation_weights)}")
            
            if len(valid_weights) > 0:
                activation_weights_dict = {
                    'weights': np.array(valid_weights),
                    'nino_indices': np.array(valid_ninos),
                    'sample_indices': np.array(valid_sample_ids)
                }
                
                print(f"\n修复后的激活权重:")
                print(f"  权重形状: {activation_weights_dict['weights'].shape}")
                print(f"  Nino 形状: {activation_weights_dict['nino_indices'].shape}")
            else:
                print(f"  错误：没有有效的样本")
                activation_weights_dict = None
        
        # 保存权重数据
        if activation_weights_dict is not None and save_activation_weights:
            weight_save_path = f"{os.path.dirname(adr_model)}/activation_weights_test.npz"
            np.savez(
                weight_save_path,
                weights=activation_weights_dict['weights'],
                nino_indices=activation_weights_dict['nino_indices'],
                sample_indices=activation_weights_dict['sample_indices']
            )
            print(f"激活权重已保存至: {weight_save_path}")
    
    del out_var, input_var
    del mymodel, dataCS, dataloader_test

    # 数据处理部分
    len_data = test_group - lead_max
    print("len_data:", len_data)
    
    start_idx = 12 + lead_max - 1
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
        cut_nino_true,
        activation_weights_dict
    )


def load_model_predictions(
        model_path, 
        mypara, 
        adr_datain, 
        adr_oridata, 
        activation_config=None,
        save_activation_weights=True
    ):
    """
    载入单一模型的预测结果，包含激活权重分析
    
    Args:
        model_path: 模型文件路径
        mypara: 模型参数
        adr_datain: 输入数据地址
        adr_oridata: 原始数据地址
        activation_config: 激活函数配置
        save_activation_weights: 是否保存激活权重
    
    Returns:
        cut_nino_true_jx: 真实 Niño 指数（3个月移动平均）
        cut_nino_pred_jx: 预测 Niño 指数（3个月移动平均）
        activation_weights_dict: 激活权重字典
    """
    
    result = func_pre(
        mypara=mypara,
        adr_model=model_path,
        adr_datain=adr_datain,
        adr_oridata=adr_oridata,
        needtauxy=mypara.needtauxy,
        activation_config=activation_config,
        save_activation_weights=save_activation_weights,
    )
    
    cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true, activation_weights_dict = result
    
    lead_max = mypara.output_length
    cut_nino_true_jx = deepcopy(cut_nino_true[(24 - lead_max + 1):])
    cut_nino_pred_jx = deepcopy(cut_nino_pred[:, (24 - lead_max + 1):])
    
    # 3个月移动平均
    cut_nino_true_jx = runmean(cut_nino_true_jx, 3)
    for l in range(lead_max):
        cut_nino_pred_jx[l] = runmean(cut_nino_pred_jx[l], 3)
    
    return cut_nino_true_jx, cut_nino_pred_jx, activation_weights_dict
