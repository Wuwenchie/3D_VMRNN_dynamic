import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List, Optional

class SimpleActivationSelector(nn.Module):
    """
    簡化的動態激活函數選擇器，只負責激活函數選擇
    """
    def __init__(self, activation_configs: Dict[str, Dict], hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation_configs = activation_configs
        
        # 創建各種激活函數
        self.activations = nn.ModuleDict()
        for name, config in activation_configs.items():
            self.activations[name] = self._create_activation(config)
        
        # 分類器網絡 - 用於預測當前數據屬於哪種氣候模式
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, len(activation_configs)),
            nn.Softmax(dim=1)
        )
        
        # 初始化權重
        self.weights_history = []
    
    def _create_activation(self, config):
        """
        根據配置創建激活函數
        """
        if isinstance(config, str):
            if config == 'relu':
                return nn.ReLU(inplace=True)
            elif config == 'gelu':
                return nn.GELU()
            elif config == 'silu':
                return nn.SiLU(inplace=True)
            else:
                return nn.ReLU(inplace=True)
        elif isinstance(config, dict):
            activation_type = config.get('type', 'relu')
            if activation_type == 'relu':
                return nn.ReLU(inplace=True)
            elif activation_type == 'gelu':
                return nn.GELU()
            elif activation_type == 'silu':
                return nn.SiLU(inplace=True)
            elif activation_type == 'snake_learned':
                try:
                    from GeoVMRNN_enhance import LearnedSnake
                    return LearnedSnake(in_features=1)  # 使用較小的參數
                except:
                    return nn.ReLU(inplace=True)
            elif activation_type == 'snake_fixed':
                try:
                    from GeoVMRNN_enhance import FixedSnake
                    return FixedSnake(a=config.get('params', {}).get('a', 1.0))
                except:
                    return nn.ReLU(inplace=True)
            elif activation_type == 'snake_adaptive':
                try:
                    from GeoVMRNN_enhance import AdaptiveSnake
                    return AdaptiveSnake(in_features=1)  # 使用較小的參數
                except:
                    return nn.ReLU(inplace=True)
            else:
                return nn.ReLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x, features=None):
        """
        前向傳播，動態選擇激活函數
        
        Args:
            x: 輸入張量
            features: 用於分類的特徵，如果為None則使用x
        
        Returns:
            激活後的輸出和分類權重
        """
        # 使用特徵進行分類
        if features is None:
            # 如果沒有提供特徵，使用輸入的平均值作為特徵
            if len(x.shape) > 2:  # 對於卷積層輸入
                # 對空間維度進行平均池化
                pooled = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
                if pooled.size(1) > self.hidden_dim:
                    pooled = pooled[:, :self.hidden_dim]
                elif pooled.size(1) < self.hidden_dim:
                    # 如果特徵維度小於hidden_dim，進行填充
                    padding = torch.zeros(pooled.size(0), self.hidden_dim - pooled.size(1), device=x.device)
                    pooled = torch.cat([pooled, padding], dim=1)
            else:  # 對於全連接層輸入
                pooled = x.view(x.size(0), -1)
                if pooled.size(1) > self.hidden_dim:
                    pooled = pooled[:, :self.hidden_dim]
                elif pooled.size(1) < self.hidden_dim:
                    padding = torch.zeros(pooled.size(0), self.hidden_dim - pooled.size(1), device=x.device)
                    pooled = torch.cat([pooled, padding], dim=1)
        else:
            pooled = features
        
        # 確保 pooled 的維度正確
        if pooled.dim() != 2:
            pooled = pooled.view(x.size(0), -1)
        
        # 預測分類權重
        weights = self.classifier(pooled)
        self.weights_history.append(weights.detach().mean(dim=0))
        
        # 應用每個激活函數並根據權重加權
        outputs = []
        for i, (name, activation) in enumerate(self.activations.items()):
            try:
                activated = activation(x)
                outputs.append(activated.unsqueeze(0))
            except Exception as e:
                print(f"Error in activation {name}: {e}")
                # 如果某個激活函數出錯，使用 ReLU
                activated = F.relu(x, inplace=False)
                outputs.append(activated.unsqueeze(0))
        
        # 堆疊所有激活函數的輸出 [num_activations, batch, ...]
        stacked = torch.cat(outputs, dim=0)
        
        # 應用權重
        num_activations = stacked.shape[0]
        batch_size = stacked.shape[1]
        
        # 將權重重塑為適當的形狀
        weights_expanded = weights.t().contiguous()  # [num_activations, batch]
        
        # 為了廣播，需要將權重擴展到與stacked相同的維度
        for _ in range(stacked.dim() - 2):  # 除了前兩個維度
            weights_expanded = weights_expanded.unsqueeze(-1)
        
        # 加權求和
        weighted_sum = (stacked * weights_expanded).sum(dim=0)
        
        return weighted_sum, weights
    
class DynamicActivationSelector(nn.Module):
    """
    動態激活函數選擇器
    根據輸入數據特徵動態選擇最適合的激活函數
    """
    def __init__(self, activation_configs: Dict[str, Dict], hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation_configs = activation_configs
        
        # 創建各種激活函數
        self.activations = nn.ModuleDict()
        for name, config in activation_configs.items():
            self.activations[name] = self._create_activation(config)
        
        # 分類器網絡 - 用於預測當前數據屬於哪種氣候模式
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, len(activation_configs)),
            nn.Softmax(dim=1)
        )
        
        # 初始化權重
        self.weights_history = []
    
    def _create_activation(self, config: Dict):
        """
        根據配置創建激活函數
        """
        if isinstance(config, str):
            if config == 'relu':
                return nn.ReLU(inplace=True)
            elif config == 'gelu':
                return nn.GELU()
            elif config == 'silu':
                return nn.SiLU(inplace=True)
            else:
                return nn.ReLU(inplace=True)
        elif isinstance(config, dict):
            activation_type = config.get('type', 'relu')
            if activation_type == 'relu':
                return nn.ReLU(inplace=True)
            elif activation_type == 'gelu':
                return nn.GELU()
            elif activation_type == 'silu':
                return nn.SiLU(inplace=True)
            elif activation_type == 'snake_learned':
                from GeoVMRNN_enhance import LearnedSnake
                return LearnedSnake(in_features=self.hidden_dim)
            elif activation_type == 'snake_fixed':
                from GeoVMRNN_enhance import FixedSnake
                return FixedSnake(a=config.get('params', {}).get('a', 1.0))
            elif activation_type == 'snake_adaptive':
                from GeoVMRNN_enhance import AdaptiveSnake
                return AdaptiveSnake(in_features=self.hidden_dim)
            else:
                return nn.ReLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x, features=None):
        """
        前向傳播，動態選擇激活函數
        
        Args:
            x: 輸入張量
            features: 用於分類的特徵，如果為None則使用x
        
        Returns:
            激活後的輸出和分類權重
        """
        # 使用特徵進行分類
        if features is None:
            # 如果沒有提供特徵，使用輸入的平均值作為特徵
            if len(x.shape) > 2:  # 對於卷積層輸入
                # 對空間維度進行平均池化
                pooled = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
                if pooled.size(1) > self.hidden_dim:
                    pooled = pooled[:, :self.hidden_dim]
                elif pooled.size(1) < self.hidden_dim:
                    # 如果特徵維度小於hidden_dim，進行填充
                    padding = torch.zeros(pooled.size(0), self.hidden_dim - pooled.size(1), device=x.device)
                    pooled = torch.cat([pooled, padding], dim=1)
            else:  # 對於全連接層輸入
                pooled = x.mean(dim=0, keepdim=True).expand(x.size(0), -1)
                if pooled.size(1) > self.hidden_dim:
                    pooled = pooled[:, :self.hidden_dim]
                elif pooled.size(1) < self.hidden_dim:
                    padding = torch.zeros(pooled.size(0), self.hidden_dim - pooled.size(1), device=x.device)
                    pooled = torch.cat([pooled, padding], dim=1)
        else:
            pooled = features
        
        # 預測分類權重
        weights = self.classifier(pooled)
        self.weights_history.append(weights.detach().mean(dim=0))
        
        # 應用每個激活函數並根據權重加權
        outputs = []
        for i, (name, activation) in enumerate(self.activations.items()):
            activated = activation(x)
            outputs.append(activated.unsqueeze(0))
        
        # 堆疊所有激活函數的輸出 [num_activations, batch, ...]
        stacked = torch.cat(outputs, dim=0)
        
        # 應用權重 [batch, num_activations, 1, 1, ...] * [num_activations, batch, ...]
        weight_shape = [1] * len(stacked.shape)
        weight_shape[0] = stacked.shape[0]  # 激活函數數量
        weight_shape[1] = stacked.shape[1]  # batch size
        
        # 轉置權重以匹配維度
        weights_expanded = weights.t().view(*weight_shape)
        
        # 加權求和
        weighted_sum = (stacked * weights_expanded).sum(dim=0)
        
        return weighted_sum, weights
    
    def get_weights_history(self):
        """
        獲取權重歷史
        """
        if not self.weights_history:
            return None
        return torch.stack(self.weights_history)


class DynamicActivationFusionLayer(nn.Module):
    """
    動態激活函數融合層
    用於替代GeoVMRNN中的fusion層，實現動態選擇激活函數
    """
    def __init__(self, in_channels, out_channels, activation_configs):
        super().__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.selector = DynamicActivationSelector(activation_configs, out_channels)
    
    def forward(self, x, climate_features=None):
        """
        前向傳播
        
        Args:
            x: 輸入特徵
            climate_features: 氣候特徵，用於分類
        
        Returns:
            激活後的輸出和分類權重
        """
        # x = self.norm(self.conv(x))
        return self.selector(x, climate_features)


def create_dynamic_activation_geo_vmrnn(mypara, activation_configs=None):
    """
    創建具有動態激活函數選擇能力的GeoVMRNN模型
    
    Args:
        mypara: 模型參數
        activation_configs: 激活函數配置字典，格式為 {name: config}
    
    Returns:
        具有動態激活函數選擇能力的GeoVMRNN模型
    """
    from GeoVMRNN_ablation import GeoVMRNN_Supervised, get_ablation_configs
    
    # 如果沒有提供配置，使用默認配置
    if activation_configs is None:
        activation_configs = get_ablation_configs()
    
    # 創建基本模型
    model = GeoVMRNN_Supervised(mypara)
    
    # 替換fusion層為動態激活函數融合層
    in_channels = 256 + 256  # 根據原始模型中fusion層的輸入通道數
    out_channels = 512      # 根據原始模型中fusion層的輸出通道數
    
    # 創建動態激活函數融合層
    dynamic_fusion = DynamicActivationFusionLayer(in_channels, out_channels, activation_configs)
    
    # 替換原始fusion層
    model.fusion_activation = dynamic_fusion
    
    return model


class DynamicActivationLoss(nn.Module):
    """
    動態激活函數損失
    包含主要任務損失和激活函數選擇的正則化項
    """
    def __init__(self, main_loss_fn, regularization_weight=0.01):
        super().__init__()
        self.main_loss_fn = main_loss_fn
        self.regularization_weight = regularization_weight
    
    def forward(self, pred, target, activation_weights=None, climate_labels=None):
        """
        計算損失
        
        Args:
            pred: 模型預測
            target: 目標值
            activation_weights: 激活函數選擇權重
            climate_labels: 氣候類型標籤 (0: 正常, 1: 聖嬰, 2: 反聖嬰)
        
        Returns:
            總損失
        """
        # 主要任務損失
        main_loss = self.main_loss_fn(pred, target)
        
        # 如果沒有提供激活函數權重或氣候標籤，只返回主要損失
        if activation_weights is None or climate_labels is None:
            return main_loss
        
        # 計算激活函數選擇的正則化損失
        # 對於聖嬰現象(1)，偏好第一個激活函數
        # 對於正常現象(0)，偏好第二個激活函數
        # 對於反聖嬰現象(2)，偏好第二個激活函數
        
        # 創建目標權重分佈
        batch_size = activation_weights.size(0)
        num_activations = activation_weights.size(1)
        target_weights = torch.zeros_like(activation_weights)
        
        # 根據氣候標籤設置目標權重
        for i in range(batch_size):
            if climate_labels[i] == 1:  # 聖嬰
                target_weights[i, 0] = 1.0  # 偏好第一個激活函數 (baseline)
            else:  # 正常或反聖嬰
                target_weights[i, 1] = 1.0  # 偏好第二個激活函數 (fusion_learned)
        
        # 計算KL散度損失
        kl_loss = F.kl_div(
            F.log_softmax(activation_weights, dim=1),
            target_weights,
            reduction='batchmean'
        )   
        
        # 總損失
        total_loss = main_loss + self.regularization_weight * kl_loss
        
        return total_loss


def detect_climate_phenomenon(sst_anomaly, nino_region):
    """
    檢測氣候現象類型 (聖嬰/正常/反聖嬰)
    
    Args:
        sst_anomaly: SST異常值，形狀為 [batch, time, lat, lon]
        nino_region: Nino區域 (lat_min, lat_max, lon_min, lon_max)
    
    Returns:
        氣候類型標籤 (0: 正常, 1: 聖嬰, 2: 反聖嬰)
    """
    # 提取Nino區域
    lat_min, lat_max, lon_min, lon_max = nino_region
    nino_sst = sst_anomaly[:, :, lat_min:lat_max, lon_min:lon_max].mean(dim=(2, 3))
    
    # 計算Nino指數 (最後時間步的平均值)
    nino_index = nino_sst[:, -1]  # [batch]
    
    # 根據Nino指數分類
    # 聖嬰: > 0.5°C
    # 反聖嬰: < -0.5°C
    # 正常: 其他
    climate_labels = torch.zeros_like(nino_index, dtype=torch.long)
    climate_labels[nino_index > 0.5] = 1  # 聖嬰
    climate_labels[nino_index < -0.5] = 2  # 反聖嬰
    
    return climate_labels
