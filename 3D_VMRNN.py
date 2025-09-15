import torch
import torch.nn as nn
import torch.nn.functional as F
from vmamba import VSSBlock, SS2D
from typing import Optional, Callable, Dict, Union, List
from functools import partial

class LearnedSnake(nn.Module):
    """學習性Snake激活函數"""
    def __init__(self, in_features=1, a=None):
        super().__init__()
        if a is not None:
            self.a = nn.Parameter(torch.tensor(a))
        else:
            self.a = nn.Parameter(torch.ones(1))
        self.a.requires_grad = True
        
    def forward(self, x):
        return x + torch.square(torch.sin(self.a * x)) / (torch.abs(self.a) + 1e-8)


class FixedSnake(nn.Module):
    """固定參數Snake激活函数"""
    def __init__(self, a=1.0):
        super().__init__()
        self.a = a
        
    def forward(self, x):
        return x + torch.square(torch.sin(self.a * x)) / self.a


class AdaptiveSnake(nn.Module):
    """自適應Snake激活函數"""
    def __init__(self, in_features):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.a.requires_grad = True
        
    def forward(self, x):
        return x + torch.square(torch.sin(self.a * x)) / (torch.abs(self.a) + 1e-8)


class ActivationFactory:
    """激活函數工廠類"""
    @staticmethod
    def get_activation(activation_config: Union[str, Dict], hidden_dim: int = None):
        """
        根據配置創建激活函數
        
        Args:
            activation_config: 字符串或字典配置
            hidden_dim: 隱藏維度（某些激活函數需要）
            
        Returns:
            nn.Module: 激活函數
        """
        if isinstance(activation_config, str):
            activation_type = activation_config
            params = {}
        else:
            activation_type = activation_config.get('type', 'relu')
            params = activation_config.get('params', {})
        
        if activation_type == 'relu':
            return nn.ReLU(inplace=params.get('inplace', True))
        elif activation_type == 'gelu':
            return nn.GELU()
        elif activation_type == 'silu':
            return nn.SiLU(inplace=params.get('inplace', True))
        elif activation_type == 'snake_learned':
            return LearnedSnake(in_features=params.get('in_features', hidden_dim))
        elif activation_type == 'snake_fixed':
            return FixedSnake(a=params.get('a', 1.0))
        elif activation_type == 'snake_adaptive':
            return AdaptiveSnake(in_features=params.get('in_features', hidden_dim))
        elif activation_type == 'dynamic':
            # 導入動態激活函數選擇器
            from DynamicActivationSelector import DynamicActivationSelector
            # 獲取激活函數配置
            configs = get_ablation_configs()
            # 創建動態激活函數選擇器
            return DynamicActivationSelector(configs, hidden_dim)
        else:
            return nn.ReLU(inplace=True)
        
class VSB(VSSBlock):
    def __init__(
        self,
        hidden_dim: int = 0,
        input_resolution: tuple = None,  # None，強制傳入
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs
    ):
        # 如果沒有傳入 input_resolution，使用默認值
        if input_resolution is None:
            input_resolution = (224, 224)
            
        super().__init__(
            hidden_dim=hidden_dim,
            input_resolution=input_resolution,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_resolution = input_resolution

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.ln_1(x)

        if hx is not None:
            hx = self.ln_1(hx)
            x = torch.cat((x, hx), dim=-1)
            x = self.linear(x)
        x = x.view(B, H, W, C) 

        x = self.drop_path(self.self_attention(x))
 
        x = x.view(B, H * W, C)
        x = shortcut + x

        return x


class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution, depth,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, d_state=16, **kwargs):
        super(VMRNNCell, self).__init__()

        self.VSBs = nn.ModuleList(
            VSB(hidden_dim=hidden_dim, 
                input_resolution=input_resolution,  # 確保傳入正確的分辨率
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                norm_layer=norm_layer, attn_drop_rate=attn_drop,
                d_state=d_state, **kwargs)
            for i in range(depth))

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)
        else:
            hx, cx = hidden_states
        
        outputs = []
        for index, layer in enumerate(self.VSBs):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)
            else:
                x = layer(outputs[-1], None)
                outputs.append(x)
                
        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t)
        cell = torch.tanh(o_t)
        Ct = Ft * (cx + cell)
        Ht = Ft * torch.tanh(Ct)

        return Ht, (Ht, Ct)

class GeoCNN(nn.Module):
    """地理空間特徵提取卷積模塊"""
    def __init__(self, in_channels, out_channels, activation_config, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = ActivationFactory.get_activation(activation_config, out_channels)
        # self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class SpatialAttention(nn.Module):
    """空間注意力機制"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class SupervisedPredictionHead(nn.Module):
    """支持監督式學習的預測頭"""
    def __init__(self, input_dim, cube_dim, activation_config, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.cube_dim = cube_dim
        self.num_layers = num_layers
        
        layers = []
        hidden_dim = input_dim // 2
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1))
            elif i == num_layers - 1:
                layers.append(nn.Conv2d(hidden_dim, cube_dim, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(hidden_dim))
                activation = ActivationFactory.get_activation(activation_config, hidden_dim)
                layers.append(activation)
        
        self.prediction_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.prediction_layers(x)


class GeoVMRNN_Enhance(nn.Module):
    """支持監督式學習的 GeoVMRNN 模型"""
    def __init__(self, mypara, activation_config=None):
        super().__init__()
        self.mypara = mypara
        self.device = mypara.device

        # 默認激活函數配置
        if activation_config is None:
            activation_config = 'relu'
        self.activation_config = activation_config

        # 使用與 Geoformer 相同的 patch 參數
        if hasattr(mypara, 'patch_size'):
            self.patch_size = mypara.patch_size
        else:
            self.patch_size = (4, 4)
            
        # 計算地理尺寸
        if hasattr(mypara, 'lat_range') and hasattr(mypara, 'lon_range'):
            # 根據地理範圍計算实际尺寸
            lat_span = mypara.lat_range[1] - mypara.lat_range[0]
            lon_span = mypara.lon_range[1] - mypara.lon_range[0]
            
            # 如果有分辨率信息，使用分辨率計算
            if hasattr(mypara, 'resolution'):
                self.img_height = int(lat_span / mypara.resolution)
                self.img_width = int(lon_span / mypara.resolution)
            else:
                # 否則直接使用度數作為像素數（可能需要調整）
                self.img_height = int(lat_span)
                self.img_width = int(lon_span)
                
        elif hasattr(mypara, 'H0') and hasattr(mypara, 'W0'):
            # 如果直接給出了patch后的尺寸
            self.H0 = mypara.H0
            self.W0 = mypara.W0
            self.img_height = self.H0 * self.patch_size[0]
            self.img_width = self.W0 * self.patch_size[1]
        else:
            # 默認值
            self.img_height = 224
            self.img_width = 224
            
        # 計算patch後的尺寸
        self.H0 = self.img_height // self.patch_size[0]
        self.W0 = self.img_width // self.patch_size[1]
        self.emb_spatial_size = self.H0 * self.W0
        
        # 地理輸入分辨率（patch後的尺寸）
        self.geo_input_resolution = (self.H0, self.W0)
            
        # 計算cube_dim（與Geoformer保持一致）
        if self.mypara.needtauxy:
            self.cube_dim = mypara.input_channal + 2
            self.input_channels = mypara.input_channal + 2
        else:
            self.cube_dim = mypara.input_channal
            self.input_channels = mypara.input_channal

        # 處理激活函數配置
        cnn_config, fusion_config, pred_config = self._parse_activation_config()

        # 編碼器：地理空間特徵提取
        self.geo_cnn = nn.Sequential(
            GeoCNN(self.input_channels, 64, activation_config=cnn_config),
            GeoCNN(64, 128, activation_config=cnn_config),
            GeoCNN(128, 256, activation_config=cnn_config),
            SpatialAttention(256)
        )
        
        # 嵌入維度
        self.embed_dim = 256
        
        # 編碼器：VMRNN Cell（使用地理分辨率）
        self.encoder_vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim,
            input_resolution=self.geo_input_resolution,  # 使用地理分辨率
            depth=2,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            d_state=16
        )
        
        # 解碼器：VMRNN Cell（使用地理分辨率）
        self.decoder_vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim,
            input_resolution=self.geo_input_resolution,  # 使用地理分辨率
            depth=2,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            d_state=16
        )
        
        # patch嵌入層
        self.patch_project = nn.Conv2d(256, self.embed_dim, 
                                     kernel_size=self.patch_size, 
                                     stride=self.patch_size)
        
        # 將patch序列轉換回圖像格式
        self.patch_to_img = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=256,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # 特徵融合層
        self.fusion_conv = nn.Conv2d(256 + 256, 512, kernel_size=1)
        self.fusion_norm = nn.BatchNorm2d(512)
        self.fusion_activation = ActivationFactory.get_activation(fusion_config, 512)
        
        # 記錄是否使用動態激活函數
        # self.use_dynamic_activation = isinstance(self.fusion_activation, nn.Module) and hasattr(self.fusion_activation, 'selector')

        # 修復：正確創建動態激活函數或普通激活函數
        if fusion_config == 'dynamic':
            try:
                # 創建簡單的動態激活函數選擇器
                configs = get_ablation_configs()
                # 只保留非動態配置來避免循環依賴
                filtered_configs = {k: v for k, v in configs.items() if k != 'dynamic_fusion'}
                
                # 使用簡化版本
                class SimpleDynamicActivation(nn.Module):
                    def __init__(self, hidden_dim):
                        super().__init__()
                        # 創建多個激活函數
                        self.activations = nn.ModuleList([
                            nn.ReLU(inplace=True),
                            LearnedSnake(in_features=1),
                            nn.GELU()
                        ])
                        
                        # 簡單的權重預測器
                        self.weight_predictor = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            nn.Linear(hidden_dim, len(self.activations)),
                            nn.Softmax(dim=1)
                        )
                        self.weights_history = []
                    
                    def forward(self, x, climate_features=None):
                        # 預測權重
                        weights = self.weight_predictor(x)  # [batch, num_activations]
                        self.weights_history.append(weights.detach().mean(dim=0))
                        
                        # 應用激活函數
                        outputs = []
                        for activation in self.activations:
                            try:
                                outputs.append(activation(x))
                            except:
                                outputs.append(F.relu(x, inplace=False))
                        
                        # 加權組合
                        stacked = torch.stack(outputs, dim=0)  # [num_act, batch, C, H, W]
                        weights_expanded = weights.t().view(-1, weights.size(0), 1, 1, 1)  # [num_act, batch, 1, 1, 1]
                        
                        result = (stacked * weights_expanded).sum(dim=0)
                        
                        return result, weights
                
                self.fusion_activation = SimpleDynamicActivation(512)
                self.use_dynamic_activation = True
                print("使用簡化動態激活函數")
                
            except Exception as e:
                print(f"Warning: 動態激活函數創建失敗，使用普通激活函數: {e}")
                self.fusion_activation = ActivationFactory.get_activation('relu', 512)
                self.use_dynamic_activation = False
        else:
            self.fusion_activation = ActivationFactory.get_activation(fusion_config, 512)
            self.use_dynamic_activation = False

        # 記錄是否使用動態激活函數
        print(f"使用動態激活函數: {self.use_dynamic_activation}")
        
        # 監督式預測頭（每次只預測一個時間步）
        self.prediction_head = SupervisedPredictionHead(
            input_dim=512,
            cube_dim=self.cube_dim,
            num_layers=3,
            activation_config=pred_config
        )
      
        # 初始化誤差預測
        self.residual_head = SupervisedPredictionHead(
            input_dim=512,  # 或與 fusion_conv 輸出一致
            cube_dim=self.cube_dim,
            num_layers=2,
            activation_config=pred_config
        )
        # self.error_correction_ann = ErrorCorrectionANN(input_dim=self.cube_dim)
        
        # 打印調試信息
        print(f"地理尺寸: {self.img_height} x {self.img_width}")
        print(f"Patch后尺寸: {self.H0} x {self.W0}")
        print(f"使用的input_resolution: {self.geo_input_resolution}")
        print(f"使用混合激活函數配置: {self.activation_config}")
        self._print_activation_summary()

    def _parse_activation_config(self):
        """解析激活函數配置"""
        if isinstance(self.activation_config, str):
            # 全部使用相同激活函數
            return (self.activation_config,) * 3
        elif isinstance(self.activation_config, dict):
            # 字典形式配置
            cnn_config = self.activation_config.get('cnn', 'relu')
            fusion_config = self.activation_config.get('fusion', 'relu')
            pred_config = self.activation_config.get('prediction', 'relu')
            return cnn_config, fusion_config, pred_config
        else:
            return ('relu',) * 3

    def _print_activation_summary(self):
        """打印激活函數使用摘要"""
        cnn_config, fusion_config, pred_config = self._parse_activation_config()
        
        print("激活函數配置摘要:")
        print(f"  CNN特徵提取層: {cnn_config}")
        print(f"  特徵融合層: {fusion_config}")
        print(f"  預測頭: {pred_config}")
        
    def encode(self, predictor):
        """編碼器：處理歷史數據"""
        batch_size, seq_len, C, H, W = predictor.shape
        # print(f"Encode - Input shape: {predictor.shape}")
        
        # 验证输入尺寸是否匹配
        assert H == self.img_height and W == self.img_width, \
            f"输入尺寸 {H}x{W} 不匹配预期的地理尺寸 {self.img_height}x{self.img_width}"
        
        # 初始化編碼器隱藏狀態
        encoder_hidden = None
        
        # 編碼所有歷史時間步
        for t in range(seq_len):
            # 地理空間特徵提取
            geo_feat = self.geo_cnn(predictor[:, t])
            # print(f"Encode - After CNN shape at step {t}: {geo_feat.shape}")
            
            # 轉換為patch序列
            patch_feat = self.patch_project(geo_feat)
            B, C_embed, H_patch, W_patch = patch_feat.shape
            # print(f"Encode - After patch projection shape at step {t}: {patch_feat.shape}")
            
            # 验证patch尺寸
            assert H_patch == self.H0 and W_patch == self.W0, \
                f"Patch尺寸 {H_patch}x{W_patch} 不匹配预期的 {self.H0}x{self.W0}"
            
            patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
            # print(f"Encode - After reshape and permute at step {t}: {patch_embed_feat.shape}")
            
            # 編碼器VMRNN處理
            encoded_feat, encoder_hidden = self.encoder_vmrnn_cell(patch_embed_feat, encoder_hidden)
            # print(f"Encode - After VMRNN at step {t}: {encoded_feat.shape}")
        
        # print(f"Encode - Final output shape: {encoded_feat.shape}")
        return encoded_feat, encoder_hidden
    
    def decode_step(self, current_input, encoder_output, decoder_hidden, climate_features=None):
        """解碼器：單步預測"""
        # 地理空間特徵提取
        geo_feat = self.geo_cnn(current_input)
        
        # 轉換為patch序列
        patch_feat = self.patch_project(geo_feat)
        B, C_embed, H_patch, W_patch = patch_feat.shape
        patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
        
        # 解碼器VMRNN處理
        decoded_feat, new_decoder_hidden = self.decoder_vmrnn_cell(patch_embed_feat, decoder_hidden)
        
        # 轉換回圖像格式
        decoded_img = decoded_feat.permute(0, 2, 1).view(B, self.embed_dim, H_patch, W_patch)
        vmrnn_feat = self.patch_to_img(decoded_img)
        
        # 特徵融合
        encoder_img = encoder_output.permute(0, 2, 1).view(B, self.embed_dim, H_patch, W_patch)
        encoder_feat = self.patch_to_img(encoder_img)
        
        fused_features = torch.cat([geo_feat, vmrnn_feat], dim=1)
        fused_features_conv = self.fusion_norm(self.fusion_conv(fused_features))
        
        # 修復：正確處理動態激活函數的返回值
        activation_weights = None
        
        if self.use_dynamic_activation and hasattr(self.fusion_activation, 'forward'):
            try:
                # 檢查是否需要氣候特徵
                if hasattr(self.fusion_activation, 'weight_predictor'):
                    # 使用簡化版動態激活函數
                    fused_features, activation_weights = self.fusion_activation(fused_features_conv)
                else:
                    # 如果有其他版本的動態激活函數
                    if climate_features is None:
                        # 如果沒有提供氣候特徵，從當前輸入中提取
                        local_climate_features = current_input[:, self.sstlevel if hasattr(self, 'sstlevel') else 0, 
                                            self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                                            self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1]].mean(dim=[1, 2])
                        local_climate_features = local_climate_features.view(B, -1)
                    else:
                        # 使用提供的氣候特徵
                        local_climate_features = climate_features
                    
                    # 確保特徵維度正確
                    if local_climate_features.size(1) < 512:
                        padding = torch.zeros(B, 512 - local_climate_features.size(1), device=local_climate_features.device)
                        local_climate_features = torch.cat([local_climate_features, padding], dim=1)
                    elif local_climate_features.size(1) > 512:
                        local_climate_features = local_climate_features[:, :512]
                    
                    fused_features, activation_weights = self.fusion_activation(fused_features_conv, local_climate_features)
                    
            except Exception as e:
                print(f"Warning: 動態激活函數執行失敗，使用普通激活函數: {e}")
                # 降級到普通激活函數
                if hasattr(self.fusion_activation, 'activations') and len(self.fusion_activation.activations) > 0:
                    # 使用第一個激活函數作為fallback
                    fused_features = self.fusion_activation.activations[0](fused_features_conv)
                else:
                    fused_features = F.relu(fused_features_conv, inplace=False)
        else:
            # 普通激活函數
            if callable(self.fusion_activation):
                fused_features = self.fusion_activation(fused_features_conv)
            else:
                fused_features = F.relu(fused_features_conv, inplace=False)
        
        # 確保 fused_features 是 tensor 而不是 tuple
        if isinstance(fused_features, (list, tuple)):
            fused_features = fused_features[0]  # 取第一個元素
        
        # 預測下一時間步
        prediction = self.prediction_head(fused_features)
        # 誤差補償
        residual = self.residual_head(fused_features)
        # 加總成最終預測
        final_prediction = prediction + residual
        # 恢復到原始尺寸
        H, W = current_input.shape[-2:]
        final_prediction = final_prediction.view(B, self.cube_dim, H, W)
        next_step = final_prediction
        
        return next_step, prediction, residual, new_decoder_hidden, activation_weights
    
    def correct_error(self, pred_t):
        return pred_t + self.error_correction_ann(pred_t)

    def forward(self, predictor, predictand=None, train=True, sv_ratio=0, climate_features=None):
        """
        Args:
            predictor: (batch, input_length, C, H, W)
            predictand: (batch, output_length, C, H, W) - 訓練時使用
            train: 是否為訓練模式
            sv_ratio: 監督比例 (與Geoformer一致)
            climate_features: 用於動態激活函數的氣候特徵 (可選)
        Returns:
            outvar_pred: (batch, output_length, C, H, W)
        """
        batch_size, seq_len, C, H, W = predictor.shape
        # print(f"Forward - Input predictor shape: {predictor.shape}")
        if predictand is not None:
            # print(f"Forward - Input predictand shape: {predictand.shape}")
            # print(f"Forward - Expected output length: {self.mypara.output_length}")
            assert predictand.size(1) == self.mypara.output_length, \
                f"Predictand sequence length {predictand.size(1)} does not match expected length {self.mypara.output_length}"
        
        # 1. 編碼階段
        encoder_output, encoder_hidden = self.encode(predictor)
        # print(f"Forward - Encoder output shape: {encoder_output.shape}")
        
        # 2. 解碼階段
        if train:
            # 訓練模式：使用teacher forcing
            assert predictand is not None, "在訓練模式下必須提供 predictand"
            
            # 監督式訓練
            decoder_hidden = encoder_hidden
            outputs = []
            mse = nn.MSELoss()
            residual_losses = []
            
            # Use the previous step's ground truth as input for the next step
            current_input = predictor[:, -1]  # Start with last input step
            
            for t in range(self.mypara.output_length):
                next_step, coarse_pred, residual_pred, decoder_hidden, activation_weights = self.decode_step(
                    current_input, encoder_output, decoder_hidden, climate_features
                )
                outputs.append(next_step)

                # residual supervision: (ground truth - coarse prediction) as target
                residual_target = predictand[:, t] - coarse_pred
                loss_r = mse(residual_pred, residual_target)
                residual_losses.append(loss_r)
                
                # Use ground truth as next input if available (teacher forcing)
                if t < self.mypara.output_length - 1:
                    current_input = predictand[:, t]
            
            outvar_pred = torch.stack(outputs, dim=1)
            # print(f"Forward - Output prediction shape: {outvar_pred.shape}")
            assert outvar_pred.size(1) == self.mypara.output_length, \
                f"Output prediction sequence length {outvar_pred.size(1)} does not match expected length {self.mypara.output_length}"
            
            # 應用監督比例（與Geoformer相同）
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio * torch.ones(batch_size, self.mypara.output_length - 1, 1, 1, 1)
                ).to(self.device)
                
                # 混合真實值和預測值
                mixed_predictand = (
                    supervise_mask * predictand[:, :-1] + 
                    (1 - supervise_mask) * outvar_pred[:, :-1]
                )
                
                # 重新進行預測
                decoder_hidden = encoder_hidden
                outputs = []
                mse = nn.MSELoss()
                residual_losses = []
                current_input = predictor[:, -1]  # Start with last input step
                
                for t in range(self.mypara.output_length):
                    next_step, coarse_pred, residual_pred, decoder_hidden, activation_weights = self.decode_step(
                        current_input, encoder_output, decoder_hidden, climate_features
                    )
                    outputs.append(next_step)

                    # residual supervision: (ground truth - coarse prediction) as target
                    residual_target = predictand[:, t] - coarse_pred
                    loss_r = mse(residual_pred, residual_target)
                    residual_losses.append(loss_r)
                    
                    # Use mixed ground truth/prediction as next input
                    if t < self.mypara.output_length - 1:
                        current_input = mixed_predictand[:, t]
                
                outvar_pred = torch.stack(outputs, dim=1)
        
        else:
            # 推理模式：自回歸生成
            decoder_hidden = encoder_hidden
            outputs = []
            current_input = predictor[:, -1]  # Start with last input step
            
            # 在推理模式下，不計算殘差損失或返回零損失
            residual_losses = []
            
            for t in range(self.mypara.output_length):
                next_step, coarse_pred, residual_pred, decoder_hidden, activation_weights = self.decode_step(
                    current_input, encoder_output, decoder_hidden, climate_features
                )
                outputs.append(next_step)
                current_input = next_step  # 使用預測結果作為下一步輸入

                # 推理模式下不計算殘差損失，或者可以設為零
                # 如果需要在推理時也計算殘差損失，需要提供 predictand
                if predictand is not None:
                    mse = nn.MSELoss()
                    residual_target = predictand[:, t] - coarse_pred
                    loss_r = mse(residual_pred, residual_target)
                    residual_losses.append(loss_r)
                else:
                    # 推理模式下沒有真實值，設為零損失
                    residual_losses.append(torch.tensor(0.0, device=self.device))
            
            outvar_pred = torch.stack(outputs, dim=1)

            # print(f"Forward - Output prediction shape (inference): {outvar_pred.shape}")
            assert outvar_pred.size(1) == self.mypara.output_length, \
                f"Inference output sequence length {outvar_pred.size(1)} does not match expected length {self.mypara.output_length}"
        
        # 返回平均殘差損失
        if residual_losses:
            mean_residual_loss = torch.stack(residual_losses).mean()
        else:
            mean_residual_loss = torch.tensor(0.0, device=self.device)
        
        return outvar_pred, mean_residual_loss
    
    def predict(self, predictor):
        """推理模式"""
        return self.forward(predictor, train=False)

# 添加維度檢查
def check_dimensions(self, x, stage):
    print(f"{stage}: {x.shape}")
    return x

# 工廠函數
def create_supervised_geo_vmrnn(mypara, activation_config=None):
    """創建支持監督式學習的 GeoVMRNN 模型"""
    return GeoVMRNN_Enhance(mypara, activation_config=activation_config)

def get_ablation_configs():
    """獲取示例混合激活函數配置"""
    
    # 配置1
    config_baseline = {
        'cnn': 'relu',
        'fusion': 'relu',
        'prediction': 'relu'
    }
    
    # 配置2
    fusion_learned = {
        'cnn': 'relu',
        'fusion': 'snake_learned',
        'prediction': 'relu'
    }
    
    # 配置3 - 動態選擇
    dynamic_fusion = {
        'cnn': 'relu',
        'fusion': 'dynamic',
        'prediction': 'relu'
    }
    
    return {
        'baseline': config_baseline,
        'fusion_learned': fusion_learned,
        'dynamic_fusion': dynamic_fusion
    }
