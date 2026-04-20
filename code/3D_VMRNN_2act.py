import torch
import torch.nn as nn
import torch.nn.functional as F
from vmamba import VSSBlock, SS2D
from typing import Optional, Callable, Dict, Union, List
from functools import partial

# =============================================================================
# 基礎激活函數
# =============================================================================

class LearnedSnake(nn.Module):
    """學習性Snake激活函數"""
    def __init__(self, in_features=1, a=None):
        super().__init__()
        if a is not None:
            self.a = nn.Parameter(torch.tensor(float(a)))
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


class ImprovedLearnedSnake(nn.Module):
    """
    方案 D：改進的學習型 Snake 激活函數
    - 添加可學習的縮放因子與偏移
    - 限制參數範圍避免數值不穩定
    """
    def __init__(self, init_a=1.0):
        super().__init__()
        self.a     = nn.Parameter(torch.tensor(float(init_a)))
        self.scale = nn.Parameter(torch.ones(1))
        self.bias  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        a = torch.clamp(self.a, min=0.1, max=10.0)
        snake_term = torch.sin(a * x) ** 2 / (a + 1e-8)
        out = x + snake_term
        return self.scale * out + self.bias

    def get_params_info(self):
        return {
            'a':     self.a.item(),
            'scale': self.scale.item(),
            'bias':  self.bias.item(),
        }


class DiversityRegularizer(nn.Module):
    """
    方案 B：多樣性正則化器
    用 KL 散度將 batch 平均權重推向目標分佈（預設均勻分佈）
    """
    def __init__(self, num_activations=2, target_distribution=None):
        super().__init__()
        if target_distribution is None:
            uniform = torch.ones(num_activations) / num_activations
            self.register_buffer('target_distribution', uniform)
        else:
            if not isinstance(target_distribution, torch.Tensor):
                target_distribution = torch.tensor(target_distribution, dtype=torch.float32)
            self.register_buffer('target_distribution', target_distribution)

    def forward(self, weights):
        eps = 1e-8
        batch_mean = weights.mean(dim=0)
        target = self.target_distribution.to(weights.device)
        kl_loss = F.kl_div(
            torch.log(batch_mean + eps),
            target,
            reduction='batchmean'
        )
        return kl_loss


# =============================================================================
# ActivationFactory
# =============================================================================

class ActivationFactory:
    """激活函數工廠類"""
    @staticmethod
    def get_activation(activation_config: Union[str, Dict], hidden_dim: int = None):
        if isinstance(activation_config, str):
            activation_type = activation_config
            params = {}
        else:
            activation_type = activation_config.get('type', 'relu')
            params = activation_config.get('params', {})
        
        if activation_type == 'relu':
            return nn.ReLU(inplace=False)
        elif activation_type == 'gelu':
            return nn.GELU()
        elif activation_type == 'silu':
            return nn.SiLU(inplace=False)
        elif activation_type == 'snake_learned':
            return LearnedSnake(in_features=params.get('in_features', hidden_dim))
        elif activation_type == 'snake_fixed':
            return FixedSnake(a=params.get('a', 1.0))
        elif activation_type == 'snake_adaptive':
            return AdaptiveSnake(in_features=params.get('in_features', hidden_dim))
        elif activation_type in ('dynamic', 'physical_gating'):
            return None
        else:
            return nn.ReLU(inplace=False)

    @staticmethod
    def is_physical_gating(activation_config) -> bool:
        if isinstance(activation_config, str):
            return activation_config in ('dynamic', 'physical_gating')
        if isinstance(activation_config, dict):
            return activation_config.get('type', '') in ('dynamic', 'physical_gating')
        return False


# =============================================================================
# PhysicalLatentGating — 2激活函數版（ReLU + LearnedSnake）
# ★★★ 與原版的差異：
#   1. activations 只有 2 個：ReLU (index 0) + LearnedSnake (index 1)
#   2. ENSO_PHASE_TO_ACTIVATION：La Niña/Neutral → Snake(1)，El Niño → ReLU(0)
#   3. num_activations 預設為 2
#   4. diversity_target 改為 [0.5, 0.5]
#   5. latent_shortcut 和 gate 輸出維度均為 2
# =============================================================================

# ★ 2激活函數版的相位映射（ReLU=0, LearnedSnake=1）
# 依據消融實驗：ReLU 在 El Niño 誤差最小，Snake 在 La Niña/Neutral 誤差最小
ENSO_PHASE_TO_ACTIVATION = {
    0: 1,   # La Niña → LearnedSnake (index 1)
    1: 1,   # Neutral → LearnedSnake (index 1)
    2: 0,   # El Niño → ReLU         (index 0)
}


class PhysicalLatentGating(nn.Module):
    """
    物理感知門控激活函數 — 2激活函數版（ReLU + LearnedSnake）

    ★ 與原版（3激活函數）的差異：
      - activations[0] = ReLU          → El Niño 目標
      - activations[1] = LearnedSnake  → La Niña / Neutral 目標
      - gate logit 為 2 維，Softmax 後自由度為 1D（最簡潔）
      - diversity_target = [0.5, 0.5]（均勻二元分佈）

    這組設定與消融實驗（Table 5）一致：
      - Fixed ReLU 在 El Niño 誤差最小
      - Fixed Learned Snake 在 La Niña/Neutral 誤差最小
    """

    def __init__(
        self,
        hidden_dim,
        # ★ 改為 2
        num_activations   = 2,
        use_hard_selection= False,
        # 方案 1：溫度調度
        init_temperature  = 5.0,
        min_temperature   = 1.0,
        temperature_decay = 0.999,
        # 方案 4：課程學習
        warmup_steps      = 10000,
        noise_scale       = 1.0,
        # 方案 A：強制權重平衡
        enforce_balance   = False,
        min_weight        = 0.05,
        balance_steps     = 5000,
        # 方案 B：多樣性正則化
        diversity_target  = None,
        # 方案 F：ENSO 相位監督
        phase_gate_loss_weight = 1.0,
    ):
        super().__init__()

        # ── ★ 2激活函數：ReLU + LearnedSnake（不分高低頻）─────────────── #
        # index 0：ReLU         → El Niño 目標
        # index 1：LearnedSnake → La Niña / Neutral 目標（a 可學習）
        self.activations = nn.ModuleList([
            nn.ReLU(inplace=False),          # [0] El Niño
            LearnedSnake(in_features=1),     # [1] La Niña / Neutral（a 從 1.0 開始學）
        ])

        self.input_channels = 512

        # ── 空間特徵壓縮器 ────────────────────────────────────────────── #
        self.spatial_compressor = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=1),
            nn.ReLU(inplace=False),
        )

        self.spatial_feature_extractor = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.AdaptiveAvgPool2d((2, 2)),
        ])
        spatial_feat_dim = 64 + 64 + 64 * 4   # = 384

        # ── 物理統計量提取器 ─────────────────────────────────────────── #
        self.stat_extractor = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 64, kernel_size=1),
        )
        stat_feat_dim = 64

        # ── 方案 3：深度特徵提取網路 ─────────────────────────────────── #
        total_feat_dim = hidden_dim + spatial_feat_dim + stat_feat_dim  # 704

        self.enhanced_feature_extractor = nn.Sequential(
            nn.Linear(total_feat_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
        )

        # ── ★ latent 捷徑：輸出 2 維 ─────────────────────────────────── #
        self.latent_shortcut = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=False),
            nn.Linear(64, num_activations),   # ★ 輸出 2 維
        )

        # ── ★ 門控決策網路：輸出 2 維 ────────────────────────────────── #
        self.gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(inplace=False),

            nn.Linear(64, num_activations),   # ★ 輸出 2 維
        )

        self.gate_blend = nn.Parameter(torch.tensor(0.5))

        # ── 方案 1：溫度調度 ─────────────────────────────────────────── #
        self.register_buffer('temperature', torch.ones(1) * init_temperature)
        self.temperature.requires_grad = False
        self.min_temperature   = min_temperature
        self.temperature_decay = temperature_decay
        self.current_temperature = float(init_temperature)

        # ── 方案 4：課程學習 ─────────────────────────────────────────── #
        self.warmup_steps   = warmup_steps
        self.noise_scale    = noise_scale
        self.training_step  = 0

        # ── 方案 A：強制權重平衡 ─────────────────────────────────────── #
        self.enforce_balance = enforce_balance
        self.min_weight      = min_weight
        self.balance_steps   = balance_steps

        # ── 方案 B：多樣性正則化 ─────────────────────────────────────── #
        self.diversity_regularizer = DiversityRegularizer(
            num_activations    = num_activations,
            target_distribution= diversity_target,
        )

        # ── 方案 F：ENSO 相位監督 ────────────────────────────────────── #
        self.phase_gate_loss_weight = phase_gate_loss_weight
        self._last_logits      = None
        self._last_logits_grad = None

        # 相位→激活函數索引映射（2激活函數版）
        phase_to_act = torch.tensor(
            [ENSO_PHASE_TO_ACTIVATION[i] for i in range(3)],
            dtype=torch.long
        )  # [3]
        self.register_buffer('phase_to_activation_idx', phase_to_act)

        print(f"  [2激活函數版] ENSO 相位→激活函數映射:")
        print(f"           La Niña  (0) → index {ENSO_PHASE_TO_ACTIVATION[0]} (LearnedSnake)")
        print(f"           Neutral  (1) → index {ENSO_PHASE_TO_ACTIVATION[1]} (LearnedSnake)")
        print(f"           El Niño  (2) → index {ENSO_PHASE_TO_ACTIVATION[2]} (ReLU)")
        print(f"           phase_gate_loss_weight = {phase_gate_loss_weight}")

        self.use_hard_selection = use_hard_selection
        self.num_activations    = num_activations
        self.weights_history    = []

        self._print_init_summary()

    def _print_init_summary(self):
        line = "=" * 80
        print(line)
        print("PhysicalLatentGating 初始化完成（★ 2激活函數版：ReLU + LearnedSnake）")
        print(f"  激活函數（2個）:")
        print(f"           [0] ReLU         → El Niño 目標")
        print(f"           [1] LearnedSnake → La Niña / Neutral 目標（a 可學習）")
        print(f"  [方案 1] 溫度調度:")
        print(f"           初始={self.current_temperature}, "
              f"最低={self.min_temperature}, "
              f"衰減={self.temperature_decay}")
        print(f"  [方案 3] 深度特徵提取: total_feat → 1024 → 512 → 256 → 2")
        print(f"  [方案 4] 課程學習 warmup: {self.warmup_steps} 步, "
              f"噪聲={self.noise_scale}")
        print(f"  [方案 A] 強制平衡: 啟用={self.enforce_balance}, "
              f"min_weight={self.min_weight}({self.min_weight*100:.0f}%), "
              f"持續={self.balance_steps} 步")
        print(f"  [方案 B] 多樣性正則化目標分佈: "
              f"{self.diversity_regularizer.target_distribution.tolist()}")
        print(f"  [方案 F] ENSO 相位監督: 啟用，weight={self.phase_gate_loss_weight}")
        print(f"  選擇模式: {'硬選擇 (Gumbel-Softmax)' if self.use_hard_selection else '軟選擇 (加權平均)'}")
        print(line)

    def update_temperature(self):
        with torch.no_grad():
            new_temp = max(
                self.current_temperature * self.temperature_decay,
                self.min_temperature
            )
            self.current_temperature = new_temp
            self.temperature.fill_(new_temp)

    def _add_exploration_noise(self, logits):
        if self.training and self.training_step < self.warmup_steps:
            progress      = self.training_step / self.warmup_steps
            current_scale = self.noise_scale * (1.0 - progress)
            noise = torch.randn_like(logits) * current_scale
            return logits + noise
        return logits

    def _apply_weight_constraints(self, weights):
        if not self.enforce_balance:
            return weights
        if self.training_step < self.balance_steps:
            progress = self.training_step / self.balance_steps
            current_min = self.min_weight * (1.0 - 0.7 * progress)
        else:
            current_min = self.min_weight * 0.2
        constrained = torch.clamp(weights, min=current_min)
        constrained = constrained / constrained.sum(dim=1, keepdim=True)
        return constrained

    def _extract_spatial_features(self, x):
        x_compressed = self.spatial_compressor(x)
        global_avg = self.spatial_feature_extractor[0](x_compressed).flatten(1)
        global_max = self.spatial_feature_extractor[1](x_compressed).flatten(1)
        regional   = self.spatial_feature_extractor[2](x_compressed).flatten(1)
        return torch.cat([global_avg, global_max, regional], dim=1)

    def _extract_statistical_features(self, x):
        stat_map = self.stat_extractor(x)
        return F.adaptive_avg_pool2d(stat_map, 1).flatten(1)

    def compute_diversity_loss(self, weights):
        return self.diversity_regularizer(weights)

    def compute_phase_gate_loss(self, phase_labels):
        """
        2激活函數版的相位監督損失
        La Niña(0) → LearnedSnake(1)
        Neutral(1)  → LearnedSnake(1)
        El Niño(2)  → ReLU(0)
        """
        if not hasattr(self, '_last_logits_grad') or self._last_logits_grad is None:
            return torch.tensor(0.0, device=phase_labels.device)

        logits_raw     = self._last_logits_grad
        target_device  = logits_raw.device
        phase_map      = self.phase_to_activation_idx.to(target_device)
        target_act_idx = phase_map[phase_labels.to(target_device)]

        phase_gate_loss = F.cross_entropy(logits_raw, target_act_idx)
        return phase_gate_loss

    def compute_decorrelation_loss(self):
        """
        2激活函數版：logit 只有 2 維，Softmax 天然使兩個 logit 強負相關，
        decorr_loss 在此版本意義不大，但保留介面以免訓練器報錯。
        直接回傳 0。
        """
        return torch.tensor(0.0)

    def get_gate_logits_for_probing(self):
        if self._last_logits is None:
            return None
        if self._last_logits.dim() == 3:
            return self._last_logits.mean(dim=0)
        return self._last_logits

    def forward(self, x, latent_vector):
        # ── 1. 特徵提取 ─────────────────────────────────────────────────── #
        spatial_feat = self._extract_spatial_features(x)
        stat_feat    = self._extract_statistical_features(x)
        combined = torch.cat([latent_vector, spatial_feat, stat_feat], dim=1)

        # ── 2. 深度特徵提取 ──────────────────────────────────────────────── #
        enhanced = self.enhanced_feature_extractor(combined)

        # ── 3. 門控決策（雙路徑）────────────────────────────────────────── #
        logits_deep  = self.gate(enhanced)               # [B, 2]
        logits_short = self.latent_shortcut(latent_vector)  # [B, 2]

        alpha  = torch.sigmoid(self.gate_blend)
        logits = alpha * logits_deep + (1.0 - alpha) * logits_short

        # ── 4. 儲存 logits ───────────────────────────────────────────────── #
        self._last_logits_grad = logits

        detached = logits.detach()
        if self._last_logits is not None and self._last_logits.size(1) != detached.size(0):
            self._last_logits = None
        if self._last_logits is None:
            self._last_logits = detached.unsqueeze(0)
        else:
            self._last_logits = torch.cat(
                [self._last_logits, detached.unsqueeze(0)], dim=0
            )

        # ── 5. 探索噪聲 ─────────────────────────────────────────────────── #
        logits = self._add_exploration_noise(logits)

        # ── 6. 溫度縮放 Softmax ─────────────────────────────────────────── #
        weights = F.softmax(logits / self.temperature, dim=-1)

        # ── 7. 強制最小比例 ─────────────────────────────────────────────── #
        weights = self._apply_weight_constraints(weights)

        # ── 8. 儲存歷史 ─────────────────────────────────────────────────── #
        self.weights_history.append(weights.detach().cpu())

        # ── 9. 計算各激活函數輸出 ───────────────────────────────────────── #
        outputs = []
        for act in self.activations:
            try:
                outputs.append(act(x))
            except Exception as e:
                print(f"[Warning] 激活函數執行失敗，fallback ReLU: {e}")
                outputs.append(F.relu(x, inplace=False))

        stacked = torch.stack(outputs, dim=0)  # [2, B, C, H, W]

        # ── 10. 軟選擇 / 硬選擇 ─────────────────────────────────────────── #
        if self.use_hard_selection:
            if self.training:
                weights_hard = F.gumbel_softmax(
                    logits / self.temperature,
                    tau=1.0, hard=True, dim=-1
                )
            else:
                idx = torch.argmax(weights, dim=-1)
                weights_hard = F.one_hot(
                    idx, num_classes=self.num_activations
                ).float()
            w_exp = weights_hard.t().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            w_exp = weights.t().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        result = (stacked * w_exp).sum(dim=0)

        # ── 11. 更新計數器 ───────────────────────────────────────────────── #
        if self.training:
            self.training_step += 1
            if self.training_step % 200 == 0:
                self._print_step_info(weights)

        return result, weights

    def _print_step_info(self, weights):
        avg_w = weights.mean(dim=0)
        print(f"\n[PhysicalLatentGating 2act] step={self.training_step}")
        print(f"  溫度: {self.current_temperature:.4f}")
        print(f"  avg weight: ReLU={avg_w[0]:.4f}  LearnedSnake={avg_w[1]:.4f}")
        if hasattr(self.activations[1], 'a'):
            print(f"  Snake.a = {self.activations[1].a.item():.4f}")

    def get_training_stats(self):
        stats = {
            'training_step'      : self.training_step,
            'current_temperature': self.current_temperature,
            'warmup_progress'    : min(1.0, self.training_step / max(1, self.warmup_steps)),
            'balance_progress'   : min(1.0, self.training_step / max(1, self.balance_steps)),
            'is_in_warmup'       : self.training_step < self.warmup_steps,
            'is_in_balance'      : self.training_step < self.balance_steps,
        }
        if self.weights_history:
            recent = torch.cat(self.weights_history[-100:], dim=0)
            stats['recent_mean_weights'] = recent.mean(dim=0).numpy()
            stats['recent_std_weights']  = recent.std(dim=0).numpy()
        if hasattr(self.activations[1], 'a'):
            stats['snake_a'] = self.activations[1].a.item()
        return stats

    def reset_training_state(self):
        self.training_step = 0
        init_temp = 10.0
        self.current_temperature = init_temp
        self.temperature.fill_(init_temp)
        self.weights_history = []
        self._last_logits = None
        print("[PhysicalLatentGating 2act] 訓練狀態已重置")


# =============================================================================
# 以下所有類別與原版完全相同，僅 build_physical_latent_gating 有改動
# =============================================================================

class NinoPhaseProbe(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3),
        )
        print(f"[NinoPhaseProbe] 分類探針已初始化 (embed_dim={embed_dim}, hidden={hidden_dim})")
    
    def forward(self, latent_vector):
        return self.classifier(latent_vector)


def get_enso_phase_label_tensor(nino_values):
    labels = torch.zeros_like(nino_values, dtype=torch.long)
    labels[nino_values < -0.5] = 0
    labels[nino_values > 0.5] = 2
    labels[(nino_values >= -0.5) & (nino_values <= 0.5)] = 1
    return labels


class NinoProbeHead(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=64):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent_vector):
        return self.probe(latent_vector).squeeze(-1)


class VSB(VSSBlock):
    def __init__(
        self,
        hidden_dim: int = 0,
        input_resolution: tuple = None,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs
    ):
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
        assert L == H * W
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
                input_resolution=input_resolution,
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
    def __init__(self, in_channels, out_channels, activation_config, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        act = ActivationFactory.get_activation(activation_config, out_channels)
        self.activation = act if act is not None else nn.ReLU(inplace=False)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class SupervisedPredictionHead(nn.Module):
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
                act = ActivationFactory.get_activation(activation_config, hidden_dim)
                layers.append(act if act is not None else nn.ReLU(inplace=False))
        self.prediction_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.prediction_layers(x)


# =============================================================================
# GeoVMRNN_Enhance — 主模型（與原版相同，不需改動）
# =============================================================================

class GeoVMRNN_Enhance(nn.Module):
    def __init__(self, mypara, activation_config=None):
        super().__init__()
        self.mypara = mypara
        self.device = mypara.device

        if activation_config is None:
            activation_config = 'relu'
        self.activation_config = activation_config

        self.embed_dim = 256

        self.nino_phase_probe = NinoPhaseProbe(embed_dim=self.embed_dim, hidden_dim=64)
        self.nino_probe = NinoProbeHead(embed_dim=self.embed_dim, hidden_dim=64)
        self.nino_probe_enabled = True

        if hasattr(mypara, 'patch_size'):
            self.patch_size = mypara.patch_size
        else:
            self.patch_size = (4, 4)

        if hasattr(mypara, 'lat_range') and hasattr(mypara, 'lon_range'):
            lat_span = mypara.lat_range[1] - mypara.lat_range[0]
            lon_span = mypara.lon_range[1] - mypara.lon_range[0]
            if hasattr(mypara, 'resolution'):
                self.img_height = int(lat_span / mypara.resolution)
                self.img_width  = int(lon_span / mypara.resolution)
            else:
                self.img_height = int(lat_span)
                self.img_width  = int(lon_span)
        elif hasattr(mypara, 'H0') and hasattr(mypara, 'W0'):
            self.H0 = mypara.H0
            self.W0 = mypara.W0
            self.img_height = self.H0 * self.patch_size[0]
            self.img_width  = self.W0 * self.patch_size[1]
        else:
            self.img_height = 224
            self.img_width  = 224

        self.H0 = self.img_height // self.patch_size[0]
        self.W0 = self.img_width  // self.patch_size[1]
        self.emb_spatial_size = self.H0 * self.W0
        self.geo_input_resolution = (self.H0, self.W0)

        if self.mypara.needtauxy:
            self.cube_dim = mypara.input_channal + 2
            self.input_channels = mypara.input_channal + 2
        else:
            self.cube_dim = mypara.input_channal
            self.input_channels = mypara.input_channal

        cnn_config, fusion_config, pred_config = self._parse_activation_config()

        self.geo_cnn = nn.Sequential(
            GeoCNN(self.input_channels, 64,  activation_config=cnn_config),
            GeoCNN(64,  128, activation_config=cnn_config),
            GeoCNN(128, 256, activation_config=cnn_config),
            SpatialAttention(256)
        )

        self.encoder_vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim, input_resolution=self.geo_input_resolution,
            depth=2, drop=0.0, attn_drop=0.0, drop_path=0.0,
            norm_layer=nn.LayerNorm, d_state=16)
        self.decoder_vmrnn_cell = VMRNNCell(
            hidden_dim=self.embed_dim, input_resolution=self.geo_input_resolution,
            depth=2, drop=0.0, attn_drop=0.0, drop_path=0.0,
            norm_layer=nn.LayerNorm, d_state=16)

        self.patch_project = nn.Conv2d(
            256, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_to_img = nn.ConvTranspose2d(
            in_channels=self.embed_dim, out_channels=256,
            kernel_size=self.patch_size, stride=self.patch_size)

        self.fusion_conv = nn.Conv2d(256 + 256, 512, kernel_size=1)
        self.fusion_norm = nn.BatchNorm2d(512)

        if ActivationFactory.is_physical_gating(fusion_config):
            self.fusion_activation = build_physical_latent_gating(
                hidden_dim=self.embed_dim, use_hard_selection=False)
            self.use_dynamic_activation = True
            print(f"[fusion] 使用 PhysicalLatentGating（★ 2激活函數版），latent_dim={self.embed_dim}")
        else:
            act = ActivationFactory.get_activation(fusion_config, 512)
            self.fusion_activation = act if act is not None else nn.ReLU(inplace=False)
            self.use_dynamic_activation = False

        self.nino_injector = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 256))

        self.prediction_head = SupervisedPredictionHead(
            input_dim=512, cube_dim=self.cube_dim, num_layers=3,
            activation_config=pred_config)
        self.residual_head = SupervisedPredictionHead(
            input_dim=512, cube_dim=self.cube_dim, num_layers=2,
            activation_config=pred_config)

        self.collect_features = False
        self.feature_buffer = {
            'L1_geo_cnn'      : [],
            'L2_patch_project': [],
            'L3_latent_probe' : [],
            'L4_fusion_conv'  : [],
            'L5_gate_logits'  : [],
            'L6_gate_weights' : [],
            'L7_after_gate'   : [],
        }

        self._verify_all_on_device()

    def _verify_all_on_device(self):
        print(f"\n{'='*60}")
        print("模型設備配置檢查:")
        print(f"  目標設備: {self.device}")
        for name, module in [
            ('geo_cnn', self.geo_cnn),
            ('encoder_vmrnn_cell', self.encoder_vmrnn_cell),
            ('decoder_vmrnn_cell', self.decoder_vmrnn_cell),
            ('fusion_conv', self.fusion_conv),
            ('fusion_norm', self.fusion_norm),
        ]:
            try:
                first_param = next(module.parameters())
                print(f"  {name}: {first_param.device}")
            except StopIteration:
                print(f"  {name}: 無參數")
        print(f"{'='*60}\n")

    def _parse_activation_config(self):
        if isinstance(self.activation_config, str):
            return (self.activation_config,) * 3
        elif isinstance(self.activation_config, dict):
            return (
                self.activation_config.get('cnn', 'relu'),
                self.activation_config.get('fusion', 'relu'),
                self.activation_config.get('prediction', 'relu'),
            )
        return ('relu',) * 3

    def _print_activation_summary(self):
        cnn_config, fusion_config, pred_config = self._parse_activation_config()
        print(f"激活函數配置: CNN={cnn_config}  Fusion={fusion_config}  Pred={pred_config}")

    def encode(self, predictor):
        batch_size, seq_len, C, H, W = predictor.shape
        assert H == self.img_height and W == self.img_width
        encoder_hidden = None
        for t in range(seq_len):
            geo_feat = self.geo_cnn(predictor[:, t])
            patch_feat = self.patch_project(geo_feat)
            B, C_embed, H_patch, W_patch = patch_feat.shape
            patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
            encoded_feat, encoder_hidden = self.encoder_vmrnn_cell(patch_embed_feat, encoder_hidden)
        latent_for_probe = encoder_hidden[0].mean(dim=1)
        if self.collect_features:
            self.feature_buffer['L3_latent_probe'].append(
                latent_for_probe.detach().cpu().numpy())
        return encoded_feat, encoder_hidden, latent_for_probe

    def decode_step(self, current_input, encoder_output, decoder_hidden, climate_features=None):
        if not hasattr(self, '_decode_step_count'):
            self._decode_step_count = 0
        self._decode_step_count += 1
        _is_first = (self._decode_step_count == 1)

        geo_feat = self.geo_cnn(current_input)
        patch_feat = self.patch_project(geo_feat)
        B, C_embed, H_patch, W_patch = patch_feat.shape
        patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
        decoded_feat, new_decoder_hidden = self.decoder_vmrnn_cell(patch_embed_feat, decoder_hidden)
        decoded_img  = decoded_feat.permute(0, 2, 1).view(B, self.embed_dim, H_patch, W_patch)
        vmrnn_feat   = self.patch_to_img(decoded_img)
        fused_features = torch.cat([geo_feat, vmrnn_feat], dim=1)
        fused_features_conv = self.fusion_norm(self.fusion_conv(fused_features))

        activation_weights = None

        if self.use_dynamic_activation and isinstance(self.fusion_activation, PhysicalLatentGating):
            decoder_latent = new_decoder_hidden[0].max(dim=1).values
            if hasattr(self, '_current_nino_signal') and self._current_nino_signal is not None:
                nino_embed = self.nino_injector(self._current_nino_signal.unsqueeze(-1))
                latent_context = decoder_latent + nino_embed
            else:
                latent_context = decoder_latent
            try:
                fused_features, activation_weights = self.fusion_activation(
                    fused_features_conv, latent_context)
                if self.collect_features and _is_first:
                    gate_mod = self.fusion_activation
                    if gate_mod._last_logits_grad is not None:
                        self.feature_buffer['L5_gate_logits'].append(
                            gate_mod._last_logits_grad.detach().cpu().numpy())
                        self.feature_buffer['L6_gate_weights'].append(
                            F.softmax(gate_mod._last_logits_grad.detach(), dim=-1).cpu().numpy())
                    if isinstance(fused_features, torch.Tensor) and fused_features.dim() == 4:
                        self.feature_buffer['L7_after_gate'].append(
                            fused_features.detach().mean(dim=[2, 3]).cpu().numpy())
            except Exception as e:
                print(f"[Warning] PhysicalLatentGating 失敗: {e}")
                fused_features = F.relu(fused_features_conv, inplace=False)
        elif self.use_dynamic_activation:
            fused_features, activation_weights = self.fusion_activation(fused_features_conv)
        else:
            fused_features = self.fusion_activation(fused_features_conv) \
                if callable(self.fusion_activation) \
                else F.relu(fused_features_conv, inplace=False)

        if isinstance(fused_features, (list, tuple)):
            fused_features = fused_features[0]

        prediction = self.prediction_head(fused_features)
        residual   = self.residual_head(fused_features)
        final_prediction = prediction + residual
        H, W = current_input.shape[-2:]
        final_prediction = final_prediction.view(B, self.cube_dim, H, W)

        return final_prediction, prediction, residual, new_decoder_hidden, activation_weights

    def enable_feature_collection(self):
        self.collect_features = True
        self._decode_step_count = 0
        for k in self.feature_buffer:
            self.feature_buffer[k] = []

    def disable_feature_collection(self):
        self.collect_features = False
        import numpy as np
        result = {}
        for k, v in self.feature_buffer.items():
            if v:
                result[k] = np.concatenate(v, axis=0)
        return result

    def reset_decode_step_count(self):
        self._decode_step_count = 0

    def forward(self, predictor, predictand=None, train=True, sv_ratio=0,
                climate_features=None, nino_true=None, nino_probe_weight=0.1):
        batch_size, seq_len, C, H, W = predictor.shape

        encoder_output, encoder_hidden, latent_for_probe = self.encode(predictor)

        with torch.no_grad():
            nino_estimate = self.nino_probe(latent_for_probe)
        self._current_nino_signal = nino_estimate

        probe_loss = torch.tensor(0.0, device=self.device)
        if self.nino_probe_enabled and nino_true is not None and train:
            nino_pred_probe = self.nino_probe(latent_for_probe)
            regression_loss = F.huber_loss(
                nino_pred_probe, nino_true.float().to(self.device), delta=1.0)
            phase_labels_for_probe = get_enso_phase_label_tensor(nino_true.to(self.device))
            phase_logits = self.nino_phase_probe(latent_for_probe)
            classification_loss = F.cross_entropy(phase_logits, phase_labels_for_probe)
            probe_loss = classification_loss + 0.3 * regression_loss

        if train:
            decoder_hidden = encoder_hidden
            outputs = []
            mse = nn.MSELoss()
            residual_losses = []
            current_input = predictor[:, -1]

            for t in range(self.mypara.output_length):
                next_step, coarse_pred, residual_pred, decoder_hidden, activation_weights = \
                    self.decode_step(current_input, encoder_output, decoder_hidden, climate_features)
                outputs.append(next_step)
                residual_losses.append(mse(residual_pred, predictand[:, t] - coarse_pred))
                if t < self.mypara.output_length-1:
                    current_input = predictand[:, t]

            outvar_pred = torch.stack(outputs, dim=1)

            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio * torch.ones(batch_size, self.mypara.output_length - 1, 1, 1, 1)
                ).to(self.device)
                mixed_predictand = (
                    supervise_mask * predictand[:, :-1] +
                    (1 - supervise_mask) * outvar_pred[:, :-1]
                )
                decoder_hidden = encoder_hidden
                outputs = []
                residual_losses = []
                current_input = predictor[:, -1]
                for t in range(self.mypara.output_length):
                    next_step, coarse_pred, residual_pred, decoder_hidden, activation_weights = \
                        self.decode_step(current_input, encoder_output, decoder_hidden, climate_features)
                    outputs.append(next_step)
                    residual_losses.append(mse(residual_pred, predictand[:, t] - coarse_pred))
                    if t < self.mypara.output_length - 1:
                        current_input = mixed_predictand[:, t]
                outvar_pred = torch.stack(outputs, dim=1)
        else:
            decoder_hidden = encoder_hidden
            outputs = []
            current_input = predictor[:, -1]
            residual_losses = []
            for t in range(self.mypara.output_length):
                next_step, coarse_pred, residual_pred, decoder_hidden, activation_weights = \
                    self.decode_step(current_input, encoder_output, decoder_hidden, climate_features)
                outputs.append(next_step)
                current_input = next_step
                if predictand is not None:
                    residual_losses.append(
                        nn.MSELoss()(residual_pred, predictand[:, t] - coarse_pred))
                else:
                    residual_losses.append(torch.tensor(0.0, device=self.device))
            outvar_pred = torch.stack(outputs, dim=1)

        mean_residual_loss = torch.stack(residual_losses).mean() if residual_losses \
            else torch.tensor(0.0, device=self.device)

        return outvar_pred, mean_residual_loss, probe_loss

    def predict(self, predictor):
        return self.forward(predictor, train=False)

    def get_activation_weights_history(self):
        if not self.use_dynamic_activation:
            return None
        if hasattr(self.fusion_activation, 'weights_history') and \
           self.fusion_activation.weights_history:
            return torch.cat(self.fusion_activation.weights_history, dim=0).numpy()
        return None

    def clear_activation_weights_history(self):
        if self.use_dynamic_activation and hasattr(self.fusion_activation, 'weights_history'):
            self.fusion_activation.weights_history = []


# =============================================================================
# 工廠函數
# =============================================================================

def create_supervised_geo_vmrnn(mypara, activation_config=None):
    return GeoVMRNN_Enhance(mypara, activation_config=activation_config)


def build_physical_latent_gating(hidden_dim, use_hard_selection=False):
    """
    ★ 2激活函數版工廠函數（ReLU + LearnedSnake）
    與原版 3激活函數版的差異：
      - num_activations = 2
      - diversity_target = [0.5, 0.5]
      - min_weight = 0.15（2類時每類最少 15%）
    """
    return PhysicalLatentGating(
        hidden_dim         = hidden_dim,
        num_activations    = 2,              # ★ 改為 2
        use_hard_selection = use_hard_selection,
        # 方案 1
        init_temperature   = 10.0,
        min_temperature    = 1.0,
        temperature_decay  = 0.9998,
        # 方案 4
        warmup_steps       = 10000,
        noise_scale        = 1.0,
        # 方案 A
        enforce_balance    = True,
        min_weight         = 0.15,           # ★ 2類時每類最少 15%
        balance_steps      = 15000,
        # 方案 B（均勻二元分佈）
        diversity_target   = [0.5, 0.5],     # ★ 改為二元均勻
        # 方案 F
        phase_gate_loss_weight = 1.0,
    )


def get_ablation_configs():
    return {
        'baseline': {
            'cnn': 'relu', 'fusion': 'relu', 'prediction': 'relu'
        },
        'fusion_learned': {
            'cnn': 'relu', 'fusion': 'snake_learned', 'prediction': 'relu'
        },
        'physical_gating_2act': {          # ★ 2激活函數版
            'cnn': 'relu',
            'fusion': {
                'type'  : 'physical_gating',
                'params': {'num_activations': 2, 'use_hard_selection': False}
            },
            'prediction': 'relu'
        },
    }
