"""
無監督動態激活函數訓練腳本
讓模型從 encoder latent features 自動發現氣候 regime
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import math
import os
import gc
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

from myconfig import mypara
from LoadData import make_dataset2, make_testdataset
from progressive_teacher_forcing import TeacherForcingScheduler
from UnsupervisedDynamicActivation import (
    UnsupervisedDynamicActivation, 
    RegimeAnalyzer,
    LearnedSnake
)

# 內存優化設置
torch.cuda.empty_cache()
if hasattr(torch.cuda, 'memory_stats'):
    print(f"初始 CUDA 內存狀態: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

try:
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("已設置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
except Exception as e:
    print(f"設置 CUDA 內存分配器配置時出錯: {e}")


class GeoVMRNN_Unsupervised(nn.Module):
    """
    整合無監督動態激活函數的 GeoVMRNN
    """
    def __init__(self, base_model, latent_dim=256, num_regimes=3):
        super().__init__()
        self.base_model = base_model
        self.latent_dim = latent_dim
        
        # 替換原來的 fusion_activation 為無監督版本
        self.unsupervised_activation = UnsupervisedDynamicActivation(
            latent_dim=latent_dim,
            num_regimes=num_regimes,
            activation_types=['relu', 'snake', 'gelu'],
            regime_detection_method='learnable',  # 可選: 'kmeans', 'som'
            use_regime_specific_weights=True
        )
        
        # 用於記錄分析
        self.activation_info_history = []
    
    def forward(self, predictor, predictand=None, train=True, sv_ratio=0):
        """
        修改的前向傳播，使用無監督動態激活
        """
        batch_size, seq_len, C, H, W = predictor.shape
        
        # 1. Encode
        encoder_output, encoder_hidden = self.base_model.encode(predictor)
        
        # encoder_output 形狀: [B, L, latent_dim]
        # 用於 regime 檢測
        
        # 2. Decode
        if train:
            assert predictand is not None, "訓練模式需要 predictand"
            
            decoder_hidden = encoder_hidden
            outputs = []
            mse = nn.MSELoss()
            residual_losses = []
            current_input = predictor[:, -1]
            
            # 記錄所有時間步的 activation info
            all_activation_info = []
            
            for t in range(self.base_model.mypara.output_length):
                # 修改 decode_step 以使用無監督激活
                next_step, coarse_pred, residual_pred, decoder_hidden, activation_info = \
                    self._decode_step_unsupervised(
                        current_input, encoder_output, decoder_hidden
                    )
                
                outputs.append(next_step)
                all_activation_info.append(activation_info)
                
                # 計算殘差損失
                residual_target = predictand[:, t] - coarse_pred
                loss_r = mse(residual_pred, residual_target)
                residual_losses.append(loss_r)
                
                if t < self.base_model.mypara.output_length - 1:
                    current_input = predictand[:, t]
            
            outvar_pred = torch.stack(outputs, dim=1)
            
            # 應用 scheduled sampling
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio * torch.ones(batch_size, self.base_model.mypara.output_length - 1, 1, 1, 1)
                ).to(self.base_model.device)
                
                mixed_predictand = (
                    supervise_mask * predictand[:, :-1] + 
                    (1 - supervise_mask) * outvar_pred[:, :-1]
                )
                
                # 重新預測
                decoder_hidden = encoder_hidden
                outputs = []
                residual_losses = []
                current_input = predictor[:, -1]
                all_activation_info = []
                
                for t in range(self.base_model.mypara.output_length):
                    next_step, coarse_pred, residual_pred, decoder_hidden, activation_info = \
                        self._decode_step_unsupervised(
                            current_input, encoder_output, decoder_hidden
                        )
                    
                    outputs.append(next_step)
                    all_activation_info.append(activation_info)
                    
                    residual_target = predictand[:, t] - coarse_pred
                    loss_r = mse(residual_pred, residual_target)
                    residual_losses.append(loss_r)
                    
                    if t < self.base_model.mypara.output_length - 1:
                        current_input = mixed_predictand[:, t]
                
                outvar_pred = torch.stack(outputs, dim=1)
            
            # 記錄 activation info
            if self.training:
                self.activation_info_history.extend(all_activation_info)
        
        else:
            # 推理模式
            decoder_hidden = encoder_hidden
            outputs = []
            current_input = predictor[:, -1]
            residual_losses = []
            all_activation_info = []
            
            for t in range(self.base_model.mypara.output_length):
                next_step, coarse_pred, residual_pred, decoder_hidden, activation_info = \
                    self._decode_step_unsupervised(
                        current_input, encoder_output, decoder_hidden
                    )
                
                outputs.append(next_step)
                all_activation_info.append(activation_info)
                current_input = next_step
                
                if predictand is not None:
                    residual_target = predictand[:, t] - coarse_pred
                    loss_r = mse(residual_pred, residual_target)
                    residual_losses.append(loss_r)
                else:
                    residual_losses.append(torch.tensor(0.0, device=self.base_model.device))
            
            outvar_pred = torch.stack(outputs, dim=1)
        
        # 計算平均殘差損失
        mean_residual_loss = torch.stack(residual_losses).mean() if residual_losses else torch.tensor(0.0)
        
        # 返回額外的 activation info 用於分析
        return outvar_pred, mean_residual_loss, all_activation_info
    
    def _decode_step_unsupervised(self, current_input, encoder_output, decoder_hidden):
        """
        修改的單步解碼，使用無監督動態激活
        """
        # 地理特徵提取
        geo_feat = self.base_model.geo_cnn(current_input)
        
        # Patch 嵌入
        patch_feat = self.base_model.patch_project(geo_feat)
        B, C_embed, H_patch, W_patch = patch_feat.shape
        patch_embed_feat = patch_feat.view(B, C_embed, H_patch * W_patch).permute(0, 2, 1)
        
        # VMRNN 解碼
        decoded_feat, new_decoder_hidden = self.base_model.decoder_vmrnn_cell(
            patch_embed_feat, decoder_hidden
        )
        
        # 轉回圖像格式
        decoded_img = decoded_feat.permute(0, 2, 1).view(B, self.base_model.embed_dim, H_patch, W_patch)
        vmrnn_feat = self.base_model.patch_to_img(decoded_img)
        
        # Encoder 特徵
        encoder_img = encoder_output.permute(0, 2, 1).view(B, self.base_model.embed_dim, H_patch, W_patch)
        encoder_feat = self.base_model.patch_to_img(encoder_img)
        
        # 融合特徵
        fused_features = torch.cat([geo_feat, vmrnn_feat], dim=1)
        fused_features_conv = self.base_model.fusion_norm(
            self.base_model.fusion_conv(fused_features)
        )
        
        # 使用無監督動態激活（使用 encoder latent features）
        fused_features, activation_info = self.unsupervised_activation(
            x=fused_features_conv,
            latent_features=encoder_output  # 傳入 encoder latent
        )
        
        # 預測
        prediction = self.base_model.prediction_head(fused_features)
        residual = self.base_model.residual_head(fused_features)
        final_prediction = prediction + residual
        
        # 恢復原始尺寸
        H, W = current_input.shape[-2:]
        final_prediction = final_prediction.view(B, self.base_model.cube_dim, H, W)
        
        return final_prediction, prediction, residual, new_decoder_hidden, activation_info


class UnsupervisedTrainer:
    """無監督動態激活訓練器"""
    def __init__(self, mypara):
        self.mypara = mypara
        self.device = mypara.device
        
        # 創建基礎模型
        from GeoVMRNN_resnet import create_resnet_geo_vmrnn
        base_model = create_resnet_geo_vmrnn(mypara, activation_config='relu')
        
        # 包裝為無監督版本
        self.mymodel = GeoVMRNN_Unsupervised(
            base_model=base_model,
            latent_dim=256,  # encoder 輸出維度
            num_regimes=3    # 預設 3 個 regime
        ).to(mypara.device)
        
        # 優化器
        self.opt = torch.optim.Adam(self.mymodel.parameters(), lr=1e-4)
        
        # SST 層級
        self.sstlevel = 2 if self.mypara.needtauxy else 0
        
        # Nino 權重
        ninoweight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(mypara.device)
        self.ninoweight = ninoweight[:self.mypara.output_length]
        
        # Teacher Forcing 調度器
        self.tf_scheduler = TeacherForcingScheduler(
            strategy='exponential',
            initial_ratio=1.0,
            final_ratio=0.0,
            decay_rate=0.9999
        )
        
        # Regime 分析器
        self.regime_analyzer = RegimeAnalyzer()
        
        print(f"模型總參數: {sum(p.numel() for p in self.mymodel.parameters()):,}")
        print(f"使用無監督動態激活函數")
    
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
        rmse = rmse.mean(dim=0).sum()
        
        if residual_losses is not None:
            if isinstance(residual_losses, torch.Tensor):
                residual_term = alpha * residual_losses
            else:
                residual_term = alpha * torch.tensor(residual_losses, device=rmse.device)
        else:
            residual_term = torch.tensor(0.0, device=rmse.device)
        
        return rmse + residual_term
    
    def loss_nino(self, y_pred, y_true):
        """計算 Nino 損失"""
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()
    
    def combine_loss(self, loss1, loss2):
        """組合損失"""
        return loss1 + loss2
    
    def train_model(self, dataset_train, dataset_eval, num_epochs=50):
        """訓練模型"""
        model_name = "GeoVMRNN_Unsupervised.pkl"
        chk_path = self.mypara.model_savepath + model_name
        torch.manual_seed(self.mypara.seeds)
        
        # 調整批次大小
        batch_size_train = max(1, self.mypara.batch_size_train // 2)
        batch_size_eval = max(1, self.mypara.batch_size_eval // 2)
        
        print(f"訓練批次大小: {batch_size_train}, 評估批次大小: {batch_size_eval}")
        
        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size_train, shuffle=False
        )
        dataloader_eval = DataLoader(
            dataset_eval, batch_size=batch_size_eval, shuffle=False
        )
        
        best = -math.inf
        global_step = 0
        
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment("GeoVMRNN_Unsupervised")
        
        with mlflow.start_run():
            mlflow.set_tag("model", "GeoVMRNN_Unsupervised")
            mlflow.log_params({
                "activation": "unsupervised_dynamic",
                "num_regimes": 3,
                "lr": 1e-4,
                "batch_size_train": batch_size_train,
                "epochs": num_epochs
            })
            
            for i_epoch in range(num_epochs):
                print("=" * 80)
                print(f"\n--> Epoch: {i_epoch}")
                
                # 訓練階段
                self.mymodel.train()
                
                for j, (input_var, var_true) in enumerate(dataloader_train):
                    # 提取 SST 和 Nino
                    SST = var_true[:, :, self.sstlevel]
                    nino_true = SST[
                        :, :,
                        self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                        self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1],
                    ].mean(dim=[2, 3])
                    
                    # 前向傳播
                    current_tf_ratio = self.tf_scheduler.get_ratio()
                    var_pred, residual_loss, activation_info_list = self.mymodel(
                        input_var.float().to(self.device),
                        var_true.float().to(self.device),
                        train=True,
                        sv_ratio=current_tf_ratio,
                    )
                    
                    # 提取預測
                    SST_pred = var_pred[:, :, self.sstlevel]
                    nino_pred = SST_pred[
                        :, :,
                        self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                        self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1],
                    ].mean(dim=[2, 3])
                    
                    # 計算損失
                    self.opt.zero_grad()
                    loss_var = self.loss_var(var_pred, var_true.float().to(self.device), residual_loss)
                    loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
                    score = self.calscore(nino_pred, nino_true.float().to(self.device))
                    combine_loss = self.combine_loss(loss_var, loss_nino)
                    
                    # 反向傳播
                    combine_loss.backward()
                    self.opt.step()
                    self.tf_scheduler.step()
                    
                    # 記錄 regime 分析數據
                    if activation_info_list and len(activation_info_list) > 0:
                        # 使用最後一個時間步的 activation info
                        last_info = activation_info_list[-1]
                        self.regime_analyzer.add_data(
                            last_info['regime_labels'],
                            SST[:, -1],  # 使用最後一個時間步的 SST
                            (
                                self.mypara.lat_nino_relative[0],
                                self.mypara.lat_nino_relative[1],
                                self.mypara.lon_nino_relative[0],
                                self.mypara.lon_nino_relative[1]
                            )
                        )
                        
                        # 記錄激活函數權重
                        regime_dist = last_info['regime_distribution']
                        mlflow.log_metric("Train/regime_entropy", 
                                         -(regime_dist * torch.log(regime_dist + 1e-8)).sum().item(),
                                         step=global_step)
                        
                        for k in range(len(regime_dist)):
                            mlflow.log_metric(f"Train/regime_{k}_prob", 
                                             regime_dist[k].item(), 
                                             step=global_step)
                    
                    # 記錄訓練指標
                    mlflow.log_metric("Train/Loss_Var", loss_var.item(), step=global_step)
                    mlflow.log_metric("Train/Loss_Nino", loss_nino.item(), step=global_step)
                    mlflow.log_metric("Train/Combine_Loss", combine_loss.item(), step=global_step)
                    mlflow.log_metric("Train/Score", score, step=global_step)
                    mlflow.log_metric("Train/tf_ratio", current_tf_ratio, step=global_step)
                    
                    global_step += 1
                    
                    # 打印進度
                    if j % 100 == 0:
                        print(f"\n--> Batch {j}: loss_var={loss_var:.2f}, "
                              f"loss_nino={loss_nino:.2f}, score={score:.3f}")
                        
                        if torch.cuda.is_available():
                            print(f"CUDA 內存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                        
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # 驗證
                    if (i_epoch + 1 >= 4) and (j + 1) % 400 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        eval_results = self.evaluate(dataloader_eval)
                        
                        print(f"\n--> Evaluation: "
                              f"loss_var={eval_results['loss_var']:.3f}, "
                              f"loss_nino={eval_results['loss_nino']:.3f}, "
                              f"score={eval_results['score']:.3f}")
                        
                        mlflow.log_metric("Eval/Loss_Var", eval_results['loss_var'], step=global_step)
                        mlflow.log_metric("Eval/Loss_Nino", eval_results['loss_nino'], step=global_step)
                        mlflow.log_metric("Eval/Score", eval_results['score'], step=global_step)
                        
                        if eval_results['score'] > best:
                            torch.save(self.mymodel.state_dict(), chk_path)
                            best = eval_results['score']
                            print(f"\n保存模型... 最佳評分: {best:.3f}")
                        
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Epoch 結束評估
                torch.cuda.empty_cache()
                gc.collect()
                
                eval_results = self.evaluate(dataloader_eval)
                
                print(f"\n--> Epoch {i_epoch} 結束: "
                      f"loss_var={eval_results['loss_var']:.3f}, "
                      f"loss_nino={eval_results['loss_nino']:.3f}, "
                      f"score={eval_results['score']:.3f}")
                
                mlflow.log_metric("Epoch/Loss_Var", eval_results['loss_var'], step=i_epoch)
                mlflow.log_metric("Epoch/Loss_Nino", eval_results['loss_nino'], step=i_epoch)
                mlflow.log_metric("Epoch/Score", eval_results['score'], step=i_epoch)
                
                if eval_results['score'] > best:
                    torch.save(self.mymodel.state_dict(), chk_path)
                    best = eval_results['score']
                    print(f"\n保存模型... 最佳評分: {best:.3f}")
                
                # 每 5 個 epoch 進行 regime 分析
                if (i_epoch + 1) % 5 == 0:
                    self.regime_analyzer.print_analysis_report()
                    save_path = os.path.join(
                        self.mypara.model_savepath, 
                        f'regime_analysis_epoch_{i_epoch}.png'
                    )
                    self.regime_analyzer.plot_regime_analysis(save_path)
                    mlflow.log_artifact(save_path)
        
        # 最終分析
        print("\n" + "="*80)
        print("最終 Regime 分析")
        print("="*80)
        self.regime_analyzer.print_analysis_report()
        final_plot = os.path.join(self.mypara.model_savepath, 'regime_analysis_final.png')
        self.regime_analyzer.plot_regime_analysis(final_plot)
        
        return best
    
    def evaluate(self, dataloader):
        """評估模型"""
        self.mymodel.eval()
        var_pred_list = []
        nino_pred_list = []
        var_true_list = []
        nino_true_list = []
        
        with torch.no_grad():
            for input_var, var_true in dataloader:
                SST = var_true[:, :, self.sstlevel]
                nino_true = SST[
                    :, :,
                    self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                
                var_pred, residual_loss, _ = self.mymodel(
                    input_var.float().to(self.device),
                    predictand=None,
                    train=False,
                )
                
                SST_pred = var_pred[:, :, self.sstlevel]
                nino_pred = SST_pred[
                    :, :,
                    self.mypara.lat_nino_relative[0]:self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0]:self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                
                var_true_list.append(var_true.cpu())
                nino_true_list.append(nino_true.cpu())
                var_pred_list.append(var_pred.cpu())
                nino_pred_list.append(nino_pred.cpu())
                
                if len(var_pred_list) % 5 == 0:
                    torch.cuda.empty_cache()
            
            var_pred = torch.cat(var_pred_list, dim=0).to(self.device)
            nino_pred = torch.cat(nino_pred_list, dim=0).to(self.device)
            nino_true = torch.cat(nino_true_list, dim=0).to(self.device)
            var_true = torch.cat(var_true_list, dim=0).to(self.device)
            
            ninosc = self.calscore(nino_pred, nino_true)
            loss_var = self.loss_var(var_pred, var_true, residual_losses=None).item()
            loss_nino = self.loss_nino(nino_pred, nino_true).item()
            combine_loss = self.combine_loss(loss_var, loss_nino)
            
            del var_pred, nino_pred, nino_true, var_true
            torch.cuda.empty_cache()
        
        return {
            'loss_var': loss_var,
            'loss_nino': loss_nino,
            'combine_loss': combine_loss,
            'score': ninosc
        }


def train_unsupervised():
    """訓練無監督動態激活模型"""
    print("\n" + "="*80)
    print("開始訓練無監督動態激活函數模型")
    print("模型將從 encoder latent features 自動發現氣候 regime")
    print("="*80 + "\n")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(mypara.__dict__)
    print("\n載入訓練數據集...")
    traindataset = make_dataset2(mypara)
    print(traindataset.selectregion())
    
    print("\n載入評估數據集...")
    evaldataset = make_testdataset(mypara, ngroup=100)
    print(evaldataset.selectregion())
    
    try:
        trainer = UnsupervisedTrainer(mypara)
        best_score = trainer.train_model(
            dataset_train=traindataset,
            dataset_eval=evaldataset,
            num_epochs=50
        )
        
        print(f"\n訓練完成！最佳評分: {best_score:.3f}")
        print("\n請檢查 regime_analysis 圖表以驗證:")
        print("  1. 無監督發現的 regime 是否與 ENSO phase 對應")
        print("  2. Mutual Information 和 ARI 指標")
        print("  3. 各 regime 的 Niño 指數分布")
        
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"\n內存不足: {e}")
            print("請減小批次大小或模型大小")
        else:
            print(f"\n運行時錯誤: {e}")
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    train_unsupervised()
