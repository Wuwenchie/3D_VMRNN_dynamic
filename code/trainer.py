from 3D_VMRNN import create_supervised_geo_vmrnn, get_ablation_configs
from DynamicActivationSelector import DynamicActivationLoss, detect_climate_phenomenon
from myconfig import mypara
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from LoadData import make_dataset2, make_testdataset
from progressive_teacher_forcing import TeacherForcingScheduler
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os
import gc  # 添加垃圾回收模块

# 设置 PyTorch 内存分配器配置
# 这有助于减少内存碎片化
torch.cuda.empty_cache()
if hasattr(torch.cuda, 'memory_stats'):
    print(f"初始 CUDA 内存状态: {torch.cuda.memory_allocated() / 1024**2:.2f} MB 已分配")

# 尝试设置 PyTorch CUDA 内存分配器配置
try:
    # 设置环境变量以避免内存碎片化
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
except Exception as e:
    print(f"设置 CUDA 内存分配器配置时出错: {e}")


class DynamicActivationTrainer:
    """動態激活函數訓練器"""
    def __init__(self, mypara):
        self.mypara = mypara
        self.device = mypara.device
        
        # 獲取激活函數配置
        self.activation_configs = get_ablation_configs()
        
        # 創建模型
        self.mymodel = create_supervised_geo_vmrnn(
            mypara, 
            activation_config=self.activation_configs['dynamic_fusion']
        ).to(mypara.device)
        
        # 添加模型參數統計
        total_params = sum(p.numel() for p in self.mymodel.parameters())
        trainable_params = sum(p.numel() for p in self.mymodel.parameters() if p.requires_grad)
        print(f"模型總參數: {total_params:,}")
        print(f"可訓練參數: {trainable_params:,}")
        print(f"使用動態激活函數配置")
        
        # 設置優化器和學習率調度器
        adam = torch.optim.Adam(self.mymodel.parameters(), lr=0)
        factor = math.sqrt(mypara.d_size * mypara.warmup) * 0.0015
        self.opt = torch.optim.Adam(self.mymodel.parameters(), lr=1e-4)
        
        # 設置 SST 層級
        self.sstlevel = 0
        if self.mypara.needtauxy:
            self.sstlevel = 2
        
        # 設置 Nino 指數權重
        ninoweight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(mypara.device)
        self.ninoweight = ninoweight[: self.mypara.output_length]

        # 創建 Teacher Forcing 調度器
        self.tf_scheduler = TeacherForcingScheduler(
            strategy='exponential',
            initial_ratio=1.0,
            final_ratio=0.0,
            decay_rate=0.9999
        )
        
        # 創建動態激活函數損失
        self.dynamic_loss = DynamicActivationLoss(
            main_loss_fn=self.combine_loss,
            regularization_weight=0.1
        )
        
        # 記錄激活函數權重歷史
        self.activation_weights_history = []
    
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
        # Ensure y_pred and y_true have the same sequence length
        min_len = min(y_pred.size(1), y_true.size(1))
        y_pred = y_pred[:, :min_len]
        y_true = y_true[:, :min_len]
        
        # Calculate RMSE over spatial dimensions first
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])  # Average over height and width
        
        # Average over batch dimension
        rmse = rmse.mean(dim=0)  # Average over batch
        
        # Sum over remaining dimensions (sequence and channels)
        rmse = rmse.sum()
        
        # 處理 residual_losses
        if residual_losses is not None:
            # 如果 residual_losses 是 tensor，直接使用
            if isinstance(residual_losses, torch.Tensor):
                residual_term = alpha * residual_losses
            else:
                # 如果是其他類型（如 float），轉換為 tensor
                residual_term = alpha * torch.tensor(residual_losses, device=rmse.device)
        else:
            # 如果沒有提供 residual_losses，設為 0
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
    
    def train_model(self, dataset_train, dataset_eval, num_epochs=50):
        """訓練模型"""
        # 設置模型保存路徑
        model_name = "GeoVMRNN_DynamicActivation.pkl"
        chk_path = self.mypara.model_savepath + model_name
        torch.manual_seed(self.mypara.seeds)
        
        # 創建數據加載器 - 减小批次大小以降低内存使用
        batch_size_train = max(1, self.mypara.batch_size_train // 2)  # 减小训练批次大小
        batch_size_eval = max(1, self.mypara.batch_size_eval // 2)    # 减小评估批次大小
        
        print(f"原始训练批次大小: {self.mypara.batch_size_train}, 调整后: {batch_size_train}")
        print(f"原始评估批次大小: {self.mypara.batch_size_eval}, 调整后: {batch_size_eval}")
        
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
        print(f"使用動態激活函數配置")
        print(f"模型將保存到: {chk_path}")

        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment("GeoVMRNN_DynamicActivation")
        
        with mlflow.start_run():
            mlflow.set_tag("model", "GeoVMRNN_DynamicActivation")
            mlflow.log_params({
                "activation_config": "dynamic_fusion",
                "lr": 1e-4,
                "batch_size_train": batch_size_train,
                "batch_size_eval": batch_size_eval,
                "epochs": num_epochs,
                "tf_strategy": self.tf_scheduler.strategy
            })
        
            for i_epoch in range(num_epochs):
                print("=========="*8)
                print(f"\n-->epoch: {i_epoch}")
                
                # 訓練階段
                self.mymodel.train()
                epoch_activation_weights = []
                
                for j, (input_var, var_true) in enumerate(dataloader_train):
                    # 提取真實 SST 和 Nino 指數
                    SST = var_true[:, :, self.sstlevel]
                    nino_true = SST[
                        :,
                        :,
                        self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                        self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                    ].mean(dim=[2, 3])
                    
                    # 檢測氣候現象類型
                    climate_labels = detect_climate_phenomenon(
                        var_true[:, :, self.sstlevel],  # 使用最後一個時間步的 SST
                        nino_region=(
                            self.mypara.lat_nino_relative[0],
                            self.mypara.lat_nino_relative[1],
                            self.mypara.lon_nino_relative[0],
                            self.mypara.lon_nino_relative[1]
                        )
                    )
                    
                    # 獲取當前的 Teacher Forcing 比例
                    current_tf_ratio = self.tf_scheduler.get_ratio()
                    
                    # 前向傳播
                    var_pred, residual_loss = self.mymodel(
                        input_var.float().to(self.device),
                        var_true.float().to(self.device),
                        train=True,
                        sv_ratio=current_tf_ratio,
                    )
                    
                    # 提取預測的 SST 和 Nino 指數
                    SST_pred = var_pred[:, :, self.sstlevel]
                    nino_pred = SST_pred[
                        :,
                        :,
                        self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                        self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                    ].mean(dim=[2, 3])
                    
                    # 計算損失
                    self.opt.zero_grad()
                    loss_var = self.loss_var(var_pred, var_true.float().to(self.device), residual_loss)
                    loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
                    score = self.calscore(nino_pred, nino_true.float().to(self.device))
                    
                    # 使用動態激活函數損失
                    # 獲取激活函數權重
                    activation_weights = None
                    if hasattr(self.mymodel, 'fusion_activation') and hasattr(self.mymodel.fusion_activation, 'selector'):
                        activation_weights = self.mymodel.fusion_activation.selector.weights_history[-1] if self.mymodel.fusion_activation.selector.weights_history else None
                    
                    if activation_weights is not None:
                        # 記錄激活函數權重
                        epoch_activation_weights.append(activation_weights.detach().cpu())
                        
                        # 計算動態激活函數損失
                        combine_loss = self.dynamic_loss(
                            pred=(var_pred, nino_pred),
                            target=(var_true.float().to(self.device), nino_true.float().to(self.device)),
                            activation_weights=activation_weights,
                            climate_labels=climate_labels.to(self.device)
                        )
                    else:
                        # 使用普通損失
                        combine_loss = self.combine_loss(loss_var, loss_nino)
                    
                    # 反向傳播
                    combine_loss.backward()
                    self.opt.step()

                    # 更新 Teacher Forcing 調度器
                    self.tf_scheduler.step()
                        
                    mlflow.log_metric("Train/Loss_Var", loss_var.item(), step=global_step)
                    mlflow.log_metric("Train/Loss_Nino", loss_nino.item(), step=global_step)
                    mlflow.log_metric("Train/Combine_Loss", combine_loss.item(), step=global_step)
                    mlflow.log_metric("Train/Score", score, step=global_step)
                    mlflow.log_metric("Train/tf_ratio", current_tf_ratio, step=global_step)
                    mlflow.log_metric("Train/Loss_Residual", residual_loss.item(), step=global_step)
                    
                    # 記錄激活函數權重
                    if activation_weights is not None:
                        for i, weight in enumerate(activation_weights):
                            mlflow.log_metric(f"Train/Activation_Weight_{i}", weight.item(), step=global_step)
                    
                    global_step += 1

                    # 打印訓練進度
                    if j % 100 == 0:
                        print(
                            f"\n-->batch:{j} loss_var:{loss_var:.2f}, loss_nino:{loss_nino:.2f}, score:{score:.3f}, dynamic_activation"
                        )
                        if activation_weights is not None:
                            print(f"Activation weights: {activation_weights.detach().cpu().numpy()}")
                        
                        # 打印内存使用情况
                        if torch.cuda.is_available():
                            print(f"CUDA 内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB 已分配, {torch.cuda.memory_reserved() / 1024**2:.2f} MB 已保留")
                        
                        # 主动清理缓存
                        torch.cuda.empty_cache()
                        gc.collect()

                    # 密集驗證 - 减少验证频率以降低内存压力
                    if (i_epoch + 1 >= 4) and (j + 1) % 400 == 0:  # 从200改为400
                        # 清理内存
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        eval_results = self.evaluate(dataloader_eval)
                        
                        print(
                            f"-->Evaluation... \nloss_var:{eval_results['loss_var']:.3f} \nloss_nino:{eval_results['loss_nino']:.3f} \nloss_com:{eval_results['combine_loss']:.3f} \nscore:{eval_results['score']:.3f}"
                        )
                        mlflow.log_metric("Eval/Loss_Var", eval_results['loss_var'], step=global_step)
                        mlflow.log_metric("Eval/Loss_Nino", eval_results['loss_nino'], step=global_step)
                        mlflow.log_metric("Eval/Combine_Loss", eval_results['combine_loss'], step=global_step)
                        mlflow.log_metric("Eval/Score", eval_results['score'], step=global_step)
                        
                        # 記錄激活函數權重
                        if 'activation_weights' in eval_results and eval_results['activation_weights'] is not None:
                            for i, weight in enumerate(eval_results['activation_weights']):
                                mlflow.log_metric(f"Eval/Activation_Weight_{i}", weight, step=global_step)

                        if eval_results['score'] > best:
                            torch.save(self.mymodel.state_dict(), chk_path)
                            best = eval_results['score']
                            count = 0
                            print(f"\nsaving model with dynamic activation...")
                        
                        # 清理内存
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # 記錄每個 epoch 的激活函數權重
                if epoch_activation_weights:
                    epoch_weights = torch.stack(epoch_activation_weights).mean(dim=0)
                    self.activation_weights_history.append(epoch_weights)
                    
                    # 繪製激活函數權重變化圖 - 减少绘图频率
                    if (i_epoch + 1) % 10 == 0:  # 从5改为10
                        self.plot_activation_weights(i_epoch)
                
                # 清理内存
                torch.cuda.empty_cache()
                gc.collect()
                
                # 每個 epoch 結束後的評估
                eval_results = self.evaluate(dataloader_eval)
                
                print(
                    f"\n-->epoch{i_epoch} end... \nloss_var:{eval_results['loss_var']:.3f} \nloss_nino:{eval_results['loss_nino']:.3f} \nloss_com:{eval_results['combine_loss']:.3f} \nscore: {eval_results['score']:.3f}"
                )
                
                mlflow.log_metric("Epoch/Loss_Var", eval_results['loss_var'], step=i_epoch)
                mlflow.log_metric("Epoch/Loss_Nino", eval_results['loss_nino'], step=i_epoch)
                mlflow.log_metric("Epoch/Combine_Loss", eval_results['combine_loss'], step=i_epoch)
                mlflow.log_metric("Epoch/Score", eval_results['score'], step=i_epoch)
                
                # 記錄激活函數權重
                if 'activation_weights' in eval_results and eval_results['activation_weights'] is not None:
                    for i, weight in enumerate(eval_results['activation_weights']):
                        mlflow.log_metric(f"Epoch/Activation_Weight_{i}", weight, step=i_epoch)

                # 檢查是否需要保存模型
                if eval_results['score'] <= best:
                    count += 1
                    print(f"\nsc is not increase for {count} epoch")
                else:
                    count = 0
                    print(
                        f"\nsc is increase from {best:.3f} to {eval_results['score']:.3f} with dynamic activation \nsaving model...\n"
                    )
                    torch.save(self.mymodel.state_dict(), chk_path)
                    best = eval_results['score']
                
                # 早停檢查
                if count == self.mypara.patience:
                    print(
                        f"\n-----!!!early stopping reached, max(sceval)= {best:.3f} with dynamic activation!!!-----"
                    )
                    break
        
        # 繪製最終的激活函數權重變化圖
        self.plot_activation_weights(num_epochs, is_final=True)
        
        return best
    
    def evaluate(self, dataloader):
        """評估模型"""
        self.mymodel.eval()
        nino_pred = []
        var_pred = []
        nino_true = []
        var_true = []
        activation_weights_list = []
        
        with torch.no_grad():
            for input_var, var_true1 in dataloader:
                # 提取真實 SST 和 Nino 指數
                SST = var_true1[:, :, self.sstlevel]
                nino_true1 = SST[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                
                # 模型預測
                out_var, residual_loss = self.mymodel(
                    input_var.float().to(self.device),
                    predictand=None,
                    train=False,
                )
                
                # 提取預測的 SST 和 Nino 指數
                SST_out = out_var[:, :, self.sstlevel]
                out_nino = SST_out[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                
                # 收集預測和真實值 - 使用CPU存储以减轻GPU内存压力
                var_true.append(var_true1.cpu())
                nino_true.append(nino_true1.cpu())
                var_pred.append(out_var.cpu())
                nino_pred.append(out_nino.cpu())
                
                # 收集激活函數權重
                if hasattr(self.mymodel, 'fusion_activation') and hasattr(self.mymodel.fusion_activation, 'selector'):
                    weights = self.mymodel.fusion_activation.selector.weights_history[-1] if self.mymodel.fusion_activation.selector.weights_history else None
                    if weights is not None:
                        activation_weights_list.append(weights.detach().cpu())
                
                # 每处理几个批次后清理内存
                if len(var_pred) % 5 == 0:
                    torch.cuda.empty_cache()
            
            # 拼接所有批次的結果
            var_pred = torch.cat(var_pred, dim=0)
            nino_pred = torch.cat(nino_pred, dim=0)
            nino_true = torch.cat(nino_true, dim=0)
            var_true = torch.cat(var_true, dim=0)
            
            # 将数据移回GPU进行计算
            var_pred_gpu = var_pred.to(self.device)
            nino_pred_gpu = nino_pred.to(self.device)
            nino_true_gpu = nino_true.to(self.device)
            var_true_gpu = var_true.to(self.device)
            
            # 計算評估指標
            ninosc = self.calscore(nino_pred_gpu, nino_true_gpu)
            loss_var = self.loss_var(var_pred_gpu, var_true_gpu, residual_losses=None).item()
            loss_nino = self.loss_nino(nino_pred_gpu, nino_true_gpu).item()
            combine_loss = self.combine_loss(loss_var, loss_nino)
            
            # 计算完成后释放GPU内存
            del var_pred_gpu, nino_pred_gpu, nino_true_gpu, var_true_gpu
            torch.cuda.empty_cache()
            
            # 計算平均激活函數權重
            activation_weights = None
            if activation_weights_list:
                activation_weights = torch.stack(activation_weights_list).mean(dim=0).numpy()
            
        return {
            'var_pred': var_pred,
            'nino_pred': nino_pred,
            'loss_var': loss_var,
            'loss_nino': loss_nino,
            'combine_loss': combine_loss,
            'score': ninosc,
            'activation_weights': activation_weights
        }
    
    def plot_activation_weights(self, epoch, is_final=False):
        """繪製激活函數權重變化圖"""
        if not self.activation_weights_history:
            return
        
        # 創建保存目錄
        save_dir = os.path.join(self.mypara.model_savepath, 'activation_weights')
        os.makedirs(save_dir, exist_ok=True)
        
        # 獲取激活函數名稱
        activation_names = list(self.activation_configs.keys())
        
        # 繪製權重變化圖
        plt.figure(figsize=(10, 6))
        weights = torch.stack(self.activation_weights_history).numpy()
        for i in range(weights.shape[1]):
            plt.plot(weights[:, i], label=activation_names[i] if i < len(activation_names) else f"Activation {i}")
        
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.title('Activation Function Weights over Training')
        plt.legend()
        plt.grid(True)
        
        # 保存圖片
        if is_final:
            plt.savefig(os.path.join(save_dir, 'final_activation_weights.png'))
        else:
            plt.savefig(os.path.join(save_dir, f'activation_weights_epoch_{epoch}.png'))
        
        plt.close()


def train_dynamic_activation():
    """訓練動態激活函數模型"""
    print(f"\n{'='*60}")
    print(f"開始訓練動態激活函數模型")
    print(f"{'='*60}")
    
    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    print(mypara.__dict__)
    print(f"\nloading pre-train dataset for dynamic activation model...")
    traindataset = make_dataset2(mypara)
    print(traindataset.selectregion())
    
    print(f"\nloading evaluation dataset for dynamic activation model...")
    evaldataset = make_testdataset(mypara, ngroup=100)
    print(evaldataset.selectregion())
    
    # 创建训练器并开始训练
    try:
        trainer = DynamicActivationTrainer(mypara)
        best_score = trainer.train_model(
            dataset_train=traindataset,
            dataset_eval=evaldataset,
            num_epochs=50
        )
        
        print(f"\n動態激活函數模型訓練完成！最佳評分: {best_score:.3f}")
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"\n内存不足错误: {e}")
            print("\n尝试减小批次大小或模型大小后重新运行")
        else:
            print(f"\n运行时错误: {e}")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 确保清理内存
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    train_dynamic_activation()
