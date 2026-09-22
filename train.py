import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from torch.utils.data import DataLoader
from models.b_kan import BayesianKAN
from models.bayesianlstm import BayesianLSTM
from models.lstm import LSTM
from models.kan import KAN
from models.transformer import Transformer
from models.gp import GaussianProcess
from models.mc_drop_transformer import MCDropoutTransformer
from utils.loss import elbo_loss
from utils.visualization import plot_rul_prediction, plot_ecr_curve, plot_feature_contribution
from utils.metrics import get_all_metrics
import argparse
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = "/root/autodl-tmp/fonts/times.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体
plt.rcParams['font.size'] = 20




def evaluate_model_efficiency(model, sample_input, device, num_mc=100):
   
    if sample_input.ndim in [2, 3]:
        single_input = sample_input[0:1].to(device)
    else:
        single_input = sample_input.to(device)

   
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6

   
    flops_m = "N/A"
    flops_str = "N/A"
    
 ---------------------------------------------
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            flop_analysis = FlopCountAnalysis(model, single_input)
           
            flop_analysis.unsupported_ops_warnings(False)
            flop_analysis.uncalled_modules_warnings(False)
            flops = flop_analysis.total()
            
        flops_m = flops / 1e6
       
        flops_str = f"{flops_m:.4f} M ({flops/1e3:.2f} K)" if flops_m < 0.01 else f"{flops_m:.4f} M"

    except ImportError:
      
        try:
            from ptflops import get_model_complexity_info
            model.eval()
            # ptflops 需要传入不含 batch_size 的 shape tuple
            input_shape = tuple(single_input.shape[1:])
            
            # ptflops 统计的是 MACs (Multiply-Accumulate Operations)
            # 标准转换规则：1 MAC ≈ 2 FLOPs
            macs, _ = get_model_complexity_info(
                model, input_shape, 
                as_strings=False, 
                print_per_layer_stat=False, 
                verbose=False
            )
            flops = macs * 2
            flops_m = flops / 1e6
            flops_str = f"{flops_m:.4f} M ({flops/1e3:.2f} K)" if flops_m < 0.01 else f"{flops_m:.4f} M"

        except Exception as e:
            print(f"[Warning] FLOPs calculation skipped (Please `pip install fvcore` or `ptflops`): {e}")

   
    model.train()  # 激活概率模型的 MC Sampling 模式
    
    # GPU 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(single_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_mc):
            _ = model(single_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0

  
    print("\n" + "-"*40)
    print("Probabilistic Model Computational Efficiency:")
    print(f" -> Params        : {params_m:.4f} M ({total_params} Params)")
    print(f" -> FLOPs (1 Pass): {flops_str}")
    print(f" -> Latency(100MC): {latency_ms:.2f} ms")
    print("-" * 40 + "\n")

    return params_m, flops_m, latency_ms
def plot_uncertainty_decomposition(aleatoric, epistemic, save_path):
   
    
    plt.figure(figsize=(8, 5.5))  # 微调比例更接近原图
    x = epistemic.flatten()
    y = aleatoric.flatten()
   
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
   
    plt.scatter(x, y, color='#4c72b0', s=35, alpha=0.5, edgecolors='none', zorder=3)
    
    plt.axvline(x=mean_x, color='#c44e52', linestyle=':', linewidth=1.5, 
                label=f'Mean Epi: {mean_x:.4f}', zorder=4)
    plt.axhline(y=mean_y,color='#555555', linestyle=':', linewidth=1.5, 
                label=f'Mean Alea: {mean_y:.4f}', zorder=4)
    plt.xlabel('Epistemic Uncertainty')
    plt.ylabel('Aleatoric Uncertainty')
    plt.legend(loc='best', frameon=True, edgecolor='none', facecolor='white', framealpha=0.8)
    
    plt.grid(True, linestyle='--', alpha=0.6, zorder=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def train_one_epoch(model, loader, optimizer, device, kl_weight, criterion=None):
    model.train()
    total_loss = 0
    num_batches = len(loader)
    if criterion is None: criterion = nn.MSELoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        if isinstance(output, tuple):
            mu, logvar, kl = output
            current_kl_weight = kl_weight / num_batches
            loss, nll = elbo_loss(mu, y, logvar, kl, current_kl_weight)
        else: 
            loss = criterion(output.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_batches

def validate_model(model, loader, device, kl_weight, criterion=None):
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)

    if criterion is None:
        criterion = nn.MSELoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            if isinstance(output, tuple):
                mu, logvar, kl = output

                # 与训练阶段保持一致
                current_kl_weight = kl_weight / num_batches

                loss, _ = elbo_loss(
                    mu, y, logvar, kl,
                    current_kl_weight
                )
            else:
                loss = criterion(
                    output.squeeze(),
                    y.squeeze()
                )

            total_loss += loss.item()

    return total_loss / num_batches
def evaluate_model(model, loader, device):
    model.eval()
    dummy_x, _ = next(iter(loader))
    dummy_x = dummy_x.to(device)
    with torch.no_grad():
        dummy_out = model(dummy_x)
    
    is_probabilistic = isinstance(dummy_out, tuple) and len(dummy_out) == 3
    if is_probabilistic:
        model.train() 
        preds_mu, preds_std, truths = [], [], []
        preds_aleatoric, preds_epistemic = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                batch_mus = []
                batch_aleatorics = []
                for _ in range(100): 
                    mu, logvar, _ = model(x)
                    batch_mus.append(mu.cpu().numpy())
                    batch_aleatorics.append(torch.exp(logvar).cpu().numpy())
                
                batch_mus = np.array(batch_mus)
                batch_aleatorics = np.array(batch_aleatorics)
                
                mean = batch_mus.mean(axis=0)
                epistemic_var = batch_mus.var(axis=0)
                aleatoric_var = batch_aleatorics.mean(axis=0)
                total_std = np.sqrt(epistemic_var + aleatoric_var)
                
                preds_mu.append(mean)
                preds_std.append(total_std)
                preds_aleatoric.append(aleatoric_var)
                preds_epistemic.append(epistemic_var)
                truths.append(y.numpy())
                
        return (np.concatenate(preds_mu), np.concatenate(preds_std), np.concatenate(truths),
                np.concatenate(preds_aleatoric), np.concatenate(preds_epistemic))
    else:
        preds, truths = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                out = model(x) 
                preds.append(out.cpu().numpy())
                truths.append(y.numpy())
        p = np.concatenate(preds)
        zeros = np.zeros_like(p)
        return p, zeros, np.concatenate(truths), zeros, zeros
def evaluate_single_bearing(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    dummy_x, _ = next(iter(loader))
    dummy_x = dummy_x.to(device)
    model.eval()
    with torch.no_grad():
        dummy_out = model(dummy_x)
    is_probabilistic = isinstance(dummy_out, tuple) and len(dummy_out) >= 2
    if is_probabilistic:
        model.train() 
    else:
        model.eval()  
    
    preds_mu, preds_std, truths = [], [], []
    preds_aleatoric, preds_epistemic = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            truths.append(y.numpy()) 
            
            if is_probabilistic:
                batch_mus = []
                batch_aleatorics = []
                for _ in range(100): 
                    mu, logvar, _ = model(x) 
                    batch_mus.append(mu.cpu().numpy())
                    batch_aleatorics.append(torch.exp(logvar).cpu().numpy()) 
                
                batch_mus = np.array(batch_mus) 
                batch_aleatorics = np.array(batch_aleatorics)
                
                mean = batch_mus.mean(axis=0)
                epistemic_var = batch_mus.var(axis=0)    
                aleatoric_var = batch_aleatorics.mean(axis=0) 
                
                total_variance = epistemic_var + aleatoric_var
                total_std = np.sqrt(total_variance)    
                
                preds_mu.append(mean)
                preds_std.append(total_std)
                preds_aleatoric.append(aleatoric_var)
                preds_epistemic.append(epistemic_var)
            else:
                out = model(x) 
                pred = out.cpu().numpy()
                preds_mu.append(pred)
                preds_std.append(np.zeros_like(pred))
                preds_aleatoric.append(np.zeros_like(pred))
                preds_epistemic.append(np.zeros_like(pred))

    return (np.concatenate(preds_mu), np.concatenate(preds_std), np.concatenate(truths), 
            np.concatenate(preds_aleatoric), np.concatenate(preds_epistemic))
def plot_training_history(
        train_losses,
        val_losses,
        best_epoch,
        save_path):

    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(
        epochs,
        train_losses,
        color="#539AF8",
        linewidth=2.2,
        label="Training loss"
    )

    ax.plot(
        epochs,
        val_losses,
        color="#73C986",
        linewidth=2.2,
        label="Validation loss"
    )

    ax.axvline(
        best_epoch,
        color="#E76F51",
        linestyle="--",
        linewidth=1.8,
        label=f"Selected epoch = {best_epoch}"
    )

    ax.scatter(
        best_epoch,
        val_losses[best_epoch - 1],
        color="#E76F51",
        s=65,
        zorder=5
    )

    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)

    ax.tick_params(
        axis="both",
        labelsize=17
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.5
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        fontsize=15,
        frameon=True,
        edgecolor="none"
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(fig)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--model', type=str, default='b-kan', choices=['b-kan', 'kan','b-lstm', 'lstm','transformer','gp','mc-transformer'])
    parser.add_argument('--loss', type=str, default='mse')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    loss_suffix = "elbo" if args.model in ['b-kan', 'b-lstm','gp','mc-transformer'] else args.loss
    run_id = f"{args.model}_{loss_suffix}"
    
    model_save_path = os.path.join(args.result_dir, f"best_weight_{run_id}.pth")
    csv_save_path = os.path.join(args.result_dir, f"metrics_{run_id}.csv")
    image_save_dir = os.path.join(args.result_dir, f"{run_id}_figures")
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir, exist_ok=True)
        
    DATA_DIR = './data/XJTU-SY'
    ALL_CONDITIONS = ['35Hz12kN']
    TRAIN_IDS = [1]
    VAL_IDS = [1] 
    selected_features = None 

    print("\n" + "="*50)
    print("Phase 1: Mixed Training (Raw Features)")
    print("="*50)

    my_fpt_dict = {('37.5Hz11kN', 3): 0}
    
    try:
        train_loader, val_loader, scaler = get_xjtu_dataloader(
            DATA_DIR, ALL_CONDITIONS, TRAIN_IDS, VAL_IDS, batch_size=1, feature_select=selected_features, fpt_dict=my_fpt_dict
        )
        feature_names = train_loader.dataset.feature_keys
        input_dim = len(feature_names)
        print(f"[Config] Input Dim: {input_dim}, Features: {feature_names}")
    except Exception as e:
        print(f"Fatal Error loading data: {e}")
        return
        
    if args.model == 'b-kan':
        model = BayesianKAN(input_dim=input_dim, hidden_dims=[64, 32]).to(device)
    elif args.model == 'kan':
        model = KAN(input_dim=input_dim, hidden_dims=[32, 16], output_dim=1).to(device)
    elif args.model == 'b-lstm':
        model = BayesianLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2).to(device)
    elif args.model == 'lstm':
        model = LSTM(input_dim=input_dim, hidden_dim=16, num_layers=2, output_dim=1).to(device)
    elif args.model == 'transformer':
        model = Transformer(input_dim=input_dim, output_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.1).to(device)
    elif args.model == 'gp':
        model = GaussianProcess(input_dim=input_dim, num_inducing=50).to(device)
    elif args.model == 'mc-transformer':
        model = MCDropoutTransformer(input_dim=input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2, output_dim=1).to(device)


    
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse_criterion = nn.MSELoss()
    
    print("\nStart Training...")

    best_loss = float("inf")
    best_epoch = 0
    
    train_history = []
    val_history = []
    
    patience = 10
    early_stop_counter = 0
    min_delta = 1e-6
    
    for epoch in range(args.epochs):
    
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            kl_weight=0.1,
            criterion=mse_criterion
        )
    
        val_loss = validate_model(
            model,
            val_loader,
            device,
            kl_weight=0.1,
            criterion=mse_criterion
        )
    
        train_history.append(train_loss)
        val_history.append(val_loss)
    
        improved = val_loss < best_loss - min_delta
    
        if improved:
            best_loss = val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
    
            torch.save(
                model.state_dict(),
                model_save_path
            )
        else:
            early_stop_counter += 1
    
        print(
            f"Epoch {epoch + 1:3d}: "
            f"Train={train_loss:.6f} | "
            f"Val={val_loss:.6f}"
            + (" [Saved Best]" if improved else "")
        )
    
        if early_stop_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch}"
            )
            break
    
    print(
        f"Training Finished. "
        f"Best epoch: {best_epoch}, "
        f"Best validation loss: {best_loss:.6f}"
    )
    
    loss_history_path = os.path.join(
        args.result_dir,
        f"loss_history_{run_id}.csv"
    )
    
    pd.DataFrame({
        "Epoch": np.arange(1, len(train_history) + 1),
        "TrainingLoss": train_history,
        "ValidationLoss": val_history
    }).to_csv(loss_history_path, index=False)
    
    
    print(f"Loss data saved to: {loss_history_path}")
    print(f"Loss curve saved to: {loss_figure_path}")

    print("\n" + "="*50)
    print("Phase 2: Individual Evaluation for ALL Bearings")
    print("="*50)
    
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    else:
        print(f"[Error] Model file {model_save_path} not found! Skipping evaluation.")
        return
    dummy_sample_x, _ = next(iter(train_loader))
    params_m, flops_m, latency_ms = evaluate_model_efficiency(model, dummy_sample_x, device, num_mc=100)
    
    print("\n" + "-"*40)
    print("Probabilistic Model Computational Efficiency:")
    print(f" -> Params        : {params_m:.4f} M" if isinstance(params_m, float) else f" -> Params        : {params_m}")
    print(f" -> FLOPs (1 Pass): {flops_m:.4f} M" if isinstance(flops_m, float) else f" -> FLOPs (1 Pass): {flops_m}")
    print(f" -> Latency(100MC): {latency_ms:.2f} ms")
    print("-" * 40 + "\n")

    results_list = []
    all_bearing_ids = [3]
    
    for cond in ALL_CONDITIONS:
        for bid in all_bearing_ids:
            role = "Train" if bid in TRAIN_IDS else "Test"
            print(f"Testing: {cond} - Bearing {bid} ({role})...", end="")
            
            try:
                dataset = XJTU_SY_Dataset(DATA_DIR, cond, [bid], scaler=scaler, is_train=False, feature_select=selected_features)
                total_life_actual = len(dataset)
                data_loader = DataLoader(dataset, batch_size=200, shuffle=False)
                if len(dataset) == 0:
                    print(" [Skipped: No Data Found]")
                    continue
                
                mu, std, y_true, aleatoric, epistemic = evaluate_single_bearing(model, dataset, device)
                
                is_deterministic = (np.sum(std) == 0)
                std_for_metrics = None if is_deterministic else std.flatten()
                metrics = get_all_metrics(y_true.flatten(), mu.flatten(), std_for_metrics)
                
                res = {
                    "Condition": cond, "Bearing": bid, "Role": role,
                    "RMSE": round(metrics['RMSE'], 4), "MAE": round(metrics['MAE'], 4), "R2": round(metrics['R2'], 4)
                }
                if 'NLL' in metrics:
                    res["NLL"], res["PICP"], res["MPIW"] = round(metrics['NLL'], 4), round(metrics['PICP'], 4), round(metrics['MPIW'], 4)
                else:
                    res["NLL"], res["PICP"], res["MPIW"] = "-", "-", "-"
                results_list.append(res)
                
                plot_rulname = os.path.join(image_save_dir, f"rul_{run_id}_{cond}_b{bid}.jpg")
                plot_ecrname = os.path.join(image_save_dir, f"ecr_{run_id}_{cond}_b{bid}.jpg")
                plot_intername = os.path.join(image_save_dir, f"inter_{run_id}_{cond}_b{bid}.jpg")
                plot_uncertaintyname = os.path.join(image_save_dir, f"uncertainty_decomp_{run_id}_{cond}_b{bid}.jpg")
                
            
                plot_rul_prediction(y_true.flatten(), mu.flatten(), std.flatten(), save_path=plot_rulname)
                
                if not is_deterministic:
                    plot_uncertainty_decomposition(aleatoric.flatten(), epistemic.flatten(), save_path=plot_uncertaintyname)
                    print(f" -> Saved {os.path.basename(plot_uncertaintyname)}")
                    
                    idx = -50
                    if len(mu) > 50:
                        total_life = len(dataset)
                        rul_mean_real = mu[idx].item() * total_life
                        rul_std_real = std[idx].item() * total_life
                        plot_ecr_curve(rul_mean_real, rul_std_real, nominal_life=total_life_actual, Cp=100, Cf=500, save_path=plot_ecrname)
                        print(" -> Saved result_ecr.jpg")
                        
                x_sample, _ = next(iter(data_loader))
                x_sample = x_sample.to(device)
                model.eval()
                contrib = model.get_feature_importance(x_sample).cpu().detach().numpy()
                time_steps = np.arange(len(contrib))
                plot_feature_contribution(time_steps, contrib, feature_names, save_path=plot_intername)
                print(" -> Saved result_interpretation.jpg")
                
                log_msg = f" Done. RMSE={metrics['RMSE']:.4f}"
                if 'NLL' in metrics: log_msg += f", NLL={metrics['NLL']:.4f}"
                print(log_msg)
                
            except Exception as e:
                print(f" [Error: {e}]")
                results_list.append({"Condition": cond, "Bearing": bid, "Role": role, "RMSE": "Error", "NLL": str(e)})
    
    if len(results_list) > 0:
        df = pd.DataFrame(results_list)
        cols = ["Condition", "Bearing", "Role", "RMSE", "R2", "CRPS", "PICP", "MPIW"]
        existing_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        df = df[existing_cols]
        df.to_csv(csv_save_path, index=False)
        print(f"\n[Success] Detailed results saved to {csv_save_path}")
    else:
        print("\n[Warning] No results to save.")
    print("\nAll Done.")

if __name__ == "__main__":
    main()