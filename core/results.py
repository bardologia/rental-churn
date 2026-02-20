import sys
import os
from pathlib import Path
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import median_absolute_error
from scipy.stats import gaussian_kde


class Results:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model.to(self.device)
        self.target_scaler = model.target_scaler
    
    def evaluate(self, model, loader, target_scaler):
        all_preds, all_targets, all_lengths = [], [], []
        model.eval()
        with torch.no_grad():
            for x_cat, x_cont, y, lengths in loader:
                x_cat   = x_cat.to(self.device)
                x_cont  = x_cont.to(self.device)
                lengths = lengths.to(self.device)
                
                preds = model(x_cat, x_cont, lengths)
                
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
                all_lengths.extend(lengths.cpu().numpy())

        preds_tensor   = torch.cat(all_preds)
        targets_tensor = torch.cat(all_targets)
        
        seq_lengths = np.array(all_lengths)
        den_targets = np.expm1(target_scaler.inverse_transform(targets_tensor.reshape(-1, 1)))
        den_preds   = np.expm1(target_scaler.inverse_transform(preds_tensor.reshape(-1, 1)))
        
        den_preds   = np.clip(den_preds, 0, None)
        den_targets = np.clip(den_targets, 0, None)
        
        df_preds = pd.DataFrame({
            'pred': den_preds.flatten(), 
            'target': den_targets.flatten(),
            'error': (den_preds - den_targets).flatten(),
            'abs_error': np.abs(den_preds - den_targets).flatten(),
            'seq_length': seq_lengths
        })
        return den_preds, den_targets, seq_lengths, df_preds

    @staticmethod
    def compute_metrics(den_preds: np.ndarray, den_targets: np.ndarray, seq_lengths: np.ndarray = None) -> dict:
        den_preds   = np.asarray(den_preds).flatten()
        den_targets = np.asarray(den_targets).flatten()
        
        mae = float(np.mean(np.abs(den_preds - den_targets)))
        rmse = float(np.sqrt(np.mean((den_preds - den_targets) ** 2)))
        ss_res = float(np.sum((den_targets - den_preds) ** 2))
        ss_tot = float(np.sum((den_targets - np.mean(den_targets)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        std = float(np.std(den_targets - den_preds))
        medae = float(median_absolute_error(den_targets, den_preds))
        max_error = float(np.max(np.abs(den_targets - den_preds)))
        p50 = np.percentile(np.abs(den_targets - den_preds), 50)
        p90 = np.percentile(np.abs(den_targets - den_preds), 90)
        p95 = np.percentile(np.abs(den_targets - den_preds), 95)

        abs_err = np.abs(den_targets - den_preds)
        signed_err = den_preds - den_targets
        
        error_bin_0_5 = float(np.mean(abs_err <= 5) * 100)
        error_bin_5_10 = float(np.mean((abs_err > 5) & (abs_err <= 10)) * 100)
        error_bin_10_15 = float(np.mean((abs_err > 10) & (abs_err <= 15)) * 100)
        error_bin_15_20 = float(np.mean((abs_err > 15) & (abs_err <= 20)) * 100)
        error_bin_20_25 = float(np.mean((abs_err > 20) & (abs_err <= 25)) * 100)
        error_bin_above_25 = float(np.mean(abs_err > 25) * 100)

        def mean_target_range(low, high=None):
            mask = (den_targets > low) & (den_targets <= high)
            vals = abs_err[mask]
            return float(np.mean(vals)) if len(vals) > 0 else float('nan')

     
        bias = float(np.mean(signed_err))  
        bias_pct = float((bias / np.mean(den_targets) * 100)) if np.mean(den_targets) != 0 else 0
        
        mape = float(np.mean(np.abs((den_targets - den_preds) / (den_targets + 1e-8))) * 100)
        
        is_delayed = den_targets > 0
        predicted_delayed = den_preds > 0
        
        tp = np.sum((predicted_delayed) & (is_delayed))
        tn = np.sum((~predicted_delayed) & (~is_delayed))
        fp = np.sum((predicted_delayed) & (~is_delayed))
        fn = np.sum((~predicted_delayed) & (is_delayed))
        
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0 
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0 
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0    
        f1_score = float(2 * (precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0
        
        q1_targets = np.percentile(den_targets, 25)
        q3_targets = np.percentile(den_targets, 75)
        mae_q1 = float(np.mean(abs_err[den_targets <= q1_targets])) if np.sum(den_targets <= q1_targets) > 0 else np.nan
        mae_mid = float(np.mean(abs_err[(den_targets > q1_targets) & (den_targets < q3_targets)])) if np.sum((den_targets > q1_targets) & (den_targets < q3_targets)) > 0 else np.nan
        mae_q3 = float(np.mean(abs_err[den_targets >= q3_targets])) if np.sum(den_targets >= q3_targets) > 0 else np.nan
        
        underestimate_count = float(np.mean((den_preds < den_targets)) * 100)
        overestimate_count = float(np.mean((den_preds > den_targets)) * 100)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'std': std,
            'p50': p50,
            'p90': p90,
            'p95': p95,
            'medae': medae,
            'max_error': max_error,
            'mape': mape,
            'bias': bias,
            'bias_pct': bias_pct,
            
            'error_0_5': error_bin_0_5,
            'error_5_10': error_bin_5_10,
            'error_10_15': error_bin_10_15,
            'error_15_20': error_bin_15_20,
            'error_20_25': error_bin_20_25,
            'error_above_25': error_bin_above_25,
    
            'target_0_5': mean_target_range(0, 5),
            'target_5_10': mean_target_range(5, 10),
            'target_10_15': mean_target_range(10, 15),
            'target_15_20': mean_target_range(15, 20),
            'target_20_25': mean_target_range(20, 25),
            'target_above_25': mean_target_range(25, 30),
            
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            
            'mae_q1_low': mae_q1,
            'mae_mid': mae_mid,
            'mae_q3_high': mae_q3,
            
            'underestimate_pct': underestimate_count,
            'overestimate_pct': overestimate_count,
        }
        
        return metrics

    @staticmethod
    def _make_plots(den_preds: np.ndarray, den_targets: np.ndarray, seq_lengths: np.ndarray) -> dict:
        errors = np.ravel(den_preds - den_targets)
        figs = {}

        fig = plt.figure(figsize=(8, 5))
        ax = fig.gca()
        ax.hist(errors, bins='auto', color='skyblue', edgecolor='black')
        ax.set_title('Histogram of Prediction Errors (Residuals)')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        figs['errors_hist'] = fig

        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(seq_lengths, errors, alpha=0.5, color='royalblue', edgecolor='k')
        axes[0].set_title('Error vs. Sequence Length')
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Prediction Error')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        axes[1].scatter(den_targets, errors, alpha=0.5, color='darkorange', edgecolor='k')
        axes[1].set_title('Error vs. Target Value')
        axes[1].set_xlabel('True Target Value')
        axes[1].set_ylabel('Prediction Error')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        fig2.tight_layout()
        figs['error_vs_length_and_target'] = fig2

        fig3 = plt.figure(figsize=(7, 6))
        ax3 = fig3.gca()
        ax3.scatter(den_targets, den_preds, alpha=0.5, color='mediumseagreen', edgecolor='k')
        ax3.plot([den_targets.min(), den_targets.max()], [den_targets.min(), den_targets.max()], 'r--', lw=2, label='Ideal Fit')
        ax3.set_title('Predicted vs. True Values')
        ax3.set_xlabel('True Target Value')
        ax3.set_ylabel('Predicted Value')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)
        fig3.tight_layout()
        figs['pred_vs_true'] = fig3

        fig4 = plt.figure(figsize=(7, 5))
        ax4 = fig4.gca()
        ax4.scatter(den_preds, errors, alpha=0.5, color='slateblue', edgecolor='k')
        ax4.axhline(0, color='red', linestyle='--', lw=2)
        ax4.set_title('Residuals vs. Predicted Values')
        ax4.set_xlabel('Predicted Value')
        ax4.set_ylabel('Residual (Error)')
        ax4.grid(True, linestyle='--', alpha=0.6)
        fig4.tight_layout()
        figs['residuals_vs_predicted'] = fig4

        fig5 = plt.figure(figsize=(6, 6))
        ax5 = fig5.gca()
        stats.probplot(errors, dist="norm", plot=ax5)
        ax5.set_title('QQ-plot of Residuals')
        ax5.grid(True, linestyle='--', alpha=0.6)
        fig5.tight_layout()
        figs['qqplot_residuals'] = fig5

        sorted_abs_errors = np.sort(np.abs(errors))
        cum_abs_error = np.cumsum(sorted_abs_errors)
        fig6 = plt.figure(figsize=(8, 5))
        ax6 = fig6.gca()
        ax6.plot(np.arange(1, len(cum_abs_error) + 1), cum_abs_error, color='teal')
        ax6.set_title('Cumulative Absolute Error')
        ax6.set_xlabel('Sample (sorted by error)')
        ax6.set_ylabel('Cumulative Absolute Error')
        ax6.grid(True, linestyle='--', alpha=0.6)
        fig6.tight_layout()
        figs['cumulative_abs_error'] = fig6


        fig7 = plt.figure(figsize=(8, 5))
        ax7 = fig7.gca()
        underestimate = errors[errors < 0]
        overestimate = errors[errors > 0]
        ax7.hist([underestimate, overestimate], bins='auto', label=['Underestimate', 'Overestimate'], 
                 color=['#ff7f0e', '#1f77b4'], alpha=0.7, edgecolor='black')
        ax7.set_title('Distribution of Under/Overestimation')
        ax7.set_xlabel('Prediction Error')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        ax7.grid(True, linestyle='--', alpha=0.6)
        fig7.tight_layout()
        figs['bias_analysis'] = fig7

        fig8 = plt.figure(figsize=(10, 5))
        ax8 = fig8.gca()
        ax8.scatter(den_targets, np.abs(errors), alpha=0.4, c=den_preds, cmap='viridis', edgecolor='k', s=20)
        cbar = plt.colorbar(ax8.collections[0], ax=ax8)
        cbar.set_label('Predicted Value')
        ax8.set_title('Absolute Error vs True Target Value (colored by prediction)')
        ax8.set_xlabel('True Target (days)')
        ax8.set_ylabel('Absolute Error (days)')
        ax8.grid(True, linestyle='--', alpha=0.6)
        fig8.tight_layout()
        figs['error_by_magnitude'] = fig8

        n_bins = 10
        bin_edges = np.linspace(0, max(den_targets.max(), den_preds.max()), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_preds_per_bin = []
        mean_targets_per_bin = []
        counts = []
        
        for i in range(len(bin_edges) - 1):
            mask = (den_preds >= bin_edges[i]) & (den_preds < bin_edges[i+1])
            if np.sum(mask) > 0:
                mean_preds_per_bin.append(np.mean(den_preds[mask]))
                mean_targets_per_bin.append(np.mean(den_targets[mask]))
                counts.append(np.sum(mask))
        
        fig9 = plt.figure(figsize=(8, 6))
        ax9 = fig9.gca()
        ax9.scatter(mean_preds_per_bin, mean_targets_per_bin, s=np.array(counts)/5, alpha=0.6, edgecolor='k')
        min_val = min(min(mean_preds_per_bin) if mean_preds_per_bin else 0, 
                      min(mean_targets_per_bin) if mean_targets_per_bin else 0)
        max_val = max(max(mean_preds_per_bin) if mean_preds_per_bin else 1, 
                      max(mean_targets_per_bin) if mean_targets_per_bin else 1)
        ax9.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Calibration')
        ax9.set_title('Calibration Plot: Mean Predictions vs Mean Actuals (bubble size = sample count)')
        ax9.set_xlabel('Mean Predicted Value')
        ax9.set_ylabel('Mean Actual Value')
        ax9.legend()
        ax9.grid(True, linestyle='--', alpha=0.6)
        fig9.tight_layout()
        figs['calibration_curve'] = fig9

        threshold = 0
        is_delayed = den_targets > threshold
        predicted_delayed = den_preds > threshold
        tp = np.sum((predicted_delayed) & (is_delayed))
        tn = np.sum((~predicted_delayed) & (~is_delayed))
        fp = np.sum((predicted_delayed) & (~is_delayed))
        fn = np.sum((~predicted_delayed) & (is_delayed))
        
        fig10, ax10 = plt.subplots(figsize=(8, 6))
        confusion = np.array([[tn, fp], [fn, tp]])
        im = ax10.imshow(confusion, cmap='Blues', aspect='auto')
        ax10.set_xticks([0, 1])
        ax10.set_yticks([0, 1])
        ax10.set_xticklabels(['No Delay (Pred)', 'Delay (Pred)'])
        ax10.set_yticklabels(['No Delay (True)', 'Delay (True)'])
        ax10.set_ylabel('True Label')
        ax10.set_xlabel('Predicted Label')
        ax10.set_title(f'Confusion Matrix (Threshold = {threshold} days)\nSensitivity={tp/(tp+fn):.2f}, Specificity={tn/(tn+fp):.2f}')
        
        for i in range(2):
            for j in range(2):
                text = ax10.text(j, i, confusion[i, j], ha="center", va="center", color="w" if confusion[i, j] > confusion.max()/2 else "k", fontsize=14, weight='bold')
        fig10.tight_layout()
        figs['confusion_matrix'] = fig10

        fig11 = plt.figure(figsize=(10, 5))
        ax11 = fig11.gca()
        length_bins = pd.cut(seq_lengths, bins=5)
        bp_data = [np.abs(errors)[length_bins == bin_val] for bin_val in length_bins.unique()]
        ax11.boxplot(bp_data, labels=[f'{i.left:.0f}-{i.right:.0f}' for i in sorted(length_bins.unique())])
        ax11.set_title('Absolute Error Distribution by Sequence Length Bins')
        ax11.set_xlabel('Sequence Length Range')
        ax11.set_ylabel('Absolute Error (days)')
        ax11.grid(True, linestyle='--', alpha=0.6)
        fig11.tight_layout()
        figs['error_by_seq_length_boxplot'] = fig11

        fig12 = plt.figure(figsize=(8, 5))
        ax12 = fig12.gca()
        ax12.scatter(den_preds, np.abs(errors), alpha=0.5, c=np.abs(den_preds - den_targets), cmap='Reds', edgecolor='k')
        cbar2 = plt.colorbar(ax12.collections[0], ax=ax12)
        cbar2.set_label('Absolute Error')
        ax12.set_title('Predicted Value vs Absolute Error')
        ax12.set_xlabel('Predicted Value (days)')
        ax12.set_ylabel('Absolute Error (days)')
        ax12.grid(True, linestyle='--', alpha=0.6)
        fig12.tight_layout()
        figs['prediction_confidence'] = fig12

        fig13 = plt.figure(figsize=(10, 6))
        ax13 = fig13.gca()
        percentiles = np.arange(0, 101, 5)
        errors_by_percentile = [np.percentile(np.abs(errors), p) for p in percentiles]
        ax13.plot(percentiles, errors_by_percentile, marker='o', linewidth=2, markersize=6, color='darkgreen')
        ax13.fill_between(percentiles, 0, errors_by_percentile, alpha=0.3, color='green')
        ax13.set_title('Error Percentile Curve')
        ax13.set_xlabel('Percentile')
        ax13.set_ylabel('Absolute Error (days)')
        ax13.grid(True, linestyle='--', alpha=0.6)
        fig13.tight_layout()
        figs['error_percentile_curve'] = fig13

        fig14, axes14 = plt.subplots(1, 2, figsize=(14, 5))
        axes14[0].hist(den_targets, bins=50, alpha=0.7, color='blue', edgecolor='black', label='True')
        axes14[0].hist(den_preds, bins=50, alpha=0.7, color='red', edgecolor='black', label='Predicted')
        axes14[0].set_title('Distribution of True vs Predicted Values')
        axes14[0].set_xlabel('Days to Payment')
        axes14[0].set_ylabel('Frequency')
        axes14[0].legend()
        axes14[0].grid(True, linestyle='--', alpha=0.6)
        
        kde_targets = gaussian_kde(den_targets.flatten())
        kde_preds = gaussian_kde(den_preds.flatten())
        x_range = np.linspace(min(den_targets.min(), den_preds.min()), max(den_targets.max(), den_preds.max()), 200)
        axes14[1].plot(x_range, kde_targets(x_range), label='True (KDE)', linewidth=2, color='blue')
        axes14[1].plot(x_range, kde_preds(x_range), label='Predicted (KDE)', linewidth=2, color='red')
        axes14[1].set_title('Kernel Density Estimation')
        axes14[1].set_xlabel('Days to Payment')
        axes14[1].set_ylabel('Density')
        axes14[1].legend()
        axes14[1].grid(True, linestyle='--', alpha=0.6)
        fig14.tight_layout()
        figs['distribution_comparison'] = fig14

        return figs

    def run(self, loader, make_plots: bool = False):
        den_preds, den_targets, seq_lengths, df_preds = self.evaluate(self.model, loader, self.target_scaler)
        metrics = self.compute_metrics(den_preds, den_targets, seq_lengths)

        plots = {}
        if make_plots:
            plots = self._make_plots(den_preds, den_targets, seq_lengths)

        return {
            'metrics': metrics,
            'preds_df': df_preds,
            'plots': plots
        }


