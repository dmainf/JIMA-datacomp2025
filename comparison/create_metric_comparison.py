import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11

base_path = '.'
tft_df = pd.read_csv(f'{base_path}/tft/evaluation_all.csv')
chronos_df = pd.read_csv(f'{base_path}/chronos_t5/evaluation_all.csv')
chronos_ft_df = pd.read_csv(f'{base_path}/chronos_t5+FT/evaluation_all.csv')
chronos_bolt_df = pd.read_csv(f'{base_path}/chronos_bolt/evaluation_all.csv')
chronos_bolt_ft_df = pd.read_csv(f'{base_path}/chronos_bolt+FT/evaluation_all.csv')

output_dir = 'metric_comparison'
os.makedirs(output_dir, exist_ok=True)

raw_metrics = ['MAE', 'RMSE', 'wQL_0.1', 'wQL_0.5', 'wQL_0.9', 'wQL_Mean', 'Coverage_80%']

label_map = {
    'MAE': r'MAE',
    'RMSE': r'RMSE',
    'wQL_0.1': r'$QL_{0.1}$',
    'wQL_0.5': r'$QL_{0.5}$',
    'wQL_0.9': r'$QL_{0.9}$',
    'wQL_Mean': r'$QL_{mean}$',
    'Coverage_80%': r'Cov ($80\%$)'
}

metric_direction = {
    'MAE': 'lower',
    'RMSE': 'lower',
    'wQL_0.1': 'lower',
    'wQL_0.5': 'lower',
    'wQL_0.9': 'lower',
    'wQL_Mean': 'lower',
    'Coverage_80%': 'higher'
}

tft_means = [tft_df[metric].mean() for metric in raw_metrics]
chronos_means = [chronos_df[metric].mean() for metric in raw_metrics]
chronos_ft_means = [chronos_ft_df[metric].mean() for metric in raw_metrics]
chronos_bolt_means = [chronos_bolt_df[metric].mean() for metric in raw_metrics]
chronos_bolt_ft_means = [chronos_bolt_ft_df[metric].mean() for metric in raw_metrics]

models = ['TFT', 'Chronos', 'Chronos+FT', 'Chronos-Bolt', 'Chronos-Bolt+FT']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
markers = ['o', '^', 's', 'p', 'D']

for i, metric in enumerate(raw_metrics):
    fig, ax = plt.subplots(figsize=(10, 6))

    values = [tft_means[i], chronos_means[i], chronos_ft_means[i], chronos_bolt_means[i], chronos_bolt_ft_means[i]]
    x = np.arange(len(models))

    bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)
    for j, (xj, val, marker, color) in enumerate(zip(x, values, markers, colors)):
        ax.plot(xj, val, marker=marker, color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=1.5)

    if metric == 'Coverage_80%':
        diffs = [abs(v - 0.8) for v in values]
        best_idx = np.argmin(diffs)
    else:
        direction = metric_direction[metric]
        if direction == 'lower':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

    y_range = max(values) - min(values)
    for idx, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        y_offset = height + y_range * 0.05  # 棒の上から5%上に配置
        text_str = f'{value:.4f}' if value < 1000 else f'{value:.2e}'
        ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                text_str, ha='center', va='bottom', fontsize=10)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + y_range * 0.15)

    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel(label_map[metric], fontsize=13, fontweight='bold')
    ax.set_title(f'{label_map[metric]} Comparison', fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_idx:
            label.set_fontweight('bold')

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    filename = f'{metric}'
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/{filename}.png")

print(f"\nAll plots saved in '{output_dir}/' directory")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for i, metric in enumerate(raw_metrics):
    ax = axes[i]

    values = [tft_means[i], chronos_means[i], chronos_ft_means[i], chronos_bolt_means[i], chronos_bolt_ft_means[i]]
    x = np.arange(len(models))

    bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)

    for j, (xj, val, marker, color) in enumerate(zip(x, values, markers, colors)):
        ax.plot(xj, val, marker=marker, color=color, markersize=10,
                markeredgecolor='black', markeredgewidth=1.5)

    if metric == 'Coverage_80%':
        diffs = [abs(v - 0.8) for v in values]
        best_idx = np.argmin(diffs)
    else:
        direction = metric_direction[metric]
        if direction == 'lower':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

    y_range = max(values) - min(values) if max(values) != min(values) else max(values)
    for idx, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        y_offset = height + y_range * 0.05
        text_str = f'{value:.4f}' if value < 1000 else f'{value:.2e}'
        ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                text_str, ha='center', va='bottom', fontsize=8)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + y_range * 0.15)

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel(label_map[metric], fontsize=11)
    ax.set_title(f'{label_map[metric]}', fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    for idx, label in enumerate(ax.get_xticklabels()):
        if idx == best_idx:
            label.set_fontweight('bold')

    ax.grid(axis='y', alpha=0.3, linestyle='--')

for i in range(len(raw_metrics), len(axes)):
    axes[i].axis('off')

legend_elements = [
    plt.Line2D([0], [0], marker=markers[0], color='w', markerfacecolor=colors[0],
               markersize=10, label='TFT', markeredgecolor='black', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker=markers[1], color='w', markerfacecolor=colors[1],
               markersize=10, label='Chronos', markeredgecolor='black', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker=markers[2], color='w', markerfacecolor=colors[2],
               markersize=10, label='Chronos+FT', markeredgecolor='black', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker=markers[3], color='w', markerfacecolor=colors[3],
               markersize=10, label='Chronos-Bolt', markeredgecolor='black', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker=markers[4], color='w', markerfacecolor=colors[4],
               markersize=10, label='Chronos-Bolt+FT', markeredgecolor='black', markeredgewidth=1.5)
]
fig.legend(handles=legend_elements, loc='lower right', fontsize=12,
           frameon=True, edgecolor='black', ncol=5, bbox_to_anchor=(0.95, 0.02))

plt.suptitle('Comparison of All Metrics Across Models', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.03, 1, 0.99])
plt.savefig(f'{output_dir}/all_metrics_combined.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nCombined plot saved: {output_dir}/all_metrics_combined.png")