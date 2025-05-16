import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys

def normalize(val, vmin, vmax, baseline=20):
    return baseline + (val - vmin) / (vmax - vmin) * (100 - baseline)

def normalize_reversed(val, vmin, vmax, baseline=20):
    return baseline + (vmax - val) / (vmax - vmin) * (100 - baseline)

def normalize_cost(val, max_tick, baseline=20):
    return baseline + (max_tick - min(val, max_tick)) / max_tick * (100 - baseline)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def plot_radar(config):
    axis_labels = config['axis_labels']
    model_name = config['model_name']
    baseline = config.get('baseline', 20)
    ticks = config['ticks']
    data = config['data']
    color_map = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    t_key = 'TPOT' if 'Performance' in axis_labels[0] else 'Throughput'

    t_vals = [v[t_key] for v in data.values()]
    c_vals = [v['Cost'] for v in data.values()]
    a_vals = [v['Accuracy'] for v in data.values()]
    
    t_min, t_max = min(t_vals), max(t_vals)
    c_max = ticks[-2]
    a_min, a_max = min(a_vals), 1.0

    angles = np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, len(axis_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 100)
    ax.set_title(model_name, fontsize=16, pad=20)

    for t in np.linspace(baseline, 100, 5):
        ax.plot(np.linspace(0, 2 * np.pi, 500), [t]*500, color="gray", lw=0.3, linestyle='dotted')
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 100], color='gray', lw=0.3)

    for i, (system, values) in enumerate(data.items()):
        val1 = values[t_key]
        val2 = values['Cost']
        val3 = values['Accuracy']

        norm_vals = [
            normalize_reversed(val1, t_min, t_max, baseline) if t_key == 'TPOT' else normalize(val1, t_min, t_max, baseline),
            normalize_cost(val2, c_max, baseline),
            normalize(val3, a_min, a_max, baseline)
        ]
        norm_vals += norm_vals[:1]

        ax.plot(angles, norm_vals, label=system, linewidth=2, color=color_map[i % len(color_map)])
        ax.fill(angles, norm_vals, alpha=0.25, color=color_map[i % len(color_map)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=14)
    ax.set_yticklabels([])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_')}_radar.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python draw_radar.py path/to/config.yaml")
    else:
        config = load_config(sys.argv[1])
        plot_radar(config)
