import yaml
import plotly.graph_objects as go
import os
import tempfile
import shutil
import time

def normalize(val, vmin, vmax, baseline=20):
    return baseline + (val - vmin) / (vmax - vmin) * (100 - baseline)

def normalize_reversed(val, vmin, vmax, baseline=20):
    return baseline + (vmax - val) / (vmax - vmin) * (100 - baseline)

def normalize_cost(val, max_tick, baseline=20):
    return baseline + (max_tick - min(val, max_tick)) / max_tick * (100 - baseline)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def plot_interactive_radar(config):
    axis_labels = config['axis_labels']
    axis_keys = config['axis_keys']
    model_name = config['model_name']
    baseline = config.get('baseline', 20)
    ticks = config['ticks']
    data = config['data']

    perf_key, cost_key, acc_key = axis_keys

    # Extract and normalize values
    perf_vals = [v[perf_key] for v in data.values()]
    cost_vals = [v[cost_key] for v in data.values()]
    acc_vals = [v[acc_key] for v in data.values()]

    perf_min, perf_max = min(perf_vals), max(perf_vals)
    cost_max = ticks[-2]
    acc_min, acc_max = min(acc_vals), 1.0

    categories = axis_labels + [axis_labels[0]]
    fig = go.Figure()

    for system, values in data.items():
        raw_vals = [values[perf_key], values[cost_key], values[acc_key]]
        norm_vals = [
            normalize_reversed(values[perf_key], perf_min, perf_max, baseline),
            normalize_cost(values[cost_key], cost_max, baseline),
            normalize(values[acc_key], acc_min, acc_max, baseline)
        ]
        norm_vals += [norm_vals[0]]
        hovertext = [
            f"{axis_labels[0]}: {raw_vals[0]}",
            f"{axis_labels[1]}: {raw_vals[1]}",
            f"{axis_labels[2]}: {raw_vals[2]}",
            f"{axis_labels[0]}: {raw_vals[0]}"
        ]
        fig.add_trace(go.Scatterpolar(
            r=norm_vals,
            theta=categories,
            fill='toself',
            name=system,
            text=hovertext,
            hoverinfo='text+name',
            line=dict(width=2)
        ))

    fig.update_layout(
        title=model_name,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(
                tickfont=dict(size=12),
                rotation=30,  # rotate the radar chart by 30 degrees
                direction='clockwise'
            ),
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        margin=dict(t=60, b=60, l=60, r=60),
        height=800,
        width=800
    )

    # Save static image
    png_path = f"{model_name.replace(' ', '_')}_radar.png"
    fig.write_image(png_path, width=1000, height=1000, scale=2)
    print(f"PNG saved to: {png_path}")

    # Save HTML (safe atomic overwrite)
    html_path = f"{model_name.replace(' ', '_')}_radar.html"
    with tempfile.NamedTemporaryFile('w', delete=False, suffix=".html") as tmp_file:
        fig.write_html(tmp_file.name)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        shutil.move(tmp_file.name, html_path)
    print(f"Interactive HTML saved to: {html_path}")

    # Optional: Delay to avoid browser loading half-written file
    time.sleep(8)

    # Show in default browser
    fig.show(config={'displayModeBar': True})

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python draw_radar.py path/to/config.yaml")
    else:
        config = load_config(sys.argv[1])
        plot_interactive_radar(config)
