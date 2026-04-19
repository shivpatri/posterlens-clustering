import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
VISUAL_DATA_PATH = ROOT / "visual_data.npy"
PERFORMANCE_PATH = ROOT / "performance.npy"
OUTPUT_DIR = ROOT / "docs"
OUTPUT_PATH = OUTPUT_DIR / "index.html"

METHODS = {
    "kmeans": {"label": "K-Means", "offset_multiplier": 0},
    "gm": {"label": "Gaussian Mixture", "offset_multiplier": 1},
    "h": {"label": "Hierarchical", "offset_multiplier": 2},
}
PERFORMANCE_LABELS = [
    "K-Means",
    "Gaussian Mixture",
    "Hierarchical Clustering",
]


def load_visualization_data():
    visual_data = np.load(VISUAL_DATA_PATH)
    performance = np.load(PERFORMANCE_PATH)
    data = visual_data[:, :3]
    labels = visual_data[:, 3:].astype(int)
    cluster_total = performance.shape[0] // 3
    cluster_values = [2] if cluster_total == 1 else [3, 5, 7, 10, 15, 20][:cluster_total]
    return data, labels, performance, cluster_total, cluster_values


def build_scatter_figure(data, labels, cluster_total, cluster_index, method_key):
    method = METHODS[method_key]
    label_column = cluster_index + (cluster_total * method["offset_multiplier"])
    cluster_labels = labels[:, label_column]
    traces = []

    for cluster_id in range(np.unique(cluster_labels).size):
        cluster_mask = cluster_labels == cluster_id
        traces.append(
            go.Scatter3d(
                x=data[cluster_mask, 0],
                y=data[cluster_mask, 1],
                z=data[cluster_mask, 2],
                mode="markers",
                marker=dict(size=3, color=cluster_id, opacity=1),
                showlegend=True,
                name=f"Cluster {cluster_id}",
            )
        )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=" Scatter Plot ",
        title_x=0.5,
        scene=dict(
            xaxis_title="PCA: 1",
            yaxis_title="PCA: 2",
            zaxis_title="PCA: 3",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
    )
    return figure


def build_performance_figure(performance, cluster_total, cluster_values):
    traces = []
    for i, label in enumerate(PERFORMANCE_LABELS):
        start = cluster_total * i
        stop = start + cluster_total
        traces.append(
            go.Scatter(
                x=np.array(cluster_values),
                y=performance[start:stop],
                name=label,
            )
        )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=" Elbow method with Distortion ",
        title_x=0.5,
        margin=dict(l=60, r=20, b=60, t=60),
    )
    figure.update_xaxes(title="No of clusters")
    figure.update_yaxes(title="Distortion")
    return figure


def build_plot_payload():
    data, labels, performance, cluster_total, cluster_values = load_visualization_data()
    performance_figure = build_performance_figure(performance, cluster_total, cluster_values)
    scatter_figures = {}

    for method_key in METHODS:
        scatter_figures[method_key] = {}
        for cluster_index in range(cluster_total):
            scatter_figures[method_key][str(cluster_index)] = build_scatter_figure(
                data=data,
                labels=labels,
                cluster_total=cluster_total,
                cluster_index=cluster_index,
                method_key=method_key,
            ).to_plotly_json()

    return {
        "clusterValues": cluster_values,
        "methods": {key: value["label"] for key, value in METHODS.items()},
        "performanceFigure": performance_figure.to_plotly_json(),
        "scatterFigures": scatter_figures,
    }


def render_html(payload):
    payload_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PosterLens Clustering</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --page-bg: #f5f7fb;
      --panel-bg: #ffffff;
      --text: #18212f;
      --muted: #5f6c7b;
      --border: #d8e0ea;
      --shadow: 0 16px 40px rgba(23, 35, 61, 0.08);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(81, 132, 255, 0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(30, 180, 140, 0.14), transparent 28%),
        var(--page-bg);
      color: var(--text);
    }}

    .page {{
      width: min(1200px, calc(100% - 32px));
      margin: 0 auto;
      padding: 40px 0 56px;
    }}

    h1 {{
      margin: 0 0 8px;
      text-align: center;
      font-size: clamp(2rem, 3vw, 2.6rem);
      letter-spacing: 0.02em;
    }}

    .subtitle {{
      margin: 0 auto 28px;
      max-width: 720px;
      text-align: center;
      color: var(--muted);
      line-height: 1.5;
    }}

    .controls {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 240px));
      justify-content: center;
      gap: 16px;
      margin-bottom: 24px;
    }}

    label {{
      display: block;
      font-size: 0.95rem;
      font-weight: 600;
      margin-bottom: 8px;
    }}

    select {{
      width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel-bg);
      color: var(--text);
      box-shadow: var(--shadow);
      font: inherit;
    }}

    .plot-card {{
      background: var(--panel-bg);
      border: 1px solid rgba(216, 224, 234, 0.8);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 20px;
    }}

    .plot {{
      min-height: 520px;
    }}

    @media (max-width: 720px) {{
      .page {{
        width: min(100% - 20px, 1200px);
        padding-top: 28px;
      }}

      .controls {{
        grid-template-columns: 1fr;
      }}

      .plot {{
        min-height: 420px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>PosterLens Clustering</h1>
    <p class="subtitle">
      Interactive Plotly visualizations exported as a static site for GitHub Pages.
      The clustering views and legend interactions remain fully client-side.
    </p>

    <section class="controls">
      <div>
        <label for="cluster-method">Clustering method</label>
        <select id="cluster-method"></select>
      </div>
      <div>
        <label for="cluster-count">Number of clusters</label>
        <select id="cluster-count"></select>
      </div>
    </section>

    <section class="plot-card">
      <div id="scatter-plot" class="plot"></div>
    </section>

    <section class="plot-card">
      <div id="performance-plot" class="plot"></div>
    </section>
  </main>

  <script>
    const payload = {payload_json};
    const methodSelect = document.getElementById("cluster-method");
    const clusterSelect = document.getElementById("cluster-count");
    const config = {{ responsive: true, displaylogo: false }};

    Object.entries(payload.methods).forEach(([value, label]) => {{
      methodSelect.add(new Option(label, value));
    }});

    payload.clusterValues.forEach((value, index) => {{
      clusterSelect.add(new Option(String(value), String(index)));
    }});

    methodSelect.value = "kmeans";
    clusterSelect.value = "0";

    function renderPlots() {{
      const method = methodSelect.value;
      const clusterIndex = clusterSelect.value;
      const scatterFigure = payload.scatterFigures[method][clusterIndex];

      Plotly.react(
        "scatter-plot",
        scatterFigure.data,
        scatterFigure.layout,
        config,
      );
      Plotly.react(
        "performance-plot",
        payload.performanceFigure.data,
        payload.performanceFigure.layout,
        config,
      );
    }}

    methodSelect.addEventListener("change", renderPlots);
    clusterSelect.addEventListener("change", renderPlots);
    renderPlots();
  </script>
</body>
</html>
"""


def build_static_site(output_path=OUTPUT_PATH):
    payload = build_plot_payload()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(payload), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    html_path = build_static_site()
    print(f"Static visualization written to {html_path}")
