#!/usr/bin/env python3
"""Merge RAJAPerf kernel-run-data CSVs and generate compiler matrix plots.

This script is meant for compiler sweep outputs produced by
`scripts/lc-builds/run_compiler_matrix.sh`, but it also works on any directory
tree that contains RAJAPerf `*-kernel-run-data.csv` files.

The main flow is:
1. Discover matching CSV files from one or more glob patterns.
2. Find the real CSV header row even when RAJAPerf prepends text metadata.
3. Normalize column names across slightly different CSV schemas.
4. Derive compiler/build metadata from the source file path.
5. Merge everything into one dataframe and emit summary tables and plots.
"""

import argparse
import html as html_lib
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Matplotlib needs a writable config/cache directory in this environment.
# Using the current working directory keeps the behavior local to the run
# rather than depending on a user-specific home directory path.
mpl_config_dir = Path.cwd() / ".matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIME_COL = "Mean time per rep (sec.)"
BANDWIDTH_COL = "Mean Bandwidth (GiB per sec.)"
FLOPS_COL = "Mean flops (gigaFLOP per sec.)"
PROBLEM_SIZE_COL = "Problem size"
COMPILER_HINT_TOKENS = (
    "amdclang",
    "hipcc",
    "gcc",
    "clang",
    "nvcc",
    "icpx",
    "icc",
    "cce",
    "cray",
    "oneapi",
    "dpcpp",
    "rocm",
)

METRIC_ALIASES = {
    TIME_COL: [
        TIME_COL,
        "Mean time per rep",
        "Time",
    ],
    BANDWIDTH_COL: [
        BANDWIDTH_COL,
        "Bandwidth (GiB per sec.)",
        "Bandwidth (GiB/s)",
        "Mean bandwidth (GiB/s)",
        "Mean Bandwidth (GiB/s)",
    ],
    FLOPS_COL: [
        FLOPS_COL,
        "Mean flops (GFlop/s)",
        "Mean GFLOP/s",
        "GFLOP/s",
        "GFlop/s",
    ],
}

MATPLOTLIB_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "8", "d"]
PLOTLY_MARKERS = [
    "circle",
    "square",
    "triangle-up",
    "diamond",
    "triangle-down",
    "cross",
    "x",
    "star",
    "hexagon",
    "triangle-left",
    "triangle-right",
    "pentagon",
    "octagon",
    "diamond-tall",
]


def find_build_folder(path: Path) -> str:
    """Infer the build/compiler identifier from a CSV path.

    Preferred matches are actual `build_*` directories. The fallback cases are
    there for sweep outputs that were copied or reorganized but still keep a
    compiler-like directory name in the path.
    """
    candidates = [path.parent, *path.parents]

    for parent in candidates:
        if parent.name.startswith("build_"):
            return parent.name

    for parent in candidates:
        if parent.name.startswith("install_"):
            return parent.name

    for parent in candidates:
        lower_name = parent.name.lower()
        if lower_name.startswith("lc_") or any(token in lower_name for token in COMPILER_HINT_TOKENS):
            return parent.name

    return ""


def compiler_label(build_folder: str) -> str:
    """Turn a build folder name into a shorter plot-friendly compiler label."""
    label = build_folder
    if label.startswith("build_"):
        label = label[len("build_") :]
    if label.startswith("install_"):
        label = label[len("install_") :]
    if label.startswith("lc_toss4-"):
        label = label[len("lc_toss4-") :]
    elif label.startswith("lc_"):
        label = label[len("lc_") :]
    return label or "unknown-build"


def sanitize_filename(text: object) -> str:
    """Make plot/table filenames stable and shell-friendly."""
    sanitized = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_.") or "output"


def matplotlib_color_sequence() -> List[str]:
    """Use Matplotlib's categorical tab20 palette for compiler colors."""
    return [mcolors.to_hex(color) for color in plt.get_cmap("tab20").colors]


def make_style_map(values: Iterable[object], styles: List[str], style_name: str) -> Dict[str, str]:
    """Assign styles deterministically from the sorted names present in this run."""
    names = sorted({str(value).strip() for value in values if str(value).strip()})
    if len(names) > len(styles):
        print(
            f"[INFO] {len(names)} unique {style_name} values exceed {len(styles)} available styles; "
            "styles will repeat."
        )
    return {name: styles[index % len(styles)] for index, name in enumerate(names)}


def line_dash_for_variant(variant: object) -> str:
    """Use line style for the implementation family."""
    name = str(variant).strip()
    if name.startswith("Base_"):
        return "dashed"
    if name.startswith("RAJA_"):
        return "solid"
    if name.startswith("Lambda_"):
        return "dotted"
    if name.startswith("Kokkos_"):
        return "dashdot"
    return "solid"


def plotly_dash_for_variant(variant: object) -> str:
    """Use Plotly-compatible line style for the implementation family."""
    name = str(variant).strip()
    if name.startswith("Base_"):
        return "dash"
    if name.startswith("RAJA_"):
        return "solid"
    if name.startswith("Lambda_"):
        return "dot"
    if name.startswith("Kokkos_"):
        return "dashdot"
    return "solid"


def write_plot_index_pages(output_dir: Path, plot_entries: List[Dict[str, str]]) -> None:
    """Write a landing page plus one embedded plot page per kernel."""
    if not plot_entries:
        return

    by_kernel: Dict[str, List[Dict[str, str]]] = {}
    for entry in plot_entries:
        by_kernel.setdefault(entry["kernel"], []).append(entry)

    kernel_page_names = {
        kernel: f"{sanitize_filename(kernel)}.html"
        for kernel in sorted(by_kernel)
    }
    plot_kind_names = {
        "html": "Plotly HTML",
        "pdf": "PDF",
        "png": "PNG",
    }
    plot_kind = plot_kind_names.get(plot_entries[0]["kind"], plot_entries[0]["kind"].upper())

    css = """
    body { background: #111827; color: #e5e7eb; font-family: system-ui, sans-serif; margin: 2rem; }
    a { color: #93c5fd; text-decoration: none; }
    a:hover { text-decoration: underline; }
    table { border-collapse: collapse; width: 100%; max-width: 1100px; }
    th, td { border-bottom: 1px solid #374151; padding: 0.65rem 0.75rem; text-align: left; }
    th { color: #cbd5e1; font-weight: 600; }
    .metric-links a { margin-right: 1rem; white-space: nowrap; }
    .summary { color: #9ca3af; margin-bottom: 1.5rem; }
    .plot-frame { border: 1px solid #374151; border-radius: 8px; height: clamp(760px, 92vh, 1200px); width: 100%; }
    .plot-image { border: 1px solid #374151; border-radius: 8px; max-width: 100%; }
    .plot-section { margin: 2rem 0 3rem; }
    """

    index_rows = []
    for kernel in sorted(by_kernel):
        entries = by_kernel[kernel]
        page_name = kernel_page_names[kernel]
        metric_links = " ".join(
            '<a href="{page}#{anchor}">{metric}</a>'.format(
                page=html_lib.escape(page_name),
                anchor=html_lib.escape(f"metric-{sanitize_filename(entry['metric'])}"),
                metric=html_lib.escape(entry["metric"]),
            )
            for entry in entries
        )
        index_rows.append(
            "<tr><td><a href=\"{page}\">{kernel}</a></td><td class=\"metric-links\">{metrics}</td></tr>".format(
                page=html_lib.escape(page_name),
                kernel=html_lib.escape(kernel),
                metrics=metric_links,
            )
        )

    index_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RAJAPerf Compiler Matrix Plots</title>
  <style>{css}</style>
</head>
<body>
  <h1>RAJAPerf Compiler Matrix Plots</h1>
  <p class="summary">{kernel_count} kernels · {plot_count} {plot_kind} plots</p>
  <table>
    <thead><tr><th>Kernel</th><th>Plots</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
""".format(
        css=css,
        kernel_count=len(by_kernel),
        plot_count=len(plot_entries),
        plot_kind=html_lib.escape(plot_kind),
        rows="\n      ".join(index_rows),
    )
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    for kernel in sorted(by_kernel):
        sections = []
        for entry in by_kernel[kernel]:
            metric = entry["metric"]
            filename = entry["filename"]
            anchor = f"metric-{sanitize_filename(metric)}"
            if entry["kind"] == "png":
                embedded_plot = '<img class="plot-image" src="{src}" alt="{alt}">'.format(
                    src=html_lib.escape(filename),
                    alt=html_lib.escape(f"{kernel} {metric}"),
                )
            else:
                embedded_plot = '<iframe class="plot-frame" src="{src}" title="{title}"></iframe>'.format(
                    src=html_lib.escape(filename),
                    title=html_lib.escape(f"{kernel} {metric}"),
                )
            sections.append(
                '<section class="plot-section" id="{anchor}">\n'
                "  <h2>{metric}</h2>\n"
                '  <p><a href="{filename}">Open standalone plot</a></p>\n'
                "  {embedded_plot}\n"
                "</section>".format(
                    anchor=html_lib.escape(anchor),
                    metric=html_lib.escape(metric),
                    filename=html_lib.escape(filename),
                    embedded_plot=embedded_plot,
                )
            )

        kernel_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{kernel} - RAJAPerf Compiler Matrix Plots</title>
  <style>{css}</style>
</head>
<body>
  <p><a href="index.html">← Back to all kernels</a></p>
  <h1>{kernel}</h1>
  <p class="summary">{plot_count} {plot_kind} plots</p>
  {sections}
</body>
</html>
""".format(
            css=css,
            kernel=html_lib.escape(kernel),
            plot_count=len(by_kernel[kernel]),
            plot_kind=html_lib.escape(plot_kind),
            sections="\n  ".join(sections),
        )
        (output_dir / kernel_page_names[kernel]).write_text(kernel_html, encoding="utf-8")


def parse_factor(path: Path) -> float:
    """Extract the numeric throughput factor from a standard sweep filename.

    Expected examples:
      DIFFUSION3DPA_factor_1024-kernel-run-data.csv -> 1024.0
      VOL3D_factor_64-kernel-run-data.csv           -> 64.0

    Returns NaN when the filename is not from a factor sweep.
    """
    match = re.search(r"(?:^|_)factor_([0-9]+(?:\.[0-9]+)?)", path.name)
    if not match:
        return np.nan
    return float(match.group(1))


def parse_run_name(path: Path) -> str:
    """Drop the standard CSV suffix to keep the run/file stem for reporting."""
    name = path.name
    if name.endswith("-kernel-run-data.csv"):
        name = name[: -len("-kernel-run-data.csv")]
    return name


def find_csv_files(root_dir: Path, patterns: Iterable[str]) -> List[Path]:
    """Resolve one or more glob patterns into a unique sorted file list.

    Patterns are evaluated relative to root_dir so callers can merge multiple
    build trees at once, for example:
      build_*/compiler_sweep_runs/tier1_base_raja_hip/*kernel-run-data.csv
    """
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root_dir.glob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})


def header_score(line: str) -> int:
    """Score a line by how much it looks like a kernel-run-data CSV header."""
    tokens = ["Kernel", "Variant", "Tuning", "Problem size", "Mean time", "Mean flops", "Bandwidth"]
    return sum(1 for token in tokens if token in line)


def read_kernel_run_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read one RAJAPerf CSV, allowing for preamble text before the header row."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="latin-1").splitlines()
    except OSError as exc:
        print(f"[SKIP] Could not read {path}: {exc}")
        return None

    header_idx = None
    best_score = -1
    for idx, line in enumerate(lines[:50]):
        score = header_score(line)
        if score > best_score:
            best_score = score
            header_idx = idx
        if score >= 4:
            break

    if header_idx is None or best_score <= 0:
        print(f"[SKIP] Could not identify CSV header in {path}")
        return None

    try:
        # RAJAPerf often writes one or more descriptive lines before the actual
        # CSV header, so the parser starts at the detected header row.
        df = pd.read_csv(path, header=header_idx, skipinitialspace=True)
    except Exception as exc:
        print(f"[SKIP] Could not parse {path}: {exc}")
        return None

    return normalize_columns(df)


def coalesce_prefixed_columns(df: pd.DataFrame, output_name: str, prefix: str) -> pd.DataFrame:
    """Collapse duplicate/prefixed columns that can appear after CSV parsing."""
    columns = [c for c in df.columns if c == output_name or c.lower().startswith(prefix.lower())]
    if not columns:
        return df
    df[output_name] = df[columns].replace("", np.nan).bfill(axis=1).iloc[:, 0]
    for column in columns:
        if column != output_name:
            df = df.drop(columns=column)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known schema variations into one canonical column set."""
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = coalesce_prefixed_columns(df, "Kernel", "Kernel")
    df = coalesce_prefixed_columns(df, "Tuning", "Tuning")

    rename_map: Dict[str, str] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        for column in df.columns:
            if column.strip() in aliases:
                rename_map[column] = canonical
                break

    for column in df.columns:
        if column.strip() in {"Problem Size", "ProblemSize", "Size"}:
            rename_map[column] = PROBLEM_SIZE_COL

    return df.rename(columns=rename_map)


def collect_kernel_run_data(
    root_dir: Path,
    patterns: List[str],
    kernel_filters: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load, normalize, annotate, and merge all matching kernel-run-data CSVs.

    The returned dataframe is the central analysis table used by both the CLI
    plots and the notebook. Each input file contributes its original benchmark
    rows plus derived metadata columns describing where that row came from.
    """
    files = find_csv_files(root_dir, patterns)
    if verbose:
        print(f"Found {len(files)} CSV files under {root_dir}")

    frames: List[pd.DataFrame] = []
    for path in files:
        df = read_kernel_run_csv(path)
        if df is None:
            continue

        required = {"Kernel", "Variant", "Tuning", PROBLEM_SIZE_COL}
        missing = sorted(required - set(df.columns))
        if missing:
            print(f"[SKIP] {path} missing columns: {missing}")
            continue

        # These derived columns are what let downstream plots compare compilers
        # without requiring the original directory structure at plotting time.
        build_folder = find_build_folder(path)
        df["BuildFolder"] = build_folder
        df["Compiler"] = compiler_label(build_folder)
        df["SourceFile"] = str(path)
        df["RunName"] = parse_run_name(path)
        df["Factor"] = parse_factor(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    for column in ["Kernel", "Variant", "Tuning", "BuildFolder", "Compiler"]:
        combined[column] = combined[column].astype(str).str.strip()

    for column in [PROBLEM_SIZE_COL, TIME_COL, BANDWIDTH_COL, FLOPS_COL, "Factor"]:
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")

    if kernel_filters:
        lowered = [item.lower() for item in kernel_filters]
        kernel_text = combined["Kernel"].fillna("").astype(str).str.lower()
        combined = combined[kernel_text.apply(lambda value: any(item in value for item in lowered))]

    combined["VariantTuning"] = (
        combined["Variant"].astype(str).str.strip() + " | " + combined["Tuning"].astype(str).str.strip()
    )
    combined["CompilerVariantTuning"] = (
        combined["Compiler"].astype(str).str.strip() + " | " + combined["VariantTuning"]
    )
    return combined


def choose_available_metrics(df: pd.DataFrame, requested: Optional[List[str]]) -> List[str]:
    """Keep only metrics that are present and have at least one real value."""
    metrics = requested or [TIME_COL, FLOPS_COL, BANDWIDTH_COL]
    available = [metric for metric in metrics if metric in df.columns and df[metric].notna().any()]
    missing = [metric for metric in metrics if metric not in available]
    for metric in missing:
        print(f"[INFO] Metric unavailable or empty, skipping: {metric}")
    return available


def _plot_throughput_curves_matplotlib(
    sweep_df: pd.DataFrame,
    output_dir: Path,
    metrics: List[str],
    compiler_color_map: Dict[str, str],
    tuning_marker_map: Dict[str, str],
    *,
    output_format: str,
) -> List[Dict[str, str]]:
    """Matplotlib line plots for metric vs problem size."""
    plot_entries: List[Dict[str, str]] = []
    for metric in metrics:
        metric_df = sweep_df.dropna(subset=[PROBLEM_SIZE_COL, metric]).copy()
        metric_df = metric_df[(metric_df[PROBLEM_SIZE_COL] > 0) & (metric_df[metric] > 0)]
        if metric_df.empty:
            continue

        for kernel, kernel_df in metric_df.groupby("Kernel", sort=True):
            fig, ax = plt.subplots(figsize=(15, 7))
            plotted = False

            group_columns = ["Compiler", "Variant", "Tuning", "CompilerVariantTuning"]
            for (compiler, variant, tuning, label), line_df in kernel_df.groupby(group_columns, sort=True):
                line_df = line_df.sort_values([PROBLEM_SIZE_COL, "Factor"])
                if line_df.empty:
                    continue
                ax.plot(
                    line_df[PROBLEM_SIZE_COL],
                    line_df[metric],
                    color=compiler_color_map.get(str(compiler), "#1f77b4"),
                    marker=tuning_marker_map.get(str(tuning), "o"),
                    linewidth=2,
                    linestyle=line_dash_for_variant(variant),
                    markersize=5,
                    label=label,
                )
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_title(f"{kernel} throughput - {metric}", fontsize=15)
            ax.set_xlabel("Problem size")
            ax.set_ylabel(metric)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)
            ax.legend(
                title="Compiler | Variant | Tuning\n(color | line | marker)",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=8,
            )
            fig.tight_layout()
            out_path = output_dir / f"{sanitize_filename(kernel)}_{sanitize_filename(metric)}_throughput.{output_format}"
            fig.savefig(out_path, format=output_format, bbox_inches="tight")
            plt.close(fig)
            plot_entries.append(
                {"kernel": str(kernel), "metric": str(metric), "filename": out_path.name, "kind": output_format}
            )

    return plot_entries


def _plot_throughput_curves_plotly(
    sweep_df: pd.DataFrame,
    output_dir: Path,
    metrics: List[str],
    tuning_marker_map: Dict[str, str],
    *,
    write_png: bool,
) -> List[Dict[str, str]]:
    """Plotly throughput: standalone HTML, or PNG via ``write_image`` (same basename as PDF/HTML)."""
    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative
    except ImportError as exc:
        raise ImportError(
            "Throughput Plotly output requires the 'plotly' package (e.g. pip install plotly)."
        ) from exc

    if write_png:
        try:
            import kaleido  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Throughput PNG (--confluence) requires the 'kaleido' package (e.g. pip install kaleido)."
            ) from exc

    compiler_color_map = make_style_map(sweep_df["Compiler"], qualitative.Dark24, "Plotly compiler color")
    plot_entries: List[Dict[str, str]] = []

    grid_style = dict(
        showgrid=True,
        gridcolor="rgba(220, 220, 240, 0.22)",
        gridwidth=1,
        griddash="dash",
        tickfont=dict(size=15),
        title_font=dict(size=17),
    )

    for metric in metrics:
        metric_df = sweep_df.dropna(subset=[PROBLEM_SIZE_COL, metric]).copy()
        metric_df = metric_df[(metric_df[PROBLEM_SIZE_COL] > 0) & (metric_df[metric] > 0)]
        if metric_df.empty:
            continue

        for kernel, kernel_df in metric_df.groupby("Kernel", sort=True):
            traces = []

            group_columns = ["Compiler", "Variant", "Tuning", "CompilerVariantTuning"]
            for (compiler, variant, tuning, label), line_df in kernel_df.groupby(group_columns, sort=True):
                line_df = line_df.sort_values([PROBLEM_SIZE_COL, "Factor"])
                if line_df.empty:
                    continue
                color = compiler_color_map.get(str(compiler), "#1f77b4")
                traces.append(
                    go.Scatter(
                        x=line_df[PROBLEM_SIZE_COL],
                        y=line_df[metric],
                        mode="lines+markers",
                        name=str(label),
                        line=dict(width=2, dash=plotly_dash_for_variant(variant), color=color),
                        marker=dict(size=10, symbol=tuning_marker_map.get(str(tuning), "circle"), color=color),
                    )
                )

            if not traces:
                continue

            fig = go.Figure(data=traces)
            legend_rows = max(1, int(np.ceil(len(traces) / 2)))
            legend_bottom_margin = min(420, max(130, 44 * legend_rows + 70))
            plot_size = (
                dict(width=1500, height=760)
                if write_png
                else dict(autosize=True)
            )
            fig.update_layout(
                title=dict(text=f"{kernel} throughput - {metric}", font=dict(size=24)),
                xaxis_title="Problem size",
                yaxis_title=metric,
                legend_title_text="Compiler | Variant | Tuning<br>(color | line | marker)",
                legend=dict(
                    font=dict(size=14),
                    title_font=dict(size=15),
                    orientation="h",
                    yanchor="top",
                    y=-0.14,
                    xanchor="left",
                    x=0,
                ),
                hovermode="closest",
                hoverlabel=dict(namelength=-1, font=dict(size=15)),
                template="plotly_dark",
                font=dict(size=16),
                margin=dict(l=76, r=40, t=76, b=legend_bottom_margin),
                **plot_size,
            )
            fig.update_xaxes(type="log", **grid_style)
            fig.update_yaxes(type="log", **grid_style)

            stem = f"{sanitize_filename(kernel)}_{sanitize_filename(metric)}_throughput"
            if write_png:
                out_path = output_dir / f"{stem}.png"
                fig.write_image(str(out_path))
                plot_entries.append(
                    {"kernel": str(kernel), "metric": str(metric), "filename": out_path.name, "kind": "png"}
                )
            else:
                out_path = output_dir / f"{stem}.html"
                fig.write_html(
                    out_path,
                    include_plotlyjs="cdn",
                    full_html=True,
                    config={"responsive": True},
                    default_width="100%",
                    default_height="96vh",
                )
                plot_entries.append(
                    {"kernel": str(kernel), "metric": str(metric), "filename": out_path.name, "kind": "html"}
                )

    return plot_entries


def plot_throughput_curves(
    df: pd.DataFrame,
    output_dir: Path,
    metrics: List[str],
    html: bool = False,
    *,
    confluence: bool = False,
    png: bool = False,
) -> None:
    """Plot metric vs problem size for factor sweeps or multi-size runs.

    A row is considered sweep-like if either:
    - a numeric factor was parsed from the filename, or
    - the same kernel/compiler/variant/tuning appears at multiple sizes

    By default writes Matplotlib PDFs plus browser index pages. With ``png=True``,
    writes Matplotlib PNGs plus browser index pages. With ``html=True`` and
    ``confluence=False``, writes standalone Plotly HTML. With ``confluence=True``,
    writes the same Plotly figure as PNG (``*_throughput.png``, same stem as PDF/HTML)
    via ``fig.write_image``. If both ``html`` and ``confluence`` are true, PNG is used.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = df.copy()
    sweep_df = sweep_df[sweep_df["Factor"].notna() | sweep_df.duplicated(["Kernel", "CompilerVariantTuning"], keep=False)]
    if sweep_df.empty:
        print("[INFO] No factor-sweep or multi-size data found for throughput plots.")
        return

    if html or confluence:
        plotly_tuning_marker_map = make_style_map(sweep_df["Tuning"], PLOTLY_MARKERS, "Plotly tuning marker")
        plot_entries = _plot_throughput_curves_plotly(
            sweep_df,
            output_dir,
            metrics,
            plotly_tuning_marker_map,
            write_png=confluence,
        )
        write_plot_index_pages(output_dir, plot_entries)
    else:
        matplotlib_compiler_color_map = make_style_map(
            sweep_df["Compiler"],
            matplotlib_color_sequence(),
            "Matplotlib compiler color",
        )
        matplotlib_tuning_marker_map = make_style_map(sweep_df["Tuning"], MATPLOTLIB_MARKERS, "matplotlib tuning marker")
        output_format = "png" if png else "pdf"
        plot_entries = _plot_throughput_curves_matplotlib(
            sweep_df,
            output_dir,
            metrics,
            matplotlib_compiler_color_map,
            matplotlib_tuning_marker_map,
            output_format=output_format,
        )
        write_plot_index_pages(output_dir, plot_entries)


def write_summary_tables(df: pd.DataFrame, output_dir: Path, metrics: List[str]) -> None:
    """Write merged and peak-performance summary tables alongside the plots.

    The peak tables keep the single best row per kernel and
    compiler/variant/tuning label for the requested metric.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_columns = ["Kernel", "Compiler", "Variant", "Tuning", PROBLEM_SIZE_COL, "Factor", *metrics]
    present_columns = [column for column in summary_columns if column in df.columns]
    df[present_columns].sort_values(["Kernel", "Compiler", "Variant", "Tuning", PROBLEM_SIZE_COL]).to_csv(
        output_dir / "kernel_run_data_summary.csv",
        index=False,
    )

    if FLOPS_COL in metrics and FLOPS_COL in df.columns:
        idx = df.groupby(["Kernel", "CompilerVariantTuning"])[FLOPS_COL].idxmax()
        peak = df.loc[idx.dropna().astype(int), present_columns].sort_values(["Kernel", "Compiler", "Variant", "Tuning"])
        peak.to_csv(output_dir / "peak_flops_by_kernel.csv", index=False)

    if BANDWIDTH_COL in metrics and BANDWIDTH_COL in df.columns:
        idx = df.groupby(["Kernel", "CompilerVariantTuning"])[BANDWIDTH_COL].idxmax()
        peak = df.loc[idx.dropna().astype(int), present_columns].sort_values(["Kernel", "Compiler", "Variant", "Tuning"])
        peak.to_csv(output_dir / "peak_bandwidth_by_kernel.csv", index=False)


def parse_args() -> argparse.Namespace:
    """Define the small CLI used by ad hoc plotting and notebook preparation."""
    parser = argparse.ArgumentParser(
        description="Read RAJAPerf kernel-run-data CSVs and plot compiler comparisons."
    )
    parser.add_argument("--root-dir", default=".", help="Directory to search recursively (default: current directory)")
    parser.add_argument("--output-dir", default="compiler-matrix-output", help="Output directory")
    parser.add_argument(
        "--glob-pattern",
        action="append",
        default=None,
        help="Glob for input CSVs, repeatable (default: **/*kernel-run-data.csv)",
    )
    parser.add_argument(
        "--kernel",
        action="append",
        default=None,
        help="Kernel-name substring to include, repeatable",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=[TIME_COL, FLOPS_COL, BANDWIDTH_COL],
        default=None,
        help="Metric to plot, repeatable (default: time, flops, bandwidth when present)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Write throughput plots as standalone Plotly HTML pages (default: vector PDF via matplotlib)",
    )
    parser.add_argument(
        "--confluence",
        action="store_true",
        help="Write throughput as Plotly PNG (fig.write_image) for Confluence; same basename as PDF/HTML; requires kaleido",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Write throughput plots as Matplotlib PNG files with index pages instead of Matplotlib PDF",
    )
    parser.add_argument(
        "--no-throughput-plots",
        action="store_true",
        help="Skip throughput plots (PDF by default; use --html or --confluence for Plotly)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print input discovery details")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint: load data, write merged tables, and render requested plots."""
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    patterns = args.glob_pattern or ["**/*kernel-run-data.csv"]

    # Input discovery is intentionally flexible: callers can point at a single
    # build tree, the repository root, or a multi-build glob under the repo.
    df = collect_kernel_run_data(
        root_dir=root_dir,
        patterns=patterns,
        kernel_filters=args.kernel,
        verbose=args.verbose,
    )
    if df.empty:
        print("No kernel-run-data CSV rows found.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "kernel-run-data-merged.csv"
    df.to_csv(combined_path, index=False)
    print(f"Wrote merged data: {combined_path}")

    metrics = choose_available_metrics(df, args.metric)
    if not metrics:
        print("No requested metrics are available to plot.")
        return 1

    write_summary_tables(df, output_dir, metrics)
    if not args.no_throughput_plots:
        try:
            plot_throughput_curves(
                df,
                output_dir / "throughput-plots",
                metrics,
                html=args.html,
                confluence=args.confluence,
                png=args.png,
            )
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    print(f"Wrote compiler comparison outputs under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
