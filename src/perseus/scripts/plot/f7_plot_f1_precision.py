from os import path
import pickle
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import precision_score, f1_score, auc
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from perseus.settings import PROJECT_ROOT
import copy

# Load results (unflattened for batch plot)
with open(path.join(PROJECT_ROOT, "data", "new_results_btc_batched.pkl"), "rb") as file:
    batch_results = pickle.load(file)


# Global style settings
bold_linewidth = 5  # Line width for curves
size = 36  # Font size for plots

# Global legend settings
LEGEND_SIZE = size - 2
LEGEND_TITLE_SIZE = size
LEGEND_HANDLELENGTH = 2  # Adjust handle length if needed

# Mapping dictionaries for styling
line_styles = {"Directed": (0, (4, 4)), "Weighted": "solid"}  # Custom dash: 4 on, 4 off
model_colors = {"MultiGAT": "#1f77b4", "MultiGraphSAGE": "#ff7f0e"}
label_map = {"DDINA": "Directed", "DDM": "Weighted"}
legend_map = {"Directed": "D", "Weighted": "W"}
model_legend_map = {"MultiGAT": "A", "MultiGraphSAGE": "S"}

# For old plots, flatten to default batch size
default_batch_size = 2
results_m = copy.deepcopy(batch_results)
for dataset in results_m:
    for model in results_m[dataset]:
        if (
            isinstance(results_m[dataset][model], dict)
            and default_batch_size in results_m[dataset][model]
        ):
            results_m[dataset][model] = results_m[dataset][model][default_batch_size]


# Helper: Create a figure and axis with common size
def get_common_ax(figsize=(8, 8)):
    """
    Create a matplotlib figure and axis with a common size.

    Args:
        figsize (tuple, optional): Figure size in inches. Defaults to (8, 8).

    Returns:
        tuple: (fig, ax) where fig is the Figure and ax is the Axes object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# Precision Plot
def plot_precision_common_ax(ax, results, colors, linestyles, label_map, fontsize=10):
    """
    Plot precision curves for each model and dataset on a common axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        results (dict): Nested dict of results[dataset][model]["metrics"] with 'labels' and 'probs'.
        colors (dict): Mapping from model name to color.
        linestyles (dict): Mapping from dataset label to line style.
        label_map (dict): Mapping from dataset key to label string.
        fontsize (int, optional): Font size for labels and ticks. Defaults to 10.
    """
    thresholds = np.linspace(0, 1, 100)
    combined_handles = {}
    # Plot precision curves for each model and dataset
    for model in model_colors.keys():
        for dataset in label_map.keys():
            labels = results[dataset][model]["metrics"]["labels"]
            probs = results[dataset][model]["metrics"]["probs"]
            scores = [
                precision_score(labels, probs > threshold, zero_division=0)
                for threshold in thresholds
            ]
            ax.plot(
                thresholds,
                scores,
                color=colors[model],
                linestyle=linestyles[label_map[dataset]],
                linewidth=bold_linewidth,
            )
            combined_label = (
                f"{legend_map[label_map[dataset]]} {model_legend_map[model]}"
            )
            if combined_label not in combined_handles:
                combined_handles[combined_label] = Line2D(
                    [0],
                    [0],
                    color=colors[model],
                    linestyle=linestyles[label_map[dataset]],
                    linewidth=bold_linewidth,
                    label=combined_label,
                )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Threshold", fontsize=fontsize)
    ax.set_ylabel("Precision", fontsize=fontsize)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    legend = ax.legend(
        handles=list(combined_handles.values()),
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="lower center",
        handlelength=LEGEND_HANDLELENGTH,
    )
    plt.setp(legend.get_texts(), fontsize=LEGEND_SIZE)
    legend.get_title().set_fontsize(LEGEND_TITLE_SIZE)


# F1 Plot
def plot_f1_common_ax(ax, results, colors, linestyles, label_map, fontsize=10):
    """
    Plot F1 score curves for each model and dataset on a common axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        results (dict): Nested dict of results[dataset][model]["metrics"] with 'labels' and 'probs'.
        colors (dict): Mapping from model name to color.
        linestyles (dict): Mapping from dataset label to line style.
        label_map (dict): Mapping from dataset key to label string.
        fontsize (int, optional): Font size for labels and ticks. Defaults to 10.
    """
    thresholds = np.linspace(0, 1, 100)
    combined_handles = {}
    for model in model_colors.keys():
        for dataset in label_map.keys():
            labels = results[dataset][model]["metrics"]["labels"]
            probs = results[dataset][model]["metrics"]["probs"]
            scores = [
                f1_score(labels, probs > threshold, zero_division=0)
                for threshold in thresholds
            ]
            ax.plot(
                thresholds,
                scores,
                color=colors[model],
                linestyle=linestyles[label_map[dataset]],
                linewidth=bold_linewidth,
            )
            combined_label = (
                f"{legend_map[label_map[dataset]]} {model_legend_map[model]}"
            )
            if combined_label not in combined_handles:
                combined_handles[combined_label] = Line2D(
                    [0],
                    [0],
                    color=colors[model],
                    linestyle=linestyles[label_map[dataset]],
                    linewidth=bold_linewidth,
                    label=combined_label,
                )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Threshold", fontsize=fontsize)
    ax.set_ylabel("F1", fontsize=fontsize)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    legend = ax.legend(
        handles=list(combined_handles.values()),
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="lower left",
        handlelength=LEGEND_HANDLELENGTH,
    )
    plt.setp(legend.get_texts(), fontsize=LEGEND_SIZE)
    legend.get_title().set_fontsize(LEGEND_TITLE_SIZE)


# # Inference Plot
# def plot_infer_common_ax(
#     ax,
#     results,
#     label_map,
#     fontsize=12,
#     model_colors=model_colors,
#     line_styles=line_styles,
# ):
#     """
#     Plot inference speed vs. number of nodes for each model and dataset.
#
#     Args:
#         ax (matplotlib.axes.Axes): The axis to plot on.
#         results (dict): Nested dict of results[dataset][model] with 'batch_times' and 'num_nodes'.
#         label_map (dict): Mapping from dataset key to label string.
#         fontsize (int, optional): Font size for labels and ticks. Defaults to 12.
#         model_colors (dict): Mapping from model name to color.
#         line_styles (dict): Mapping from dataset label to line style.
#     """
#     combined_handles = {}
#     for model_name in model_colors.keys():
#         for dataset_key, dataset in label_map.items():
#             if model_name in results[dataset_key]:
#                 batch_times = np.array(results[dataset_key][model_name]["batch_times"])
#                 num_nodes = np.array(results[dataset_key][model_name]["num_nodes"])
#                 unique_nodes, indices = np.unique(num_nodes, return_inverse=True)
#                 average_batch_times = np.zeros_like(unique_nodes, dtype=float)
#                 for i in range(len(unique_nodes)):
#                     average_batch_times[i] = np.mean(batch_times[indices == i])
#                 if len(unique_nodes) > 3:
#                     spline = make_interp_spline(unique_nodes, average_batch_times, k=3)
#                     fine_x = np.linspace(unique_nodes.min(), unique_nodes.max(), 500)
#                     fine_y = spline(fine_x)
#                     ax.plot(
#                         fine_x,
#                         fine_y,
#                         color=model_colors[model_name],
#                         linestyle=line_styles[dataset],
#                         linewidth=bold_linewidth,
#                     )
#                 else:
#                     ax.plot(
#                         unique_nodes,
#                         average_batch_times,
#                         "o-",
#                         color=model_colors[model_name],
#                         linestyle=line_styles[dataset],
#                         linewidth=bold_linewidth,
#                     )
#                 combined_label = f"{legend_map[dataset]} {model_legend_map[model_name]}"
#                 if combined_label not in combined_handles:
#                     combined_handles[combined_label] = Line2D(
#                         [0],
#                         [0],
#                         color=model_colors[model_name],
#                         linestyle=line_styles[dataset],
#                         linewidth=bold_linewidth,
#                         label=combined_label,
#                     )
#     ax.set_xlim(3, 15)
#     ax.set_ylim(0.00009, 0.00030)
#     ax.set_xlabel("Number of Nodes", fontsize=fontsize)
#     ax.set_ylabel("Inference Speed (sec)", fontsize=fontsize)
#     ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#     ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
#     ax.yaxis.get_offset_text().set_fontsize(fontsize)
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
#     ax.tick_params(axis="both", which="major", labelsize=fontsize)
#     ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)
#     legend = ax.legend(
#         handles=list(combined_handles.values()),
#         fontsize=LEGEND_SIZE,
#         title_fontsize=LEGEND_TITLE_SIZE,
#         loc="lower center",
#         handlelength=LEGEND_HANDLELENGTH,
#     )
#     plt.setp(legend.get_texts(), fontsize=LEGEND_SIZE)
#     legend.get_title().set_fontsize(LEGEND_TITLE_SIZE)
#     ax.set_box_aspect(1)  # For Matplotlib 3.3+.


# Combined CDF Plot
def plot_combined_cdf_common_ax(
    ax, results_t, fontsize, label_map, line_styles, model_colors
):
    """
    Plot combined CDF of training times for all methods and datasets on a common axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        results_t (dict): Nested dict of results[dataset][method]["train_times"].
        fontsize (int): Font size for labels and ticks.
        label_map (dict): Mapping from dataset key to label string.
        line_styles (dict): Mapping from dataset label to line style.
        model_colors (dict): Mapping from model name to color.
    """
    # Collect data for all methods and datasets
    data = {
        method: {
            dataset: results_t[dataset][method]["train_times"]
            for dataset in label_map.keys()
        }
        for method in model_colors.keys()
    }
    all_times = []
    for method in model_colors.keys():
        for dataset in label_map.keys():
            all_times.extend(results_t[dataset][method]["train_times"])
    global_min = min(all_times)
    global_max = max(all_times)
    combined_handles = {}
    for method in data:
        for dataset in data[method]:
            sorted_times = np.sort(data[method][dataset])
            cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
            style_key = label_map.get(dataset, dataset)
            current_linestyle = line_styles.get(style_key, "solid")
            ax.plot(
                sorted_times,
                cdf,
                drawstyle="steps-post",
                color=model_colors[method],
                linestyle=current_linestyle,
                linewidth=bold_linewidth,
            )
            combined_label = f"{legend_map[style_key]} {model_legend_map[method]}"
            if combined_label not in combined_handles:
                combined_handles[combined_label] = Line2D(
                    [0],
                    [0],
                    color=model_colors[method],
                    linestyle=current_linestyle,
                    linewidth=bold_linewidth,
                    label=combined_label,
                )
    ax.set_xscale("log")
    ax.set_xlim(global_min, global_max)
    major_ticks = np.logspace(np.log10(global_min), np.log10(global_max), num=5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:.2f}" if x < 1 else f"{int(x)}")
    )
    ax.minorticks_off()
    ax.tick_params(axis="x", which="major", labelsize=fontsize)
    y_ticks = np.linspace(0, 1, 6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])
    ax.tick_params(axis="y", which="major", labelsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time per Epoch (seconds)", fontsize=fontsize)
    ax.set_ylabel("CDF", fontsize=fontsize)
    legend = ax.legend(
        handles=list(combined_handles.values()),
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="lower center",
        handlelength=LEGEND_HANDLELENGTH,
    )
    plt.setp(legend.get_texts(), fontsize=LEGEND_SIZE)
    legend.get_title().set_fontsize(LEGEND_TITLE_SIZE)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)


# Combined ROC Plot
def plot_combined_roc_common_ax(
    ax, results, fontsize, label_map, line_styles, model_colors
):
    """
    Plot combined ROC curves for all methods and datasets on a common axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        results (dict): Nested dict of results[dataset][method] with 'fpr' and 'tpr'.
        fontsize (int): Font size for labels and ticks.
        label_map (dict): Mapping from dataset key to label string.
        line_styles (dict): Mapping from dataset label to line style.
        model_colors (dict): Mapping from model name to color.
    """
    combined_handles = {}
    for method in model_colors.keys():
        for dataset in label_map.keys():
            fpr = results[dataset][method]["fpr"][0]
            tpr = results[dataset][method]["tpr"][0]
            auc_value = auc(fpr, tpr)
            style_key = label_map.get(dataset, dataset)
            current_linestyle = line_styles.get(style_key, "solid")
            ax.plot(
                fpr,
                tpr,
                color=model_colors[method],
                linestyle=current_linestyle,
                linewidth=bold_linewidth,
            )
            combined_label = f"{legend_map[style_key]} {model_legend_map[method]} AUC: {auc_value:.2f}"
            combined_handles[combined_label] = Line2D(
                [0],
                [0],
                color=model_colors[method],
                linestyle=current_linestyle,
                linewidth=bold_linewidth,
                label=combined_label,
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=fontsize)
    legend = ax.legend(
        handles=list(combined_handles.values()),
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="lower right",
        handlelength=LEGEND_HANDLELENGTH,
    )
    plt.setp(legend.get_texts(), fontsize=LEGEND_SIZE)
    legend.get_title().set_fontsize(LEGEND_TITLE_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.grid(True)


# Batch Time vs Batch Size Plot
def plot_batch_time_vs_batch_size(
    ax, results, label_map, model_colors, line_styles, fontsize=10
):
    """
    Plot average inference speed (batch time) vs. batch size for each model and dataset.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        results (dict): Nested dict of results[dataset][model][batch_size] with 'batch_times'.
        label_map (dict): Mapping from dataset key to label string.
        model_colors (dict): Mapping from model name to color.
        line_styles (dict): Mapping from dataset label to line style.
        fontsize (int, optional): Font size for labels and ticks. Defaults to 10.
    """
    combined_handles = {}
    found_any = False
    for model in model_colors.keys():
        for dataset in label_map.keys():
            batch_size_dict = results[dataset][model]
            batch_sizes = []
            avg_batch_times = []
            for bs, res in batch_size_dict.items():
                try:
                    bs_int = int(bs)
                except Exception:
                    continue
                if "batch_times" in res:
                    batch_times = np.array(res["batch_times"])
                    if len(batch_times) > 0:
                        batch_sizes.append(bs_int)
                        avg_batch_times.append(np.mean(batch_times))
            if batch_sizes:
                found_any = True
                sorted_idx = np.argsort(batch_sizes)
                batch_sizes = np.array(batch_sizes)[sorted_idx]
                avg_batch_times = np.array(avg_batch_times)[sorted_idx]
                ax.plot(
                    batch_sizes,
                    avg_batch_times,
                    marker="o",
                    color=model_colors[model],
                    linestyle=line_styles[label_map[dataset]],
                    linewidth=bold_linewidth,
                )
                combined_label = (
                    f"{legend_map[label_map[dataset]]} {model_legend_map[model]}"
                )
                if combined_label not in combined_handles:
                    combined_handles[combined_label] = Line2D(
                        [0],
                        [0],
                        color=model_colors[model],
                        linestyle=line_styles[label_map[dataset]],
                        linewidth=bold_linewidth,
                        label=combined_label,
                    )
            else:
                print(f"No valid batch_times for {dataset} {model}")
    ax.set_xlabel("Batch Size", fontsize=fontsize)
    ax.set_ylabel("Inference Speed (sec)", fontsize=fontsize)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(fontsize)
    legend = ax.legend(
        handles=list(combined_handles.values()),
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="upper left",
        handlelength=LEGEND_HANDLELENGTH,
    )
    plt.setp(legend.get_texts(), fontsize=LEGEND_SIZE)
    legend.get_title().set_fontsize(LEGEND_TITLE_SIZE)
    if not found_any:
        print("No batch time data found for any model/dataset.")
    ax.set_box_aspect(1)


# ----------------------
# Generate and Save Plots
# ----------------------

# Precision Plot
fig, ax = get_common_ax(figsize=(8, 8))
plot_precision_common_ax(
    ax, results_m, model_colors, line_styles, label_map, fontsize=size
)
fig.savefig(
    path.join(PROJECT_ROOT, "data", "mar_Precision_plot.pdf"),
    bbox_inches="tight",
    format="pdf",
)
plt.show()

# F1 Plot
fig, ax = get_common_ax(figsize=(8, 8))
plot_f1_common_ax(ax, results_m, model_colors, line_styles, label_map, fontsize=size)
fig.savefig(
    path.join(PROJECT_ROOT, "data", "mar_F1_plot.pdf"),
    bbox_inches="tight",
    format="pdf",
)
plt.show()

# # Inference Plot
# fig, ax = get_common_ax(figsize=(8, 8))
# plot_infer_common_ax(
#     ax,
#     results_m,
#     label_map,
#     fontsize=size,
#     model_colors=model_colors,
#     line_styles=line_styles,
# )
# fig.tight_layout()  # Adjust layout if necessary
# fig.savefig(
#     path.join(PROJECT_ROOT, "data", "mar_inference_plot.pdf"),
#     bbox_inches="tight",
#     format="pdf",
# )
# plt.show()

# Combined CDF Plot
fig, ax = get_common_ax(figsize=(8, 8))
plot_combined_cdf_common_ax(
    ax,
    results_m,
    fontsize=size,
    label_map=label_map,
    line_styles=line_styles,
    model_colors=model_colors,
)
fig.savefig(
    path.join(PROJECT_ROOT, "data", "mar_combined_cdf_plot.pdf"),
    bbox_inches="tight",
    format="pdf",
)
plt.show()

# Combined ROC Plot
fig, ax = get_common_ax(figsize=(8, 8))
plot_combined_roc_common_ax(
    ax,
    results_m,
    fontsize=size,
    label_map=label_map,
    line_styles=line_styles,
    model_colors=model_colors,
)
fig.savefig(
    path.join(PROJECT_ROOT, "data", "mar_combined_roc_plot.pdf"),
    bbox_inches="tight",
    format="pdf",
)
plt.show()

# Batch Time vs Batch Size Plot
fig, ax = get_common_ax(figsize=(8, 8))
plot_batch_time_vs_batch_size(
    ax, batch_results, label_map, model_colors, line_styles, fontsize=size
)
ax.set_xlim(2, 20)  # Adjust to your batch size range
ax.set_ylim(0, 0.01)  # Adjust to your data range
fig.tight_layout()
fig.savefig(
    path.join(PROJECT_ROOT, "data", "mar_inference_plot.pdf"),
    bbox_inches="tight",
    format="pdf",
)
plt.show()
