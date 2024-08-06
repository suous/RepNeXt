from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "Times New Roman"

Model = namedtuple("Model", ["name", "params", "accs", "color", "marker", "line"])
models = [
    Model(name="RepNeXt-M0", params=2.3, accs=[74.53, 80.14, 81.73, 81.97], color="#0C5DA5", marker="o", line="-"),
    Model(name="RepNeXt-M0E", params=2.5, accs=[73.56, 79.79, 81.89, 82.07], color="#474747", marker="o", line="--"),
    Model(name="RepViT-M0.6", params=2.5, accs=[71.73, 78.45, 80.16, 80.60], color="#00B945", marker="o", line="-"),
    Model(name="StarNet-s1", params=2.9, accs=[70.37, 77.26, 79.26, 80.61], color="#FF2C00", marker="o", line="-"),
]

def plot_model_performance(models, dataset="CIFAR-100", offset=0.3):
    plt.figure()
    ax = plt.gca()
    legend_models, legend_params = [], []

    for order, model in enumerate(models):
        x, y = list(range(100, 500, 100)), model.accs
        ax.plot(x, y, marker=model.marker, linestyle=model.line, markersize=model.params*4-4, label=model.name, color=model.color, zorder=len(models)-order+2)
        if model.name in ["RepNeXt-M0", "RepViT-M0.6"]:
            for epoch, acc in zip(x, y):
                ax.annotate(f"{acc:.2f}", (epoch-10, acc+offset), annotation_clip=True, color=model.color)
        kwargs = dict(marker=model.marker, linestyle=model.line, color='w', markerfacecolor=model.color, markersize=12)
        legend_models.append(Line2D([0], [0], label=model.name.ljust(23), **kwargs))
        legend_params.append(Line2D([0], [0], label=f"{model.params}M", **kwargs))

    lm = ax.legend(handles=legend_models, loc="lower right", fontsize=14, frameon=False, handletextpad=0.1)
    lp = ax.legend(handles=legend_params, handlelength=0, loc="lower right", fontsize=14, frameon=False, handletextpad=0.1)
    for item in lp.legend_handles:
        item.set_visible(False)
    ax.add_artist(lm)
    ax.set_ylabel(f"{dataset} Top-1 Accuracy (%)", fontsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.grid(alpha=0.8, linestyle="-", linewidth=0.5)
    ax.grid(which='minor', linestyle='-', linewidth='0.5', alpha=0.8)
    ax.set_xlabel('Epochs', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"compact_models_{dataset.replace('-', '_').lower()}.png", bbox_inches='tight', transparent=False, dpi=300)

plot_model_performance(models)