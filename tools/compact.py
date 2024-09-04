from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "Times New Roman"

Model = namedtuple("Model", ["name", "params", "accs", "color", "marker", "line"])
Experiment = namedtuple("Experiment", ["dataset", "models", "offset"])

experiments = [
    Experiment(
        dataset="CIFAR-100",
        models=[
            Model(name="RepNeXt-M0", params=2.039, accs=[74.53, 80.14, 81.73, 81.97], color="#0C5DA5", marker="o", line="-"),
            Model(name="RepNeXt-M0E", params=2.252, accs=[73.56, 79.79, 81.89, 82.07], color="#474747", marker="o", line="--"),
            Model(name="RepViT-M0.6", params=2.179, accs=[71.73, 78.45, 80.16, 80.60], color="#00B945", marker="o", line="-"),
        ],
        offset=0.3
    ),
    Experiment(
        dataset="ImageNet-1K",
        models=[
            Model(name="RepNeXt-M0", params=2.328, accs=[70.14, 71.93, 72.56, 72.78], color="#0C5DA5", marker="o", line="-"),
            Model(name="RepNeXt-M0E", params=2.541, accs=[69.69, 71.57, 72.18, 72.53], color="#474747", marker="o", line="--"),
            Model(name="RepViT-M0.6", params=2.468, accs=[69.29, 71.63, 72.34, 72.87], color="#00B945", marker="o", line="-"),
        ],
        offset=0.1

    )
]

def plot_model_performance(models, dataset="CIFAR-100", offset=0.3):
    plt.figure()
    ax = plt.gca()
    legend_models, legend_params = [], []

    for order, model in enumerate(models):
        x, y = list(range(100, (len(model.accs)+1)*100, 100)), model.accs
        ax.plot(x, y, marker=model.marker, linestyle=model.line, markersize=round(model.params, 1)*4-4, label=model.name, color=model.color, zorder=len(models)-order+2)
        if model.name in ["RepNeXt-M0", "RepViT-M0.6"]:
            for epoch, acc in zip(x, y):
                ax.annotate(f"{acc:.2f}", (epoch-10, acc+offset), annotation_clip=True, color=model.color, zorder=100)
        kwargs = dict(marker=model.marker, linestyle=model.line, color='w', markerfacecolor=model.color, markersize=12)
        legend_models.append(Line2D([0], [0], label=model.name.ljust(23), **kwargs))
        legend_params.append(Line2D([0], [0], label=f"{round(model.params, 1)}M", **kwargs))

    lm = ax.legend(handles=legend_models, loc="lower right", fontsize=14, frameon=False, handletextpad=0.1)
    lp = ax.legend(handles=legend_params, handlelength=0, loc="lower right", fontsize=14, frameon=False, handletextpad=0.1)
    for item in lp.legend_handles:
        item.set_visible(False)
    ax.add_artist(lm)
    ax.set_ylabel(f"{dataset} Top-1 Accuracy (%)", fontsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    is_cifar = dataset == "CIFAR-100"
    ax.yaxis.set_major_locator(MultipleLocator(1+int(is_cifar)))
    ax.yaxis.set_minor_locator(MultipleLocator(1+int(is_cifar)))
    ax.grid(alpha=0.8, linestyle="-", linewidth=0.5)
    ax.grid(which='minor', linestyle='-', linewidth='0.5', alpha=0.8)
    ax.set_xlabel('Epochs', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"../figures/compact_models_{dataset.replace('-', '_').lower()}.png", bbox_inches='tight', transparent=False, dpi=300)


def plot_experiments(experiments):
    fig, axs = plt.subplots(nrows=len(experiments), ncols=1, sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.04}, figsize=(10, 8))
    for ax, experiment in zip(axs, experiments):
        legend_models, legend_params = [], []
        for order, model in enumerate(experiment.models):
            x, y = list(range(100, (len(model.accs)+1)*100, 100)), model.accs
            ax.plot(x, y, marker=model.marker, linestyle=model.line, markersize=round(model.params, 1)*4-4, label=model.name, color=model.color, zorder=len(experiment.models)-order+2)
            if model.name == "RepNeXt-M0":
                for epoch, acc in zip(x, y):
                    ax.annotate(f"{acc:.2f}", (epoch-10, acc+experiment.offset), annotation_clip=True, color=model.color, zorder=100)
            kwargs = dict(marker=model.marker, linestyle=model.line, color='w', markerfacecolor=model.color, markersize=12)
            legend_models.append(Line2D([0], [0], label=model.name.ljust(23), **kwargs))
            legend_params.append(Line2D([0], [0], label=f"{round(model.params, 1)}M", **kwargs))

        lm = ax.legend(handles=legend_models, loc="lower right", fontsize=14, frameon=False, handletextpad=0.1)
        lp = ax.legend(handles=legend_params, handlelength=0, loc="lower right", fontsize=14, frameon=False, handletextpad=0.1)
        for item in lp.legend_handles:
            item.set_visible(False)
        ax.add_artist(lm)
        ax.set_ylabel(f"{experiment.dataset} Top-1 Accuracy (%)", fontsize=14, labelpad=2)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        is_cifar = experiment.dataset == "CIFAR-100"
        ax.yaxis.set_major_locator(MultipleLocator(1+int(is_cifar)))
        ax.yaxis.set_minor_locator(MultipleLocator(1+int(is_cifar)))
        ax.grid(alpha=0.8, linestyle="-", linewidth=0.5)
        ax.grid(which='minor', linestyle='-', linewidth='0.5', alpha=0.8)
    ax.set_xlabel('Epochs', fontsize=12, labelpad=-4)
    plt.savefig(f"../figures/compact_model_experiments.png", bbox_inches='tight', transparent=False, dpi=300)

plot_experiments(experiments)

for experiment in experiments:
    plot_model_performance(models=experiment.models, dataset=experiment.dataset, offset=experiment.offset)
