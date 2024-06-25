import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
plt.rcParams["font.family"] = "Times New Roman"


model_styles = {
    "MobileViG": ("#0C5DA5", "*"),
    "FastViT": ("#00B945", "p"),
    "EfficientFormerV2": ("#5580b0", "H"),
    "EfficientFormer": ("c", "P"),
    "SwiftFormer": ("g", "X"),
    "RepViT": ("#377e7f", "o"),
    "RepNeXt": ("r", "s"),
    "MobileOne": ("y", "x"),
    "StarNet": ("#333333", "."),
}

models_with_distill = {
    'MobileViG': ([1.27, 1.50, 1.86, 2.87], [75.7, 78.2, 80.6, 82.6], ['Ti', 'S', 'M', 'B'], '8'), 
    'FastViT': ([0.89, 1.33, 1.67, 2.78], [76.7, 80.3, 81.9, 83.4], ['T8', 'T12', 'SA12', 'SA24'], 'p'), 
    'SwiftFormer': ([1.00, 1.16, 1.62, 2.99], [75.7, 78.5, 80.9, 83.0], ['XS', 'S', 'L1', 'L3'], 'o'), 
    'EfficientFormerV2': ([0.91, 1.06, 1.63, 2.75], [75.7, 79.0, 81.6, 83.3], ['S0', 'S1', 'S2', 'L'], '+'), 
    'EfficientFormer': ([1.42, 2.80], [79.2, 82.4], ['L1', 'L3'], 'P'),
    'RepViT': ([0.89, 1.02, 1.13, 1.51, 2.24], [78.7, 80.0, 80.7, 82.3, 83.3], ['M0.9', 'M1.0', 'M1.1', 'M1.5', 'M2.3'], 'D'), 
    'RepNeXt': ([0.86, 1.00, 1.11, 1.48, 2.20], [78.8, 80.1, 80.7, 82.3, 83.3], ['M1', 'M2', 'M3', 'M4', 'M5'], '^')
}

models_without_distill = {
    "MobileOne": ([0.89, 1.14, 1.31, 1.73], [75.9, 77.4, 78.1, 79.4], ['S1', 'S2', 'S3', 'S4'], "o"),
    "StarNet": ([0.98, 1.11], [77.4, 78.4], ['S3', 'S4'], "o"),
    "RepNeXt": ([0.87, 1.01, 1.12, 1.48, 2.20], [77.5, 78.9, 79.4, 81.2, 82.4], ['M1', 'M2', 'M3', 'M4', 'M5'], "P"),
}

models_with_distill = dict(reversed(list(models_with_distill.items())))
models_without_distill = dict(reversed(list(models_without_distill.items())))

plt.figure(figsize=(10, 8))
ax = plt.gca()
for name, (x, y, n, m) in models_with_distill.items():
    plt.plot(x, y, marker=model_styles[name][1], label=name, color=model_styles[name][0])

for name, (x, y, n, m) in models_without_distill.items():
    plt.plot(x, y, "--", marker=model_styles[name][1], label=name, alpha=0.8, color=model_styles[name][0])

plt.ylim(75.5, 83.5)  
plt.xticks(np.arange(0, 6, 1))  
plt.xlim(0.8, 3.2)  

custom_lines = [Line2D([0], [0], color='#212121', lw=2), Line2D([0], [0], dashes=(2, 2, 2, 2), color='#212121', lw=2)]
leg=ax.legend(custom_lines, ['With distillation', 'Without distillation'], loc="upper left", fontsize=14, frameon=False)

plt.legend(fontsize=18)
ax.add_artist(leg)

ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.125))

def minor_tick_formatter(value, index):
    if value in range(77, 87, 2):
        return int(value)

ax.yaxis.set_minor_formatter(FuncFormatter(minor_tick_formatter))
ax.tick_params(axis='y', which='minor', pad=5)

plt.grid(which='major', linestyle='-', linewidth='1', color='lightgrey')
plt.grid(which='minor', linestyle='--', linewidth='0.5', color='lightgrey', alpha=0.8)

plt.xlabel('Latency (ms)', fontsize=22, fontproperties=FontProperties("Times New Roman"))
plt.ylabel('ImageNet-1K Top-1 Accuracy (%)', fontsize=22, fontproperties=FontProperties("Times New Roman"))

plt.xticks(fontproperties=FontProperties("Times New Roman"))
plt.yticks(fontproperties=FontProperties("Times New Roman"))
plt.savefig("figures/latency.pdf", format="pdf", bbox_inches='tight', transparent=True, dpi=600)
plt.show()
