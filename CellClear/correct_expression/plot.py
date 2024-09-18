import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
from copy import copy
from pathlib import Path
from matplotlib import pyplot as plt


def plot_distance_dot(dist, name, output, top=3):
    up_genes = dist.sort_values(by='distance', ascending=False)['Gene'][:top].tolist()
    down_genes = dist.sort_values(by='distance', ascending=True)['Gene'][:top].tolist()
    show_genes = up_genes + down_genes
    plt.scatter(dist['mean_expr'], dist['bg_mean_expr'], label='Background mean', color='blue', marker='o', s=20)
    plt.scatter(dist['mean_expr'], dist['fit'], label='Fit', color='red', marker='s', s=20)
    plt.scatter(dist['mean_expr'], dist['distance'], label='Distance', color='black', marker='x', s=15)
    plt.title(f'Cluster {name}')
    mean_expr_array = np.array(dist['mean_expr'])
    distance_array = np.array(dist['distance'])
    fit_params = np.polyfit(mean_expr_array, distance_array, 2)
    fit_line = np.polyval(fit_params, np.sort(mean_expr_array))
    plt.plot(np.sort(mean_expr_array), fit_line, label='Quadratic fit', color='green')
    plt.legend()
    for i in show_genes:
        plt.text(dist.loc[dist['Gene'] == i, 'mean_expr'].iloc[0] + 0.01, dist.loc[dist['Gene'] == i, 'distance'].iloc[0], i,
                 fontsize=8)
    plt.savefig(f'{output}/cluster_{name}_distance_dot.png')
    plt.close()


def plot_usages(counts, usages, spectra, topic_list, rescue_cont_genes, output):
    reds = copy(matplotlib.cm.Reds)
    reds.set_under('#a9a9a9')
    adata = counts.copy()

    for topic in topic_list:
        Path(f'{output}/{topic}').mkdir(parents=True, exist_ok=True)
        adata.obs[topic] = usages[topic]
        sc.pl.umap(adata, color=topic, color_map=reds, vmin=0.0001)
        plt.savefig(f'{output}/{topic}/{topic}_usage.png', bbox_inches='tight')
        plt.close()

        for gene in spectra.loc[rescue_cont_genes, topic].sort_values()[-10:].index:
            sc.pl.umap(adata, color=gene, color_map=reds, vmin=0.0001, layer='normalised')
            plt.savefig(f'{output}/{topic}/{topic}_{gene}_normalised_expression.png', bbox_inches='tight')
            plt.close()
