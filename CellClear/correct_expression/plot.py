from matplotlib import pyplot as plt
import numpy as np


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
