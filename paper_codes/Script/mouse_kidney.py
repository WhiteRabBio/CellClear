from statannot import add_stat_annotation
from functools import reduce
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import warnings
import os
import seaborn as sns

warnings.filterwarnings("ignore")

color = [
    '#0067AA', '#FF7F00', '#00A23F', '#FF1F1D', '#A763AC', '#B45B5D', '#FF8AB6', '#B6B800', '#01C1CC',
    '#85D5F8', '#FFC981', '#C8571B', '#727272', '#EFC800', '#8A5626', '#502E91', '#59A4CE', '#344B2B',
    '#FBE29D', '#FDD6E6', '#849C8C', '#F07C6F', '#000101', '#FF4500', '#6959CD', '#FF8C00', '#ADFF2F',
]

samples = ['rep1', 'rep2', 'rep3']
methods = ['Uncorrected', 'CellClear', 'CellBender', 'DecontX', 'SoupX']

os.makedirs('figure', exist_ok=True)
os.makedirs('metric', exist_ok=True)


def draw_umap_and_vlnplot():
    for name in samples:
        adata_files = [f'filtered_h5ad/{method}_{name}.h5ad' for method in methods]
        adatas = [sc.read(file) for file in adata_files]
        for adata in adatas:
            adata.uns['celltype_colors'] = color

        if name == 'rep1':
            save_name = 'fig_4A_rep1'
        else:
            save_name = f'fig_S2A_{name}'
        # draw figure A
        fig, axes = plt.subplots(1, 5, figsize=(20, 3), gridspec_kw={"wspace": 0.2})
        for ax, adata, title in zip(axes, adatas, methods):
            if title == 'SoupX':
                legend_loc = 'right margin'
            else:
                legend_loc = 'none'
            sc.pl.umap(adata, color='celltype', legend_loc=legend_loc, title=title, show=False, ax=ax)

        plt.savefig(f'figure/{save_name}_celltype_umap.pdf', dpi=300, bbox_inches="tight")
        plt.close()

        # draw figure B
        max_values = []
        min_values = []

        if name == 'rep1':
            save_name = 'fig_4B_rep1'
        else:
            save_name = f'fig_S2B_{name}'

        for adata in adatas:
            layer_data = adata[:, 'Slc34a1'].layers['normalised']
            max_values.append(np.max(layer_data))
            min_values.append(np.min(layer_data))

        max_value = max(max_values)
        min_value = min(min_values)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 3), gridspec_kw={"wspace": 0.2})
        ax_list = [ax1, ax2, ax3, ax4, ax5]

        for ax, data, title in zip(ax_list, adatas, methods):
            sc.pl.violin(data, keys='Slc34a1', groupby='celltype', use_raw=False, rotation=90, inner='box', size=0,
                         layer='normalised', show=False, ax=ax)
            if title == 'Uncorrected':
                pass
            else:
                ax.set_ylabel('')
            ax.set_title(title)
            ax.set_ylim([min_value, max_value])

        plt.savefig(f'figure/{save_name}_Slc34a1_vlnplot.pdf', dpi=300, bbox_inches="tight")
        plt.close()


########################################################################################################################
def draw_others_marker_vlnplot():
    marker = {
        'CD_IC': ['Atp6v1g3', 'Atp6v1c2', 'Atp6v0d2'],
        'Endo': ['Egfl7', 'Emcn', 'Nrp1']
    }

    for name in ['rep1']:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 3), gridspec_kw={"wspace": 0.2})
        ax_list = [ax1, ax2, ax3, ax4, ax5]

        for i in range(len(methods)):
            ax = ax_list[i]
            adata = sc.read(f'filtered_h5ad/{methods[i]}_{name}.h5ad')
            sc.pl.dotplot(adata, marker, groupby='celltype', dendrogram=False, ax=ax, show=False,
                          title=methods[i], layer='normalised')

        plt.savefig(f'figure/fig_S2C_{name}_marker_vlnplot.pdf', dpi=300, bbox_inches="tight")
        plt.close()


########################################################################################################################
def calculate_percentage(data, data_df, gene_markers):
    exp = data_df[gene_markers]
    exp['celltype'] = data.obs['celltype']
    exp = exp[~exp['celltype'].isin(['PT'])]
    gene_dict = {g: len(exp[exp[g] > 0]) / len(exp) for g in gene_markers}

    return gene_dict


def calculate_log2fc(data, gene_markers, name, method):
    if 'rank_genes_groups_celltype' not in data.uns:
        if 'log1p' in data.uns:
            data.uns['log1p']['base'] = None
        sc.tl.rank_genes_groups(data, layer="normalised", groupby='celltype',
                                method='wilcoxon', tie_correct=False, pts=True,
                                use_raw=False, key_added="rank_genes_groups_celltype")
        print(f'write deg to {method}_{name}')
        data.write(f'filtered_h5ad/{method}_{name}.h5ad')

    de_i = sc.get.rank_genes_groups_df(data, group='PT', key="rank_genes_groups_celltype")
    de_i = de_i[de_i['names'].isin(gene_markers)]
    logfc_dict = dict(zip(de_i['names'], de_i['logfoldchanges']))

    return logfc_dict


def process_sample(name, method, gene_markers):
    data = sc.read(f'filtered_h5ad/{method}_{name}.h5ad')
    data.X = data.layers['normalised'].copy()
    data_df = data.to_df()
    pct_dict = calculate_percentage(data, data_df, gene_markers)
    logfc_dict = calculate_log2fc(data, gene_markers, name, method)

    return pd.DataFrame(pct_dict, index=[f"{name}_{method}"]), \
        pd.DataFrame(logfc_dict, index=[f"{name}_{method}"])


def draw_boxplot(box_df, name):
    pairs = []
    for sample in ['rep1', 'rep2', 'rep3']:
        for method in ['CellClear', 'CellBender', 'DecontX', 'SoupX']:
            pairs.append(((sample, 'Uncorrected'), (sample, method)))

    data_melted = box_df.reset_index().melt(id_vars='index', var_name='Gene', value_name='Value')
    data_melted[['Sample', 'Method']] = data_melted['index'].str.split('_', expand=True)
    data_melted.drop(columns='index', inplace=True)

    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x='Sample', y='Value', hue='Method', data=data_melted, palette='Set3', showfliers=False)
    if name == 'PCT':
        plt.title(f'Boxplot of PTs Markers {name} Values in non-PTs cells for Each Sample and Method')
    else:
        plt.title(f'Boxplot of PTs Markers {name} Values in PTs for Each Sample and Method')

    plt.xlabel('Sample', fontsize=15)
    plt.ylabel(f'{name}', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.legend(title='Method', loc='upper left')

    add_stat_annotation(ax, data=data_melted, x='Sample', y='Value', hue='Method',
                        box_pairs=pairs, test='Mann-Whitney', text_format='star', loc='inside')

    plt.savefig(f'figure/fig_4C_{name}_boxplot.pdf', dpi=300, bbox_inches="tight")
    plt.close()


def draw_logfc_pct_boxlot():
    topPTsMarker = ['Slc34a1', 'Miox', 'Pck1', 'Slc4a4', 'Ttc36', 'Lrp2', 'Fbp1', 'Cyp2e1', 'Fut9', 'Khk']
    pct_results = []
    logfc_results = []

    for name in ['rep1', 'rep2', 'rep3']:
        for method in ['Uncorrected', 'CellClear', 'CellBender', 'DecontX', 'SoupX']:
            pct_df, logfc_df = process_sample(name, method, topPTsMarker)
            pct_results.append(pct_df)
            logfc_results.append(logfc_df)

    final_pct_df = pd.concat(pct_results)
    draw_boxplot(final_pct_df, 'PCT')
    final_pct_df.to_csv('metric/mouse_kidney_pct_metric.tsv', sep='\t')
    final_logfc_df = pd.concat(logfc_results)
    draw_boxplot(final_logfc_df, 'Log2FC')
    final_logfc_df.to_csv('metric/mouse_kidney_log2fc_metric.tsv', sep='\t')


########################################################################################################################
def calculate_cont_per_cell(name):
    ref = sc.read_10x_h5(f'matrix/{name}/Uncorrected/filtered_feature_bc_matrix.h5')
    ref_exp = ref.to_df().T

    for method in ['CellClear', 'CellBender', 'DecontX', 'SoupX']:
        tmp_column_name = [f'{method}_{name}_' + i for i in ref_exp.columns]
        tmp = ref_exp.copy()
        tmp.columns = tmp_column_name
        data = sc.read(f'h5ad/{method}_{name}.h5ad')
        data_exp = data.to_df().T
        inter_bc = list(set(data_exp.columns) & set(tmp.columns))
        data_exp = data_exp[inter_bc]
        tmp = tmp[inter_bc]
        per_cell_cont = pd.DataFrame({
            'cell': data_exp.columns,
            'cont': 1 - (data_exp.sum() / tmp.sum())
        })
        per_cell_cont.to_csv(f'contPerCell/{method}_{name}_contperCell.tsv', sep='\t', index=None)


def read_and_process_file(filepath, method, name):
    df = pd.read_csv(filepath, sep='\t')
    if method != 'Uncorrected':
        df['cell'] = df['cell'].str.replace(f'{method}_{name}_', '')
        df.columns = ['cell', method]
    else:
        df = df[['cell', 'contPerCell_binom']]
        df.columns = ['cell', 'Genotype estimate']
    return df


def merge_dfs(dfs):
    return reduce(lambda left, right: pd.merge(left, right, on='cell', how='inner'), dfs)


def calculate_and_concatenate(names):
    cont_df = pd.DataFrame()
    methods = ['Uncorrected', 'CellClear', 'CellBender', 'SoupX', 'DecontX']
    for name in names:
        # calculate_cont_per_cell(name)
        dfs = [read_and_process_file(f'contPerCell/{method}_{name}_contperCell.tsv', method, name) for method in
               methods]
        result = merge_dfs(dfs)
        result['Sample'] = name
        cont_df = pd.concat([cont_df, result])
    return cont_df


def calculate_RMSLE(cont_df, names):
    from scipy.stats import kendalltau
    methods = ['CellClear', 'CellBender', 'SoupX', 'DecontX']
    cont_dict = {}
    for name in names:
        cont_dict[name] = {}
        df = cont_df[cont_df['Sample'] == name]
        for method in methods:
            tau, _ = kendalltau(df['Genotype estimate'], df[method])
            rmsle = np.sqrt(np.mean((np.log1p(df['Genotype estimate']) - np.log1p(df[method])) ** 2))
            cont_dict[name][method] = {'tau': tau, 'rmsle': rmsle}
    result = pd.DataFrame.from_dict({(rep, method): values
                                     for rep, methods in cont_dict.items()
                                     for method, values in methods.items()},
                                    orient='index')
    return result


def draw_cont_estimate():
    cont_df = calculate_and_concatenate(['rep1', 'rep2', 'rep3'])
    rmsle = calculate_RMSLE(cont_df, ['rep1', 'rep2', 'rep3'])
    rmsle = rmsle.reset_index()
    rmsle.columns = ['Sample', 'method', 'tau', 'rmsle']
    cont_df = cont_df.drop(columns=['cell'])
    cont_df = cont_df.melt(id_vars=['Sample'],
                           value_vars=['Genotype estimate', 'CellClear', 'CellBender', 'SoupX', 'DecontX'],
                           var_name='Method',
                           value_name='Contamination Estimate')

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='Sample', y='Contamination Estimate', hue='Method', data=cont_df, palette='Set3',
                     showfliers=False)
    plt.xlabel('Sample')
    plt.ylabel('Contamination Estimate')
    plt.legend(title='Method', loc='upper left')
    ax.set_title("Contamination Estimate Values by Replicate and Method")
    ax.set_xlabel('Sample', fontsize=15)
    ax.set_ylabel('Contamination Estimate', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.savefig(f'figure/fig_4D_Cont_estimate_boxplot.pdf', dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Setting colors for each method for clarity
    methods = ["CellClear", "CellBender", "SoupX", "DecontX"]
    palette = sns.color_palette("Set3", 5)
    colors = [palette[1], palette[2], palette[3], palette[4]]

    # Creating the bar width and x positions for each replicate
    bar_width = 0.2
    x = range(len(rmsle['Sample'].unique()))

    # Plot each method for each replicate with different colors
    for i, method in enumerate(methods):
        subset = rmsle[rmsle['method'] == method]
        ax.bar([pos + i * bar_width for pos in x], subset['rmsle'], width=bar_width, color=colors[i], label=method,
               edgecolor="black")

    # Adding labels and title
    ax.set_xlabel("Replicates", fontsize=15)
    ax.set_ylabel("RMSLE", fontsize=15)
    ax.set_title("RMSLE Values by Replicate and Method")
    ax.set_xticks([pos + 1.5 * bar_width for pos in x])
    ax.set_xticklabels(rmsle['Sample'].unique())
    ax.legend(title="Methods", loc='upper left')

    plt.tight_layout()
    plt.savefig(f'figure/fig_4D_Cont_estimate_rmsle_barplot.pdf', dpi=300, bbox_inches="tight")
    plt.close()
