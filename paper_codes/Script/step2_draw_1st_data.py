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


def draw_fig_2AandB():
    for name in samples:
        for method in methods:
            adata_files = [f'filtered_h5ad/{method}_{name}.h5ad' for method in methods]
            adatas = [sc.read(file) for file in adata_files]
            for adata in adatas:
                adata.uns['celltype_colors'] = color

        # draw figure A
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), gridspec_kw={"wspace": 0.2})
        for ax, adata, title in zip(axes, adatas, methods):
            if title == 'SoupX':
                legend_loc = 'right margin'
            else:
                legend_loc = 'none'
            sc.pl.umap(adata, color='celltype', legend_loc=legend_loc, title=title, show=False, ax=ax)

        plt.savefig(f'figure/{name}_celltype_umap.pdf', dpi=300, bbox_inches="tight")
        plt.close()

        # draw figure B
        max_values = []
        min_values = []

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

        plt.savefig(f'figure/{name}_Slc34a1_vlnplot.pdf', dpi=300, bbox_inches="tight")
        plt.close()


########################################################################################################################
def draw_fig_sup1C():
    marker = {
        'CD_IC': ['Atp6v1g3', 'Atp6v1c2', 'Atp6v0d2'],
        'Endo': ['Egfl7', 'Emcn', 'Nrp1']
    }

    for name in samples:

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 3), gridspec_kw={"wspace": 0.2})
        ax_list = [ax1, ax2, ax3, ax4, ax5]

        for i in range(len(methods)):
            ax = ax_list[i]
            adata = sc.read(f'filtered_h5ad/{methods[i]}_{name}.h5ad')
            sc.pl.dotplot(adata, marker, groupby='celltype', dendrogram=False, ax=ax, show=False,
                          title=methods[i], layer='normalised')

        plt.savefig(f'figure/{name}_marker_vlnplot.pdf', dpi=300, bbox_inches="tight")
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

    plt.figure(figsize=(8, 8))
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
                        box_pairs=pairs, test='Wilcoxon', text_format='star', loc='inside')

    plt.savefig(f'figure/{name}_boxplot.pdf', dpi=300, bbox_inches="tight")
    plt.close()


def draw_fig_2C():
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
    final_pct_df.to_csv('metric/pct_metric.tsv', sep='\t')
    final_logfc_df = pd.concat(logfc_results)
    draw_boxplot(final_logfc_df, 'Log2FC')
    final_logfc_df.to_csv('metric/log2fc_metric.tsv', sep='\t')


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
        calculate_cont_per_cell(name)
        dfs = [read_and_process_file(f'contPerCell/{method}_{name}_contperCell.tsv', method, name) for method in
               methods]
        result = merge_dfs(dfs)
        result['Sample'] = name
        cont_df = pd.concat([cont_df, result])
    return cont_df


def draw_fig_2D():
    cont_df = calculate_and_concatenate(['rep1', 'rep2', 'rep3'])
    cont_df = cont_df.drop(columns=['cell'])
    cont_df = cont_df.melt(id_vars=['Sample'],
                           value_vars=['Genotype estimate', 'CellClear', 'CellBender', 'SoupX', 'DecontX'],
                           var_name='Method',
                           value_name='Contamination Estimate')

    plt.figure(figsize=(8, 8))
    ax = sns.boxplot(x='Sample', y='Contamination Estimate', hue='Method', data=cont_df, palette='Set3',
                     showfliers=False)
    plt.xlabel('Sample')
    plt.ylabel('Contamination Estimate')
    plt.legend(title='Method', loc='upper left')
    ax.set_xlabel('Sample', fontsize=15)
    ax.set_ylabel('Contamination Estimate', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.savefig(f'figure/Cont_estimate_boxplot.pdf', dpi=300, bbox_inches="tight")
    plt.close()


draw_fig_2AandB()
draw_fig_sup1C()
draw_fig_2C()
draw_fig_2D()
