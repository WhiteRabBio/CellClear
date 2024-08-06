import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from copy import copy

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings('ignore')

color = [
    '#0067AA', '#FF7F00', '#00A23F', '#FF1F1D', '#A763AC', '#B45B5D', '#FF8AB6', '#B6B800', '#01C1CC',
    '#85D5F8', '#FFC981', '#C8571B', '#727272', '#EFC800', '#8A5626', '#502E91', '#59A4CE', '#344B2B',
    '#FBE29D', '#FDD6E6', '#849C8C', '#F07C6F', '#000101', '#FF4500', '#6959CD', '#FF8C00', '#ADFF2F',
]

marker_genes_dict = {
    'Monocytes/neutrophils': ['S100A8', 'S100A9', 'S100A12'],
    'Monocytes/pDCs': ['LYZ', 'CST3', 'FCN1'],
    'T': ['IL32', 'TRAC'],
    'T CD4+ naive/TE': ['CCR7'], 'Treg': ['FOXP3'],
    'B': ['IGHD', 'CD79A'],
    'B naive': ['FCER2'],
    'B memory': ['TNFRSF13B'],
    'T CD8+': ['CD8A', 'CD8B'],
    'T cytotoxic': ['NKG7', 'GNLY'],
    'T γδ': 'TRGC1', 'MAIT': 'SLC4A10',
    'NK': ['KLRF1', 'SPON2'],
    'Monocyte NC/I': ['FCGR3A'],
    'Progenitor': ['PPBP'],
    'Baso./neutro./progenitor': ['SDPR', 'CLU'],
    'pDCs': ['LILRA4']
}
celltype = [
    'T CD4+ naive',
    'T CD8+ TE/EM',
    'T CD8+ naive',
    'T γδ/MAIT',
    'Treg 4/helper T',
    'NK',
    'B memory',
    'B naive',
    'Monocyte C',
    'Monocyte NC',
    'Progenitors',
    'pDCs'
]
sample = 'pbmc8k'
methods = ['Uncorrected', 'CellClear', 'CellBender']


def annotation():
    cluster2annotation = {'0': 'Monocyte C', '1': 'T CD4+ naive',
                          '2': 'Treg 4/helper T', '3': 'T CD8+ naive',
                          '4': 'B naive', '5': 'T CD8+ TE/EM',
                          '6': 'T γδ/MAIT', '7': 'B memory',
                          '8': 'T CD4+ naive', '9': 'NK',
                          '10': 'Monocyte NC', '11': 'Progenitors', '12': 'pDCs'}
    adata = sc.read(
        'filtered_h5ad/Uncorrected_pbmc8k.h5ad')
    adata.obs['celltype'] = adata.obs['cluster'].map(cluster2annotation).astype('category')
    adata.obs['celltype'] = adata.obs['celltype'].cat.set_categories(celltype)
    adata.write(
        'filtered_h5ad/Uncorrected_pbmc8k.h5ad',
        compression='lzf')

    cluster2annotation = {'0': 'Monocyte C', '1': 'T CD4+ naive',
                          '2': 'T CD8+ naive', '3': 'Treg 4/helper T',
                          '4': 'B naive', '5': 'T CD8+ TE/EM',
                          '6': 'T γδ/MAIT', '7': 'B memory',
                          '8': 'NK', '9': 'T CD4+ naive',
                          '10': 'Monocyte NC', '11': 'Progenitors', '12': 'pDCs'}
    adata = sc.read(
        'filtered_h5ad/CellBender_pbmc8k.h5ad')
    adata.obs['celltype'] = adata.obs['cluster'].map(cluster2annotation).astype('category')
    adata.obs['celltype'] = adata.obs['celltype'].cat.set_categories(celltype)
    adata.write(
        'filtered_h5ad/CellBender_pbmc8k.h5ad',
        compression='lzf')

    cluster2annotation = {'0': 'Monocyte C', '1': 'T CD4+ naive',
                          '2': 'Treg 4/helper T', '3': 'T CD8+ naive',
                          '4': 'B naive', '5': 'T CD8+ TE/EM',
                          '6': 'B memory', '7': 'T γδ/MAIT',
                          '8': 'NK', '9': 'Monocyte NC',
                          '10': 'Progenitors', '11': 'pDCs'}
    adata = sc.read(
        'filtered_h5ad/CellClear_pbmc8k.h5ad')
    adata.obs['celltype'] = adata.obs['cluster'].map(cluster2annotation).astype('category')
    adata.obs['celltype'] = adata.obs['celltype'].cat.set_categories(celltype)
    adata.write('filtered_h5ad/CellClear_pbmc8k.h5ad',
                compression='lzf')


def integration_h5ad():
    adatas = []
    for method in methods:
        adata = sc.read(
            f'filtered_h5ad/{method}_{sample}.h5ad'
        )
        adata.obs['Sample ID'] = method
        adatas.append(adata)
    return adatas


def draw_fig3A(adatas):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(48, 12),
        gridspec_kw={'wspace': 0.2}
    )
    labels = {
        'Uncorrected': [
            '0: Monocyte C', '1: T CD4+ naive',
            '2: Treg 4/helper T', '3: T CD8+ naive',
            '4: B naive', '5: T CD8+ TE/EM',
            '6: T γδ/MAIT', '7: B memory',
            '8: T CD4+ naive', '9: NK',
            '10: Monocyte NC', '11: Progenitors', '12: pDCs'
        ],
        'CellBender': [
            '0: Monocyte C', '1: T CD4+ naive',
            '2: T CD8+ naive', '3: Treg 4/helper T',
            '4: B naive', '5: T CD8+ TE/EM',
            '6: T γδ/MAIT', '7: B memory',
            '8: NK', '9: T CD4+ naive',
            '10: Monocyte NC', '11: Progenitors', '12: pDCs'
        ],
        'CellClear': [
            '0: Monocyte C', '1: T CD4+ naive',
            '2: Treg 4/helper T', '3: T CD8+ naive',
            '4: B naive', '5: T CD8+ TE/EM',
            '6: B memory', '7: T γδ/MAIT',
            '8: NK', '9: Monocyte NC',
            '10: Progenitors', '11: pDCs'
        ]
    }
    for ax, adata, title in zip(axes, adatas, methods):
        method_label = labels[title]
        colors = [color[i] for i in range(len(method_label))]

        sc.pl.umap(
            adata,
            color='cluster',
            legend_loc='on data',
            show=False,
            legend_fontsize='large',
            ax=ax,
            title=title
        )
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('UMAP 1', fontsize=18)
        ax.set_ylabel('UMAP 2', fontsize=18)
        legend_handles = [
            mpatches.Patch(color=colors[i], label=method_label[i])
            for i in range(len(method_label))
        ]
        ax.legend(
            handles=legend_handles,
            bbox_to_anchor=(1, 1),
            loc='upper left',
            frameon=False,
            fontsize=12
        )

    plt.savefig(
        f'figure/{sample}_cluster_umap.pdf',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def draw_fig3B(adatas):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(48, 12),
        gridspec_kw={'wspace': 0.2}
    )
    plt.rcParams.update({'font.size': 18})

    for ax, adata, title in zip(axes, adatas, methods):
        sc.pl.dotplot(
            adata,
            marker_genes_dict,
            'celltype',
            dot_max=0.5,
            dendrogram=False,
            title=title,
            show=False,
            ax=ax,
            layer='normalised'
        )
        ax.title.set_size(18)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.legend(fontsize=18, title_fontsize=18) if ax.get_legend() else None

    plt.savefig(
        f'figure/{sample}_celltype_dotplot.pdf',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def draw_fig3C(adatas):
    gene_names = ['LYZ', 'HLA-DRA']

    for idx, gene in enumerate(gene_names):
        fig, axes = plt.subplots(
            1, 3,
            figsize=(48, 12),
            gridspec_kw={'wspace': 0.2}
        )

        plt.rcParams.update({'font.size': 18})
        reds = copy(matplotlib.cm.Reds)
        reds.set_under('#a9a9a9')

        for ax, adata, title in zip(axes, adatas, methods):
            sc.pl.umap(
                adata,
                color=gene,
                title=title,
                show=False,
                ax=ax,
                layer='normalised',
                color_map=reds,
                vmin=0.0001
            )
            ax.title.set_size(18)
            ax.xaxis.label.set_size(18)
            ax.yaxis.label.set_size(18)
            ax.legend(fontsize=18, title_fontsize=18) if ax.get_legend() else None
            if ax is axes[-1]:
                ax.text(1.2, 0.5, gene, transform=ax.transAxes,
                        fontsize=30, verticalalignment='center', rotation=270)

        plt.savefig(
            f'figure/{sample}_{gene}_umap.pdf',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


def plot_enrichment():
    # plot after enrichment analysis
    enrich_df = pd.DataFrame()
    for method in methods:
        for ct in celltype:
            df = pd.read_table(
                f"enrichment/{method}/{ct}/{ct}_up.GOALL_enrichment_sig.xls"
            )
            df = pd.DataFrame(
                {
                    'Description': df['Description'],
                    'Pvalue': -np.log10(df['pvalue'])
                }
            )
            df['celltype'] = ct
            df['method'] = method
            enrich_df = pd.concat([enrich_df, df])

    for ct in celltype:
        df = enrich_df[enrich_df['celltype'] == ct]
        common_descriptions = set(df[df['method'] == 'Uncorrected']['Description']) & \
                              set(df[df['method'] == 'CellBender']['Description']) & \
                              set(df[df['method'] == 'CellClear']['Description'])

        df = df[df['Description'].isin(common_descriptions)][:10]
        df.to_csv(
            f'enrichment/enrich_per_celltype/{ct}_enrichment.csv',
            sep=',',
            index=None
        )

    focus_celltype = ['T CD8+ TE_EM', 'T γδ_MAIT', 'Treg 4_helper T']
    fig, axes = plt.subplots(
        1, 3,
        figsize=(48, 12),
        gridspec_kw={'wspace': 0.2}
    )
    axes = axes.flatten()

    for idx, ct in enumerate(focus_celltype):
        df = pd.read_csv(f'enrichment/enrich_per_celltype/{ct}_enrichment.csv')
        ct1 = df['celltype'].tolist()[0]
        ax = axes[idx]
        sns.barplot(y='Description', x='Pvalue', hue='method', data=df, palette='Set3', ax=ax)
        sns.despine()
        if ct == 'Treg 4_helper T':
            ax.legend(title='Method', loc='lower right')
        else:
            ax.legend_.remove()
        ax.set_title(ct1, fontsize=18)
        ax.set_ylabel('')
        ax.set_xlabel('-log10 of P values', fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18)

    plt.savefig(
        f'figure/{sample}_top10Pathway.pdf',
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_cellphonedb():
    # plot after cellphonedb analysis
    fig, axes = plt.subplots(
        1, 3,
        figsize=(48, 12),
        gridspec_kw={'wspace': 0.2}
    )
    axes = axes.flatten()

    for idx, title in enumerate(methods):
        df = pd.read_table(
            f'cellphonedb/{title}/{title}/{title}_count_network.xls'
        )
        pivot_table = df.pivot_table(
            index='Celltype1',
            columns='Celltype2',
            values='count',
            aggfunc='sum',
            fill_value=0
        )
        pivot_table = pivot_table.reindex(
            index=celltype,
            columns=celltype,
            fill_value=0
        )

        ax = axes[idx]
        sns.heatmap(
            pivot_table,
            annot=True,
            cmap="coolwarm",
            fmt=".0f",
            annot_kws={'size': 18},
            ax=ax
        )
        ax.set_title(title, fontsize=18)
        xticks = ax.get_xticklabels()
        for xtick in xticks:
            xtick.set_fontsize(18)
        yticks = ax.get_yticklabels()
        for ytick in yticks:
            ytick.set_fontsize(18)
        ax.set_xlabel('')
        ax.set_ylabel('')

    for ax in axes[len(methods):]:
        ax.axis('off')

    plt.savefig(
        f'figure/{sample}_celltype_interaction_heatmap.pdf',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
