import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
from copy import copy

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

color = [
    '#0067AA', '#FF7F00', '#00A23F', '#FF1F1D', '#A763AC', '#B45B5D', '#FF8AB6', '#B6B800', '#01C1CC',
    '#85D5F8', '#FFC981', '#C8571B', '#727272', '#EFC800', '#8A5626', '#502E91', '#59A4CE', '#344B2B',
    '#FBE29D', '#FDD6E6', '#849C8C', '#F07C6F', '#000101', '#FF4500', '#6959CD', '#FF8C00', '#ADFF2F',
]

sample = 'SRR21882339'
methods = ['CellClear', 'FastCar', 'Uncorrected']
celltype = ['RSC-PyNs', 'SubC-PyNs', 'CA1-PyNs', 'CA3-PyNs', 'DG-GCs', 'NSCs',
            'Interneurons', 'IPs', 'CR', 'Endo', 'Erythrocytes', 'Microglial']
os.makedirs('deg', exist_ok=True)


def draw_fig_sup2():
    # prepare data
    adatas = []
    for method in methods:
        data = sc.read(f'filtered_h5ad/{method}_{sample}.h5ad')
        data.obs['Sample ID'] = method
        adatas.append(data)

    # draw figure sup2A
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.2})
    for ax, adata, title in zip(axes, adatas, methods):
        if title == 'Uncorrected':
            legend_loc = 'right margin'
        else:
            legend_loc = 'none'
        sc.pl.umap(adata, color='celltype', legend_loc=legend_loc, title=title, show=False, ax=ax)

    plt.savefig(f'figure/{sample}_celltype_umap.pdf', dpi=300, bbox_inches="tight")
    plt.close()

    # draw figure sup2B
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"wspace": 0.2})
    reds = copy(matplotlib.cm.Reds)
    reds.set_under("#a9a9a9")

    for ax, adata, title in zip(axes, adatas, methods):
        if title == 'Uncorrected':
            legend_loc = 'right margin'
        else:
            legend_loc = 'none'
        sc.pl.umap(adata, color='Hbb-bt', legend_loc=legend_loc, title=title, show=False, ax=ax, layer='normalised',
                   color_map=reds, vmin=0.0001)

    plt.savefig(f'figure/{sample}_gene_exp.pdf', dpi=300, bbox_inches="tight")
    plt.close()

    # draw figure sup2C
    adata = adatas[2]
    genes = ['Tuba1a', 'Tubb2b', 'Tubb3']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"wspace": 0.2})
    for ax, gene, title in zip(axes, genes, genes):
        if title == 'Tubb3':
            legend_loc = 'right margin'
        else:
            legend_loc = 'none'
        sc.pl.umap(adata, color=gene, legend_loc=legend_loc, title=title, show=False, ax=ax, layer='normalised',
                   color_map=reds, vmin=0.0001)

    plt.savefig(f'figure/{sample}_multi_gene_exp.pdf', dpi=300, bbox_inches="tight")
    plt.close()


########################################################################################################################
def output_fig2F_deg():
    unique_df = pd.DataFrame()

    for method in methods:
        os.makedirs(f'deg/{method}', exist_ok=True)
        p = f'filtered_h5ad/{method}_{sample}.h5ad'
        adata = sc.read(p)

        if 'rank_genes_groups_celltype' not in adata.uns:
            if 'log1p' in adata.uns:
                adata.uns['log1p']['base'] = None
            sc.tl.rank_genes_groups(adata, layer="normalised", groupby='celltype',
                                    method='wilcoxon', tie_correct=False, pts=True,
                                    use_raw=False, key_added="rank_genes_groups_celltype")
            adata.write(p, compression='lzf')

        with open(f"deg/{method}/{method}_diffgenes.list", "w") as output:
            groups = adata.uns[f'rank_genes_groups_celltype']['logfoldchanges'].dtype.names
            for i in groups:
                de_i = sc.get.rank_genes_groups_df(adata,
                                                   group=i,
                                                   key=f"rank_genes_groups_celltype")
                de_i = de_i[de_i.logfoldchanges.abs() >= 1]
                de_i = de_i[(de_i["pct_nz_group"] >= 0.1) |
                            (de_i["pct_nz_reference"] >= 0.1)]
                de_i = de_i.sort_values(by="logfoldchanges",
                                        inplace=False,
                                        ascending=False)
                de_i_dir = f"deg/{method}/{method}_cluster{i}_diffgenes.xls"
                de_i.to_csv(de_i_dir, index=False, sep="\t")
                output.write(i + "\t" + de_i_dir + "\n")

        de_i = sc.get.rank_genes_groups_df(adata,
                                           group=None,
                                           key=f"rank_genes_groups_celltype")
        de_i = de_i[de_i.logfoldchanges >= 1]
        grouped = de_i.groupby('group')['names'].apply(set)
        unique_names_count = {}
        for group, names_set in grouped.items():
            other_groups = grouped.drop(group)
            names_in_other_groups = set.union(*other_groups)
            unique_names_count[group] = len(names_set - names_in_other_groups)
        unique_names_df = pd.DataFrame(list(unique_names_count.items()), columns=['Group', 'UniqueDEGCount'])
        unique_names_df['method'] = method
        unique_df = pd.concat([unique_df, unique_names_df])

    unique_df.to_csv('deg/unique_deg_num.csv', sep=",", index=None)


# after enrichment using clusterprofiler
def output_fig2G_enrich_data():
    enrich_df = pd.DataFrame()

    for method in methods:
        for i in celltype:
            df = pd.read_table(f'enrichment/{method}/{i}/{i}_up.GOALL_enrichment_sig.xls')
            df = df[['ID', 'Description', 'p.adjust']]
            df['celltype'] = i
            df['group'] = method
            enrich_df = pd.concat([enrich_df, df])

    enrich_df['p.adjust'] = -np.log10(enrich_df['p.adjust'])

    res_df = pd.DataFrame()
    for i in celltype:
        tmp = enrich_df[enrich_df['celltype'] == i]
        df = tmp.groupby('Description')['group'].count()
        description = df[df > 1].index.tolist()
        tmp = tmp[tmp['Description'].isin(description)].sort_values(by=['p.adjust'], ascending=False)
        tmp = tmp[tmp['Description'].isin(tmp['Description'].unique().tolist()[:5])]
        res_df = pd.concat([res_df, tmp])

    res_df['col'] = res_df['celltype'] + '_' + res_df['group']
    res_df['index'] = res_df['Description'] + '_' + res_df['celltype']

    pivot_table = res_df.pivot_table(index='index', columns=['celltype', 'group'], values='p.adjust')
    pivot_table = pivot_table.loc[res_df['index'].tolist()]
    pivot_table.columns = pivot_table.columns.map('_'.join)
    pivot_table = pivot_table[res_df['col'].unique().tolist()]
    pivot_table = pivot_table.fillna(0)
    pivot_table.drop_duplicates().to_csv("enrichment/enrich.csv")


########################################################################################################################
def draw_fig2E_data():
    genes = ['Satb2', 'Pou3f1', 'Grp', 'Prox1', 'Pax6', 'Gad1', 'Eomes', 'Lhx1', 'Cdh5', 'Hbb-bt', 'Aif1']
    deg_df = pd.DataFrame()
    for method in methods:
        adata = sc.read(f'filtered_h5ad/{method}_{sample}.h5ad')
        de_i = sc.get.rank_genes_groups_df(adata,
                                           group=None,
                                           key=f"rank_genes_groups_celltype")
        de_i = de_i[de_i['names'].isin(genes)]
        de_i['method'] = method
        deg_df = pd.concat([deg_df, de_i])

    deg_df = deg_df[['group', 'names', 'logfoldchanges', 'method']]
    deg_df['sample'] = deg_df['group'] + '_' + deg_df['method']
    new_df = deg_df[['sample', 'names', 'logfoldchanges']]
    pivot_table = new_df.pivot_table(index='sample', columns=['names'], values='logfoldchanges')
    pivot_table.to_csv('deg/deg_matrix.csv', sep=',')

    deg_df = deg_df[['sample', 'group', 'method']].drop_duplicates()
    deg_df.columns = ['sample', 'group', 'method']
    deg_df['method'] = pd.Categorical(deg_df['method'], categories=methods, ordered=True)
    deg_df['group'] = pd.Categorical(deg_df['group'], categories=celltype, ordered=True)

    deg_df.sort_values(by=['group', 'method']).to_csv('deg/deg_config.csv', sep=',', index=None)


draw_fig_sup2()
output_fig2F_deg()
draw_fig2E_data()
