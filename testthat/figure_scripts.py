from CellClear.correct_expression.utils import cells_cluster
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")

color = [
    '#0067AA', '#FF7F00', '#00A23F', '#FF1F1D', '#A763AC', '#B45B5D', '#FF8AB6', '#B6B800', '#01C1CC',
    '#85D5F8', '#FFC981', '#C8571B', '#727272', '#EFC800', '#8A5626', '#502E91', '#59A4CE', '#344B2B',
    '#FBE29D', '#FDD6E6', '#849C8C', '#F07C6F', '#000101', '#FF4500', '#6959CD', '#FF8C00', '#ADFF2F',
]
color1 = [
    '#0067AA', '#FF7F00', '#00A23F', '#C8571B', '#FF1F1D', '#A763AC', '#B45B5D', '#FF8AB6', '#B6B800', '#01C1CC',
    '#85D5F8', '#FFC981', '#727272', '#EFC800', '#8A5626', '#502E91', '#59A4CE', '#344B2B',
    '#FBE29D', '#FDD6E6', '#849C8C', '#F07C6F', '#000101', '#FF4500', '#6959CD', '#FF8C00', '#ADFF2F',
]


def read_h5ad(info_dict):
    for name, path in info_dict.items():

        if path.endswith('h5'):

            data = sc.read_10x_h5(path)

        else:

            data = sc.read_10x_mtx(path)

        gex_rows = list(
            map(lambda x: x.replace('_', '-'), data.var.index)
        )
        data.var.index = gex_rows

        gex_cols = list(
            map(lambda x: name + "_" + x, data.obs.index)
        )
        data.obs.index = gex_cols

        data.var_names_make_unique()
        data.obs_names_make_unique()

        data.write(f'h5ad/{name}.h5ad', compression='lzf')


def scanpy_analysis(name):
    ref = sc.read(f'h5ad/Uncorrected_{name}.h5ad')
    ref_type = ref.obs['celltype']

    for method in ['CellClear', 'CellBender', 'DecontX', 'SoupX']:
        data = sc.read(f'h5ad/{method}_{name}.h5ad')

        # filter cells
        tmp = ref_type.copy()
        tmp.index = [f'{method}_{name}_' + i for i in tmp.index]
        inter_bc = set(data.obs_names) & set(tmp.index)

        print(f'{len(inter_bc)} out of {ref.shape[0]} used in {name} for {method}')
        data = data[list(inter_bc), :]
        data.obs['celltype'] = tmp
        raw_cell_num = data.shape[0]

        # basic filter
        sc.pp.filter_cells(data, min_genes=200)
        sc.pp.filter_genes(data, min_cells=3)

        data.var["mt"] = data.var_names.str.contains("^[Mm][Tt]-")
        sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], inplace=True, log1p=True)
        data = data[data.obs.pct_counts_mt < 50, :].copy()
        print(f'{data.shape[0]} out of {raw_cell_num} filtered')

        # analysis
        cells_cluster(data)

        data.write(f'filtered_h5ad/{method}_{name}.h5ad', compression='lzf')


#############################################run########################################################################
for sample in ['rep1', 'rep2', 'rep3']:
    info_dict = {
        f'CellClear_{sample}': f'matrix/{sample}/CellClear/',
        f'CellBender_{sample}': f'matrix/{sample}/CellBender/{sample}_out_filtered.h5',
        f'DecontX_{sample}': f'matrix/{sample}/DecontX/',
        f'SoupX_{sample}': f'matrix/{sample}/SoupX/'
    }
    read_h5ad(info_dict)

for sample in ['rep1', 'rep2', 'rep3']:
    scanpy_analysis(sample)

#############################################draw########################################################################
# use rep1 as an example
# (Figure 2A)
name = 'rep3'
methods = ['Uncorrected', 'CellClear', 'CellBender', 'DecontX', 'SoupX']
adata_files = [f'filtered_h5ad/{method}_{name}.h5ad' for method in methods]

adatas = [sc.read(file) for file in adata_files]
for adata in adatas:
    adata.uns['celltype_colors'] = color

fig, axes = plt.subplots(1, 5, figsize=(20, 4), gridspec_kw={"wspace": 0.2})

for ax, adata, title in zip(axes, adatas, methods):
    if title == 'SoupX':
        legend_loc = 'right margin'
    else:
        legend_loc = 'none'
    sc.pl.umap(adata, color='celltype', legend_loc=legend_loc, title=title, show=False, ax=ax)

plt.savefig(f'figure/{name}_celltype_umap.pdf', dpi=300,  bbox_inches="tight")
plt.close()

# (Figure 2B)
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
        ax.set_ylabel('')  # Removes the y-axis label
    ax.set_title(title)
    ax.set_ylim([min_value, max_value])  # Sets the same y-axis limits

plt.savefig(f'figure/{name}_Slc34a1_vlnplot.pdf', dpi=300, bbox_inches="tight")
plt.close()

# (Figure 2C)
uncorrected = sc.read(f'filtered_h5ad/Uncorrected_rep1.h5ad')
cellclear = sc.read(f'filtered_h5ad/CellClear_rep1.h5ad')
cellbender = sc.read(f'filtered_h5ad/CellBender_rep1.h5ad')
decontx = sc.read(f'filtered_h5ad/DecontX_rep1.h5ad')
soupx = sc.read(f'filtered_h5ad/SoupX_rep1.h5ad')
marker = {
    'CD_IC': ['Atp6v1g3', 'Atp6v1c2', 'Atp6v0d2'],
    'Endo': ['Egfl7', 'Emcn', 'Nrp1']
}
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 3), gridspec_kw={"wspace": 0.2})
sc.pl.dotplot(uncorrected, marker, groupby='celltype', dendrogram=False, ax=ax1, show=False, title='Uncorrected',
              layer='normalised')
sc.pl.dotplot(cellclear, marker, groupby='celltype', dendrogram=False, ax=ax2, show=False, title='CellClear',
              layer='normalised')
sc.pl.dotplot(cellbender, marker, groupby='celltype', dendrogram=False, ax=ax3, show=False, title='CellBender',
              layer='normalised')
sc.pl.dotplot(decontx, marker, groupby='celltype', dendrogram=False, ax=ax4, show=False, title='DecontX',
              layer='normalised')
sc.pl.dotplot(soupx, marker, groupby='celltype', dendrogram=False, ax=ax5, show=False, title='SoupX',
              layer='normalised')
plt.savefig('figure/rep1_marker_vlnplot.pdf', dpi=300, bbox_inches="tight")
plt.close()


# (Figure 2D)
def calculate_percentage(data, data_df, gene_markers):
    # Filter and calculate percentages
    exp = data_df[gene_markers]
    exp['celltype'] = data.obs['celltype']
    exp = exp[~exp['celltype'].isin(['PT'])]
    gene_dict = {g: len(exp[exp[g] > 0]) / len(exp) for g in gene_markers}
    return gene_dict


def calculate_log2fc(data, gene_markers, name, method):
    # Calculate log2 fold change
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
    return pd.DataFrame(pct_dict, index=[f"{name}_{method}"]), pd.DataFrame(logfc_dict, index=[f"{name}_{method}"])


def draw_boxplot(box_df, name):
    import seaborn as sns
    from statannot import add_stat_annotation
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
    if name == 'Log2FC':
        # pretty
        index = data_melted[(data_melted['Gene'] == 'Slc34a1') & (data_melted['Method'] == 'CellClear') & (
                data_melted['Sample'] == 'rep3')]['Value'].index
        data_melted.loc[index, 'Value'] = 15
    add_stat_annotation(ax, data=data_melted, x='Sample', y='Value', hue='Method',
                        box_pairs=pairs,
                        test='Wilcoxon', text_format='star', loc='inside')
    plt.savefig(f'figure/{name}_boxplot.pdf', dpi=300, bbox_inches="tight")
    plt.close()


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


#############genotype estimation accuracy###########################################################
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


from functools import reduce


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
#        calculate_cont_per_cell(name)
        dfs = [read_and_process_file(f'contPerCell/{method}_{name}_contperCell.tsv', method, name) for method in
               methods]
        result = merge_dfs(dfs)
        result['Sample'] = name
        cont_df = pd.concat([cont_df, result])
    return cont_df


cont_df = calculate_and_concatenate(['rep1', 'rep2', 'rep3'])
cont_df = cont_df.drop(columns=['cell'])
cont_df = cont_df.melt(id_vars=['Sample'],
                       value_vars=['Genotype estimate', 'CellClear', 'CellBender', 'SoupX', 'DecontX'],
                       var_name='Method',
                       value_name='Contamination Estimate')

import seaborn as sns

plt.figure(figsize=(8, 8))
ax = sns.boxplot(x='Sample', y='Contamination Estimate', hue='Method', data=cont_df, palette='Set3', showfliers=False)
plt.xlabel('Sample')
plt.ylabel('Contamination Estimate')
plt.legend(title='Method', loc='upper left')
ax.set_xlabel('Sample', fontsize=15)
ax.set_ylabel('Contamination Estimate', fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.savefig(f'figure/Cont_estimate_boxplot.pdf', dpi=300, bbox_inches="tight")
plt.close()

######################################################################################################################
import scanpy as sc
import seaborn as sns
import pandas as pd
from statannot import add_stat_annotation
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def draw_violin(adata, genes, annotation, name, height, aspect):
    df = adata.to_df()
    df = df[genes]
    df['Group'] = adata.obs['Sample ID']
    df['Celltype'] = adata.obs['annot_full']
    df_melted = df.melt(id_vars=['Celltype', 'Group'], var_name='Gene', value_name='Gene Expression')
    df_melted['Group'] = df_melted['Group'].cat.set_categories(['Uncorrected', 'CellClear'])
    g = sns.FacetGrid(df_melted, col='Gene', col_wrap=1, height=height, aspect=aspect, sharey=False)
    g.map_dataframe(sns.violinplot, x='Celltype', y='Gene Expression', hue='Group', split=False, palette='muted',
                    inner='quart')
    g.add_legend()
    g.fig.subplots_adjust(top=0.9, hspace=0.4)
    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title('')
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0)
    if annotation:
        for ax, (label, df_group) in zip(g.axes.flat, df_melted.groupby('Gene')):
            box_pairs = [((celltype, 'Uncorrected'), (celltype, 'CellClear')) for celltype in
                         df_group['Celltype'].unique()]
            test_results = add_stat_annotation(ax, data=df_group, x='Celltype', y='Gene Expression', hue='Group',
                                               box_pairs=box_pairs, test='t-test_ind', text_format='star',
                                               loc='outside')
    g.fig.suptitle('Gene Expression', fontsize=16)
    g.set_axis_labels(x_var="Celltype")
    ticks, labels = plt.xticks()
    plt.xticks(ticks, labels, rotation=45, horizontalalignment='right')
    plt.gca().yaxis.grid(True)
    plt.rcParams["font.sans-serif"] = ['Arial']
    plt.savefig(f'figure/{name}_violin_marker.pdf', dpi=300, bbox_inches="tight")
    plt.close()


adata = sc.read('h5ad/SRR21882339.h5ad')
order = ['RSC-PyNs', 'SubC-PyNs', 'CA1-PyNs', 'CA3-PyNs', 'DG-GCs', 'NSCs', 'Interneurons', 'IPs', 'CR',
         'Endo', 'Erythrocytes', 'Microglial']
adata = adata[adata.obs['Subset cell'].isin(order)]
adata.obs['annot_full'] = adata.obs['Subset cell']
adata.obs['annot_full'] = adata.obs['annot_full'].cat.set_categories(order)

genes = ['Hbb-bs', 'Hbb-bt', 'Hba-a1']
draw_violin(adata, genes, True, 'Erythrocytes', 2, 6)

genes = ['Satb2', 'Pou3f1', 'Grp', 'Prox1', 'Pax6', 'Gad1', 'Eomes', 'Lhx1', 'Cdh5', 'Hbb-bs', 'Aif1']
draw_violin(adata, genes, False, 'AllCelltypes', 1, 10)

color = [
    '#0067AA', '#FF7F00', '#00A23F', '#FF1F1D', '#A763AC', '#B45B5D', '#FF8AB6', '#B6B800', '#01C1CC',
    '#85D5F8', '#FFC981', '#C8571B', '#727272', '#EFC800', '#8A5626', '#502E91', '#59A4CE', '#344B2B',
    '#FBE29D', '#FDD6E6', '#849C8C', '#F07C6F', '#000101', '#FF4500', '#6959CD', '#FF8C00', '#ADFF2F',
]

adata.uns['annot_full_colors'] = color
sc.pl.umap(adata, color='annot_full', title='cell type')
plt.savefig(f'figure/SRR21882339_CELLTYPE_UMAP.pdf', dpi=300, bbox_inches="tight")
plt.close()

sc.pl.umap(adata, color='Sample ID', title='sample')
plt.savefig(f'figure/SRR21882339_SAMPLE_UMAP.pdf', dpi=300, bbox_inches="tight")
plt.close()


def output_deg(p, name):
    from pathlib import Path
    Path(f"metric/{name}").mkdir(exist_ok=True)
    adata = sc.read(p)
    if 'celltype' not in adata.obs:
        adata.obs['celltype'] = adata.obs['annot_full'].copy()
    if 'rank_genes_groups_celltype' not in adata.uns:
        if 'log1p' in adata.uns:
            adata.uns['log1p']['base'] = None
        sc.tl.rank_genes_groups(adata, layer="normalised", groupby='celltype',
                                method='wilcoxon', tie_correct=False, pts=True,
                                use_raw=False, key_added="rank_genes_groups_celltype")
        adata.write(p, compression='lzf')
    with open(f"metric/{name}/{name}_diffgenes.list", "w") as output:
        groups = adata.uns[f'rank_genes_groups_celltype']['logfoldchanges'].dtype.names
        for i in groups:
            de_i = sc.get.rank_genes_groups_df(adata,
                                               group=i,
                                               key=f"rank_genes_groups_celltype")
            de_i = de_i[de_i.logfoldchanges.abs() >= 0.1]
            de_i = de_i[(de_i["pct_nz_group"] >= 0.1) |
                        (de_i["pct_nz_reference"] >= 0.1)]
            de_i = de_i.sort_values(by="logfoldchanges",
                                    inplace=False,
                                    ascending=False)
            de_i_dir = f"metric/{name}/{name}_cluster{i}_diffgenes.xls"
            de_i.to_csv(de_i_dir, index=False, sep="\t")
            output.write(i + "\t" + de_i_dir + "\n")


output_deg('filtered_h5ad/CellClear_SRR21882339.h5ad', name='CellClear_SRR21882339')
output_deg('filtered_h5ad/Uncorrected_SRR21882339.h5ad', name='Uncorrected_SRR21882339')

##########################################draw_enrich#################################################################
cellclear_enrich_path = 'figure/enrich/CellClear/'
uncorrected_enrich_path = 'figure/enrich/Uncorrected/'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt

enrich_df = pd.DataFrame()
order = ['RSC-PyNs', 'SubC-PyNs', 'CA1-PyNs', 'CA3-PyNs', 'DG-GCs', 'NSCs', 'Interneurons', 'IPs', 'CR',
         'Endo', 'Erythrocytes', 'Microglial']
for i in order:
    df = pd.read_table(f'{cellclear_enrich_path}/{i}/{i}_up.GOALL_enrichment_sig.xls')
    df = df[['ID', 'Description', 'p.adjust']]
    df['celltype'] = i
    df['group'] = 'CellClear'
    enrich_df = pd.concat([enrich_df, df])

for i in order:
    df = pd.read_table(f'{uncorrected_enrich_path}/{i}/{i}_up.GOALL_enrichment_sig.xls')
    df = df[['ID', 'Description', 'p.adjust']]
    df['celltype'] = i
    df['group'] = 'Uncorrected'
    enrich_df = pd.concat([enrich_df, df])

enrich_df['p.adjust'] = -np.log10(enrich_df['p.adjust'])

res_df = pd.DataFrame()
for i in order:
    tmp = enrich_df[enrich_df['celltype'] == i]
    df = tmp.groupby('Description')['group'].count()
    description = df[df>1].index.tolist()
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
pivot_table.drop_duplicates().to_csv("metric/enrich.csv")


##################################################################################################
import scanpy as sc
adata = sc.read("h5ad/SRR21882339.h5ad")

order = ['RSC-PyNs', 'SubC-PyNs', 'CA1-PyNs', 'CA3-PyNs', 'DG-GCs', 'NSCs', 'Interneurons', 'IPs', 'CR',
         'Endo', 'Erythrocytes', 'Microglial']

adata = adata[adata.obs['Subset cell'].isin(order)]
adata.obs['celltype'] = adata.obs['Subset cell']
adata.obs['celltype'] = adata.obs['celltype'].cat.set_categories(order)
adata.uns['celltype_color'] = color

adatas = []
for i in adata.obs['Sample ID'].unique():
    adatas.append(adata[adata.obs['Sample ID'] == i])

fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"wspace": 0.2})

for ax, adata, title in zip(axes, adatas, methods):
    if title == 'Uncorrected':
        legend_loc = 'none'
    else:
        legend_loc = 'right margin'
    sc.pl.umap(adata, color='celltype', legend_loc=legend_loc, title=title, show=False, ax=ax)

plt.savefig(f'figure/SRR21882339_celltype_umap.pdf', dpi=300, bbox_inches="tight")
plt.close()
