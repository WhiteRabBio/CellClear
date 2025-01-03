import pandas as pd
import scanpy as sc
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")

adata = sc.read_10x_mtx('matrix/hgmm_12k/CellClear/matrix/')
exp = adata.to_df()
hg19_sum = exp.filter(like='hg19').sum(axis=1)
mm10_sum = exp.filter(like='mm10').sum(axis=1)

corrected = pd.DataFrame({
    'hg19': hg19_sum.astype('int'),
    'mm10': mm10_sum.astype('int'),
    'species': ['hg19' if hg > mm else 'mm10' for hg, mm in zip(hg19_sum, mm10_sum)]
}, index=exp.index)
corrected.index.name = 'barcode'

uncorrected = pd.read_csv('metric/uncorrected_classification.csv', index_col=0)

uncorrected['row_sum'] = uncorrected['hg19'] + uncorrected['mm10']
uncorrected['hg19'] = uncorrected['hg19'] / uncorrected['row_sum']
uncorrected['mm10'] = uncorrected['mm10'] / uncorrected['row_sum']
uncorrected = uncorrected.drop(columns=['row_sum'])
uncorrected = uncorrected[uncorrected['species'] != 'Multiplet']

corrected['row_sum'] = corrected['hg19'] + corrected['mm10']
corrected['hg19'] = corrected['hg19'] / corrected['row_sum']
corrected['mm10'] = corrected['mm10'] / corrected['row_sum']
corrected = corrected.drop(columns=['row_sum'])

uncorrected_greater_0_95 = uncorrected.groupby('species').apply(
    lambda x: {'mm10': (x['mm10'] > 0.99).mean(), 'hg19': (x['hg19'] > 0.99).mean()})

corrected_greater_0_95 = corrected.groupby('species').apply(
    lambda x: {'mm10': (x['mm10'] > 0.99).mean(), 'hg19': (x['hg19'] > 0.99).mean()})

new_df = pd.DataFrame({
    'uncorrected': [
        uncorrected_greater_0_95['hg19']['hg19'], uncorrected_greater_0_95['mm10']['mm10']
    ],
    'corrected': [
        corrected_greater_0_95['hg19']['hg19'], corrected_greater_0_95['mm10']['mm10']
    ]
}, index=['human', 'mouse'])

ax = new_df.plot(kind='bar', figsize=(7, 6))

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')

plt.title('Percentage of human and mouse UMI > 0.99 \n before and after correction', fontsize=18)
plt.ylabel('Percentage of human and mouse UMI > 0.99', fontsize=16)
plt.xlabel('Species', fontsize=18)
plt.xticks(rotation=0)
plt.legend(title='Condition', loc='upper right')
plt.tight_layout()

plt.savefig(f'figure/fig_2B_percentage_species_umi.pdf', dpi=300, bbox_inches="tight")
plt.close()


#######################################################################################################################
corrected_data = sc.read('filtered_h5ad/CellClear_hgmm_12k.h5ad')

classification = pd.read_csv('metric/uncorrected_classification.csv', index_col=0)
corrected_classification = classification.copy()
corrected_classification.index = "CellClear_hgmm_12k_" + corrected_classification.index
inter_bc = set(corrected_classification.index) & set(corrected_data.obs_names)
corrected_data = corrected_data[list(inter_bc)]
corrected_data.obs['species'] = corrected_classification.loc[list(inter_bc), 'species']
corrected_data.obs['species'] = corrected_data.obs['species'].replace({'mm10': 'mouse', 'hg19': 'human'})
corrected_data.obs['species'] = pd.Categorical(corrected_data.obs['species'], categories=['human', 'mouse'], ordered=True)

sc.pl.umap(corrected_data, color='species', legend_loc='on data', title='UMAP Plot of Species Groups After Correction')
plt.savefig(f'figure/fig_2A_umap_after_correction.pdf', dpi=300, bbox_inches="tight")
plt.close()

########################################################################################################################
topic_counts = sc.read('matrix/hgmm_12k/CellClear/counts.h5ad')
classification = pd.read_csv('metric/uncorrected_classification.csv', index_col=0)

inter_bc = set(classification.index) & set(topic_counts.obs_names)
topic_counts.obs['species'] = classification.loc[list(inter_bc), 'species']

topic_counts.X = topic_counts.layers['counts'].copy()
exp = topic_counts.to_df()
exp['species'] = topic_counts.obs['species']

mm10_hg19_top_genes = exp[exp['species'] == 'mm10'].filter(like='hg19').sum().nlargest(20)
hg19_mm10_top_genes = exp[exp['species'] == 'hg19'].filter(like='mm10').sum().nlargest(20)
top_genes = list(mm10_hg19_top_genes.index) + list(hg19_mm10_top_genes.index)

distance = pd.read_csv('matrix/hgmm_12k/CellClear/distance.csv')
ambient_genes = distance[distance['is_top']]['Gene'].tolist()
print(f'num of intersection ambient genes: {str(len((set(top_genes) & set(ambient_genes))))}')

#######################################################################################################################
corrected_cont = pd.read_csv('metric/decontx_contamination/hgmm_12k_corrected_cont.csv', sep=',', index_col=0)
uncorrected_cont = pd.read_csv('metric/decontx_contamination/hgmm_12k_uncorrected_cont.csv', sep=',', index_col=0)
classification = pd.read_csv('metric/uncorrected_classification.csv', index_col=0)

corrected_merged = corrected_cont.join(classification, how='inner')
uncorrected_merged = uncorrected_cont.join(classification, how='inner')

corrected_merged['species'] = corrected_merged['species'].replace({'mm10': 'mouse', 'hg19': 'human'})
uncorrected_merged['species'] = uncorrected_merged['species'].replace({'mm10': 'mouse', 'hg19': 'human'})

corrected_merged['species'] = pd.Categorical(corrected_merged['species'], categories=['human', 'mouse'], ordered=True)
uncorrected_merged['species'] = pd.Categorical(uncorrected_merged['species'], categories=['human', 'mouse'], ordered=True)

print(f"uncorrected data (human): {np.median(uncorrected_merged[uncorrected_merged['species'] == 'human']['Contamination']) * 100}")
print(f"uncorrected data (mouse): {np.median(uncorrected_merged[uncorrected_merged['species'] == 'mouse']['Contamination']) * 100}")
print(f"corrected data (human): {np.median(corrected_merged[corrected_merged['species'] == 'human']['Contamination']) * 100}")
print(f"corrected data (human): {np.median(corrected_merged[corrected_merged['species'] == 'mouse']['Contamination']) * 100}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.8})

sns.violinplot(x='species', y='Contamination', data=corrected_merged, ax=axes[1], inner=None)
sns.stripplot(x='species', y='Contamination', data=corrected_merged, ax=axes[1], color='k', alpha=0.5, size=2)

axes[0].set_ylim(-0.05, 0.45)
axes[0].set_yticks([i * 0.1 for i in range(5)])

sns.violinplot(x='species', y='Contamination', data=uncorrected_merged, ax=axes[0], inner=None)
sns.stripplot(x='species', y='Contamination', data=uncorrected_merged, ax=axes[0], color='k', alpha=0.5, size=2)

axes[1].set_ylim(-0.05, 0.45)
axes[1].set_yticks([i * 0.1 for i in range(5)])

axes[0].set_title('Contamination Violin Plot (Uncorrected)')
axes[0].set_xlabel('Species')
axes[0].set_ylabel('Proportions of contamination transcripts')

axes[1].set_title('Contamination Violin Plot (Corrected)')
axes[1].set_xlabel('Species')
axes[1].set_ylabel('Proportions of contamination transcripts')

plt.tight_layout()

plt.savefig(f'figure/fig_2C_contamination_per_barcode.pdf', dpi=300, bbox_inches="tight")
plt.close()
