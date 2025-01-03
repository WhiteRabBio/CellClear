import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")
from matplotlib import pyplot as plt


cellclear = pd.read_table('cellclear_select_metadata.txt')
cellclear = cellclear[['barcode', 'cell.type', 'pseudotime']]
cellbender = pd.read_table('cellbender_select_metadata.txt')
cellbender = cellbender[['barcode', 'cell.type', 'pseudotime']]
cellbender['barcode'] = cellbender['barcode'].str.replace('Adult_1', 'Adult1_Corrected')
cellbender['barcode'] = cellbender['barcode'].str.replace('Adult_2', 'Adult2_Corrected')

uncorrect = pd.read_table('uncorrected_select_metadata.txt')
uncorrect = uncorrect[['barcode', 'cell.type', 'pseudotime']]
uncorrect['barcode'] = uncorrect['barcode'].str.replace('Adult1_Uncorrected', 'Adult1_Corrected')
uncorrect['barcode'] = uncorrect['barcode'].str.replace('Adult2_Uncorrected', 'Adult2_Corrected')


inter_bc = list(set(cellclear['barcode'].tolist()) & set(uncorrect['barcode'].tolist()) & set(cellbender['barcode'].tolist()))
cellclear = cellclear[cellclear['barcode'].isin(inter_bc)]
cellclear['group'] = 'CellClear'
del cellclear['barcode']
uncorrect = uncorrect[uncorrect['barcode'].isin(inter_bc)]
uncorrect['group'] = 'uncorrected'
del uncorrect['barcode']
cellbender = cellbender[cellbender['barcode'].isin(inter_bc)]
cellbender['group'] = 'CellBender'
del cellbender['barcode']


df = pd.concat([uncorrect, cellclear, cellbender])
df['cell.type'] = df['cell.type'].astype('category')
order = ['SSCs', 'SPGs', 'Leptotene', 'Zygotene', 'Pachytene', 'Diplotene', 'RoundS.tids', 'ElongatedS.tids', 'Sperms']
df['cell.type'] = df['cell.type'].cat.set_categories(order)

plt.figure(figsize=(36, 10))
ax = sns.boxplot(x='cell.type', y='pseudotime', hue='group', data=df, palette='Set3', showfliers=False)
# plt.xticks(rotation=45)

cell_types = df['cell.type'].cat.categories
groups = ['uncorrected', 'CellClear', 'CellBender']

for i, cell_type in enumerate(cell_types):
    for j, group in enumerate(groups):
        group_data = df[(df['cell.type'] == cell_type) & (df['group'] == group)]['pseudotime']
        if len(group_data) > 1:
            Q1 = group_data.quantile(0.25)
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            filtered_data = group_data[(group_data >= Q1 - 1.5 * IQR) & (group_data <= Q3 + 1.5 * IQR)]
            if len(filtered_data) > 1:
                variance = filtered_data.var()
                x = i + (j - 0.5) * 0.2
                y = Q3 + 10
                ax.text(x, y, f'{variance:.2f}', color='black', ha='center', fontsize=15)

plt.xlabel('Cell Type', fontsize=18)
plt.ylabel(f'Pseudotime', fontsize=18)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
plt.savefig('boxplot.pdf', dpi=300, bbox_inches="tight")
plt.close()
