import scanpy as sc
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")
from matplotlib import pyplot as plt

marker_dict = {
    'ECs': ['PECAM1', 'CDH5'],
    'Fibroblasts': ['DCN', 'COL1A1'],
    'Keratinocytes': ['KRT1', 'KRT10'],
    'LangerhansCells': ['CD207', 'CD1A'],
    'MPs': ['CD14', 'CD1C'],
    'MastCells': ['TPSAB1', 'TPSB2'],
    'Melanocytes': ['MLANA', 'PMEL'],
    'MuralCells': ['RGS5', 'ACTA2'],
    'SGCs': ['DCD', 'AQP5'],
    'SchwannCells': ['S100B', 'PMP22'],
    'TCells': ['CD2', 'CD3D']
}
celltype_order = [celltype for celltype in marker_dict.keys()]
marker_list = [gene for genes in marker_dict.values() for gene in genes]

corrected_data = sc.read('filtered_h5ad/CellClear_human_skin.h5ad')
uncorrected_data = sc.read('filtered_h5ad/Uncorrected_human_skin.h5ad')

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
for ax, adata, title in zip(axes, [uncorrected_data, corrected_data], ['Uncorrected', 'CellClear-Corrected']):
    sc.pl.matrixplot(
        adata,
        marker_list,
        "autoanno_celltype",
        dendrogram=False,
        cmap='Reds',
        layer='normalised',
        standard_scale='var',
        colorbar_title="column scaled\nexpression",
        show=False, ax=ax, title=f'{title} major cell type marker expression'
    )

plt.savefig('figure/fig_3A_human_brain_marker_matrixplot.pdf',
            bbox_inches="tight", dpi=300)
plt.close()

#######################################################################################################################
mps_dict = {
    "Macrophages": ["C1QA", "APOE"],
    "cDC2": ["FCER1A", "CD1C"],
    "cDC1": ["XCR1", "CLEC9A"],
    "pDCs": ["CLEC4C", "MZB1"],
    "MatureDCs": ["CCR7", "LAMP3"],
    "ProliferatingMPs": ["MKI67", "TOP2A"],
    "Neutrophils": ["CSF3R", "FCGR3B"],
    "Monocytes": ["VCAN", "FCN1"],
    'Keratinocytes': ['KRT1', 'KRT10'],
}

mps_marker_list = [gene for genes in mps_dict.values() for gene in genes]
corrected_mps_data = sc.read('filtered_h5ad/CellClear_human_skin_MPs.h5ad')
uncorrected_mps_data = sc.read('filtered_h5ad/Uncorrected_human_skin_MPs.h5ad')

corrected_annotation = {
    "1": "2: cDC2",
    "2": "2: cDC2",
    "3": "2: cDC2",
    "4": "1: Macrophages",
    "5": "8: Unassigned",
    "6": "1: Macrophages",
    "7": "2: cDC2",
    "8": "1: Macrophages",
    "9": "1: Macrophages",
    "10": "2: cDC2",
    "11": "1: Macrophages",
    "12": "1: Macrophages",
    "13": "6: ProliferatingMPs",
    "14": "4: cDC1/pDCs",
    "15": "3: Macrophages/cDC2",
    "16": "3: Macrophages/cDC2",
    "17": "7: Neutrophils/Monocytes",
    "18": "3: Macrophages/cDC2",
    "19": "3: Macrophages/cDC2",
    "20": "5: MatureDCs",
    "21": "7: Neutrophils/Monocytes"
}
corrected_mps_celltype_order = [value for value in corrected_annotation.values()]
corrected_mps_celltype_order = sorted(set(corrected_mps_celltype_order))

uncorrected_annotation = {
    "1": "8: Unassigned",
    "2": "8: Unassigned",
    "3": "2: cDC2",
    "4": "1: Macrophages",
    "5": "8: Unassigned",
    "6": "2: cDC2",
    "7": "1: Macrophages",
    "8": "3: Macrophages/cDC2",
    "9": "2: cDC2",
    "10": "3: Macrophages/cDC2",
    "11": "1: Macrophages",
    "12": "7: Neutrophils/Monocytes",
    "13": "3: Macrophages/cDC2",
    "14": "4: cDC1/pDCs",
    "15": "6: ProliferatingMPs",
    "16": "3: Macrophages/cDC2",
    "17": "5: MatureDCs",
    "18": "2: cDC2",
    "19": "3: Macrophages/cDC2"
}
uncorrected_mps_celltype_order = [value for value in uncorrected_annotation.values()]
uncorrected_mps_celltype_order = sorted(set(uncorrected_mps_celltype_order))

corrected_mps_data.obs['celltype'] = corrected_mps_data.obs['raw_cluster'].map(corrected_annotation).astype('category')
corrected_mps_data.obs['celltype'].cat.set_categories(corrected_mps_celltype_order, inplace=True)

uncorrected_mps_data.obs['celltype'] = uncorrected_mps_data.obs['raw_cluster'].map(uncorrected_annotation).astype(
    'category')
uncorrected_mps_data.obs['celltype'].cat.set_categories(uncorrected_mps_celltype_order, inplace=True)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
for ax, adata, title in zip(axes, [uncorrected_mps_data, corrected_mps_data], ['Uncorrected', 'CellClear-corrected']):
    celltype_counts = adata.obs['celltype'].value_counts(normalize=True) * 100
    new_labels = sorted(set([f"{celltype} ({count:.0f}%)" for celltype, count in celltype_counts.items()]))
    adata.rename_categories("celltype", new_labels)
    sc.pl.dotplot(
        adata,
        mps_marker_list,
        "celltype",
        dendrogram=False,
        show=False,
        dot_max=0.5,
        ax=ax,
        standard_scale='var',
        layer='normalised',
        colorbar_title="column scaled\nexpression",
        title=f'{title} MPs marker expression'
    )

plt.savefig('figure/fig_3B_human_brain_marker_mps_dotplot.pdf',
            bbox_inches="tight", dpi=300)
plt.close()
