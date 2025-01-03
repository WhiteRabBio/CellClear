from CellClear.correct_expression.utils import cells_cluster
import scanpy as sc
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def read_h5ad(info_dict):
    for name, path in info_dict.items():
        if path.endswith('h5'):
            data = sc.read_10x_h5(path)

        elif path.endswith('tsv'):
            data = sc.read_csv(
                path,
                delimiter='\t'
            ).T

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


def scanpy_analysis_for_mouse_kidney(name):
    ref_type = pd.read_table(f'metric/mouse_kidney_annotation/{name}_annot.tsv', index_col=0)

    methods = ['CellClear', 'CellBender', 'DecontX', 'SoupX', 'Uncorrected']

    for method in methods:
        data = sc.read(f'h5ad/{method}_{name}.h5ad')

        # filter cells
        tmp = ref_type.copy()
        tmp.index = [f'{method}_{name}_' + i for i in tmp.index]
        inter_bc = set(data.obs_names) & set(tmp.index)

        print(f'{len(inter_bc)} out of {ref_type.shape[0]} used in {name} for {method}')
        data = data[list(inter_bc), :]
        data.obs['celltype'] = tmp
        raw_cell_num = data.shape[0]

        # basic filter
        sc.pp.filter_cells(data, min_genes=200)
        sc.pp.filter_genes(data, min_cells=3)

        data.var["mt"] = data.var_names.str.contains("^[Mm][Tt]-")
        sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], inplace=True, log1p=True)
        data = data[data.obs.pct_counts_mt < 50, :].copy()
        print(f'{raw_cell_num - data.shape[0]} out of {raw_cell_num} filtered')

        # analysis
        cells_cluster(data)

        data.write(f'filtered_h5ad/{method}_{name}.h5ad', compression='lzf')


def scanpy_analysis_for_pbmc(name):
    methods = ['CellClear', 'CellBender', 'Uncorrected']

    for method in methods:
        data = sc.read(f'h5ad/{method}_{name}.h5ad')

        # basic filter
        sc.pp.filter_cells(data, min_genes=0)
        sc.pp.filter_cells(data, min_counts=0)
        sc.pp.filter_cells(data, max_genes=np.percentile(data.obs["n_genes"], 95))
        sc.pp.filter_cells(data, max_counts=np.percentile(data.obs["n_counts"], 95))

        data.var["mt"] = data.var_names.str.contains("^[Mm][Tt]-")
        sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], inplace=True, log1p=True)
        data = data[data.obs.pct_counts_mt <= 10, :].copy()

        # follow the resolution used in paper
        cells_cluster(data, resol=0.5)

        data.write(f'filtered_h5ad/{method}_{name}.h5ad', compression='lzf')


def scanpy_analysis_for_hgmm(name):
    methods = ['CellClear', 'Uncorrected']

    for method in methods:
        data = sc.read(f'h5ad/{method}_{name}.h5ad')

        # basic filter
        sc.pp.filter_cells(data, min_genes=0)
        sc.pp.filter_cells(data, min_counts=0)
        sc.pp.filter_cells(data, max_genes=np.percentile(data.obs["n_genes"], 95))
        sc.pp.filter_cells(data, max_counts=np.percentile(data.obs["n_counts"], 95))

        data.var["mt"] = data.var_names.str.contains("^[Mm][Tt]-")
        sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], inplace=True, log1p=True)
        data = data[data.obs.pct_counts_mt <= 10, :].copy()

        # follow the resolution used in paper
        cells_cluster(data, resol=0.5)

        data.write(f'filtered_h5ad/{method}_{name}.h5ad', compression='lzf')


def scanpy_analysis_for_mouse_brain(name):
    methods = ['CellClear', 'Uncorrected']

    for method in methods:
        data = sc.read(f'h5ad/{method}_{name}.h5ad')

        # basic filter
        sc.pp.filter_cells(data, min_genes=750)
        sc.pp.filter_cells(data, min_counts=0)
        sc.pp.filter_cells(data, max_genes=5000)

        data.var["mt"] = data.var_names.str.contains("^[Mm][Tt]-")
        sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], inplace=True, log1p=True)
        data = data[data.obs.pct_counts_mt <= 10, :].copy()

        # follow the resolution used in paper
        cells_cluster(data, resol=1.2)

        data.write(f'filtered_h5ad/{method}_{name}.h5ad', compression='lzf')


def step1_preprocessing():
    for sample in ['hgmm_12k', 'SRR21882339', 'SRR21882340']:
        # for sample in ['rep1', 'rep2', 'rep3', 'pbmc8k', 'hgmm_12k', 'SRR21882339', 'SRR21882340']:
        if sample in ['rep1', 'rep2', 'rep3']:
            info_dict = {
                f'Uncorrected_{sample}': f'matrix/{sample}/Uncorrected/filtered_feature_bc_matrix.h5',
                f'CellClear_{sample}': f'matrix/{sample}/CellClear/matrix',
                f'CellBender_{sample}': f'matrix/{sample}/CellBender/{sample}_out_filtered.h5',
                f'DecontX_{sample}': f'matrix/{sample}/DecontX/',
                f'SoupX_{sample}': f'matrix/{sample}/SoupX/',
            }
            read_h5ad(info_dict)
            scanpy_analysis_for_mouse_kidney(sample)
        elif sample in ['pbmc8k']:
            info_dict = {
                f'Uncorrected_{sample}': f'matrix/{sample}/Uncorrected/filtered_feature_bc_matrix.h5',
                f'CellClear_{sample}': f'matrix/{sample}/CellClear/matrix',
                f'CellBender_{sample}': f'matrix/{sample}/CellBender/{sample}_out_filtered.h5',
            }
            read_h5ad(info_dict)
            scanpy_analysis_for_pbmc(sample)
        elif sample in ['hgmm_12k']:
            info_dict = {
                f'Uncorrected_{sample}': f'matrix/{sample}/Uncorrected/',
                f'CellClear_{sample}': f'matrix/{sample}/CellClear/matrix',
            }
            read_h5ad(info_dict)
            scanpy_analysis_for_hgmm(sample)
        elif sample in ['SRR21882339', 'SRR21882340']:
            info_dict = {
                f'Uncorrected_{sample}': f'matrix/{sample}/Uncorrected/filtered_feature_bc_matrix.h5',
                f'CellClear_{sample}': f'matrix/{sample}/CellClear/matrix',
            }
            read_h5ad(info_dict)
            scanpy_analysis_for_mouse_brain(sample)


step1_preprocessing()
