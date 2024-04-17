from CellClear.correct_expression.utils import cells_cluster
import scanpy as sc
import numpy as np


def scanpy_analysis(info_dict):
    for name, path in info_dict.items():

        if path.endswith('h5'):

            # read data
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

        # filter
        sc.pp.filter_cells(data, min_genes=200)
        sc.pp.filter_cells(data, min_counts=0)
        sc.pp.filter_genes(data, min_cells=5)

        data.var["mt"] = data.var_names.str.contains("^[Mm][Tt]-")
        sc.pp.calculate_qc_metrics(
            data, qc_vars=["mt"], inplace=True, log1p=True
        )

        max_genes = np.percentile(data.obs["n_genes"], 98) + 1
        max_counts = np.percentile(data.obs["n_counts"], 98) + 1
        sc.pp.filter_cells(data, max_genes=max_genes)
        sc.pp.filter_cells(data, max_counts=max_counts)
        data = data[data.obs.pct_counts_mt < 50, :]

        # analysis
        cells_cluster(data)

        data.write(f'{name}.h5ad', compression='lzf')


#################run#################
sample_list = ['rep3']
for sample in sample_list:
    info_dict = {
        f'CellClear_{sample}': f'/Personal/huangwanxiang/test/{sample}/CellClear/',
        f'CellBender_{sample}': f'/Personal/huangwanxiang/test/{sample}/CellBender/{sample}_out_filtered.h5',
        f'DecontX_{sample}': f'/Personal/huangwanxiang/test/{sample}/DecontX/',
        f'SoupX_{sample}': f'/Personal/huangwanxiang/test/{sample}/SoupX/'
    }
    scanpy_analysis(info_dict)
