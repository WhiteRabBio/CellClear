from anndata import AnnData
from pathlib import Path
from scipy import io
import scanpy as sc
import numpy as np
import pandas as pd
import os
import scipy
import gzip

MATRIX_FILE_NAME = 'matrix.mtx.gz'
FEATURE_FILE_NAME = 'features.tsv.gz'
BARCODE_FILE_NAME = 'barcodes.tsv.gz'


def _load_data(data_dir: str, verbose=True) -> AnnData:
    """Load matrix from the input_file"""

    # Detect input data type.
    data_type = _detect_input_data_type(data_dir)

    # Load the dataset.
    try:
        if data_type == '10x-format':
            if verbose:
                print(f"Loading data from directory {data_dir}")
            data = sc.read_10x_mtx(data_dir)

        elif data_type == 'h5-format':
            if verbose:
                print(f"Loading data from file {data_dir}")
            data = sc.read_10x_h5(data_dir)

        else:
            if verbose:
                print(f"Loading data from file {data_dir}")
            data = sc.read_csv(Path(data_dir), delimiter="\t").T

    except Exception as e:
        raise e

    gex_rows = list(
        map(lambda x: x.replace('_', '-'), data.var.index)
    )
    data.var.index = gex_rows

    data.var_names_make_unique()
    data.obs_names_make_unique()

    return data


def _detect_input_data_type(data_dir: str) -> str:
    """Detect the type of input data."""

    if os.path.isdir(data_dir):
        return '10x-format'

    elif data_dir.endswith('h5'):

        return 'h5-format'

    else:
        return 'normal-format'


def norm_log(counts):
    """
    Wrapper function for sc.pp.normalize_per_cell() and sc.pp.log1p()
    """

    counts.layers['counts'] = counts.X.copy()

    sc.pp.normalize_total(
        counts,
        target_sum=1e4,
        inplace=True,
    )

    sc.pp.log1p(
        counts,
    )

    counts.layers['normalised'] = counts.X.copy()


def hvg(counts):
    """
    Wrapper function for sc.highly_variable_genes()
    """
    sc.pp.highly_variable_genes(
        counts,
        layer=None,
        n_top_genes=2000,
        min_disp=0.5,
        max_disp=np.inf,
        min_mean=0.0125,
        max_mean=3,
        span=0.3,
        n_bins=20,
        flavor='seurat',
        subset=False,
        inplace=True,
        batch_key=None,
        check_values=True
    )


def scale(counts, zero_center=True):
    """
    Wrapper function for sc.pp.scale()
    """
    sc.pp.scale(
        counts,
        zero_center=zero_center,
        max_value=10,
        copy=False,
        layer=None,
        obsm=None
    )


def pca(counts):
    """
    Wrapper function for sc.pp.pca()
    """
    sc.pp.pca(
        counts,
        n_comps=50,
        zero_center=True,
        svd_solver='auto',
        random_state=0,
        return_info=False,
        use_highly_variable=True,
        dtype='float32',
        copy=False,
        chunked=False,
        chunk_size=None
    )


def neighbors(counts):
    """
    Wrapper function for sc.pp.neighbors()
    """
    use_rep = 'X_pca_harmony' if 'X_pca_harmony' in counts.obsm else 'X_pca'
    sc.pp.neighbors(
        counts,
        n_neighbors=15,
        n_pcs=25,
        use_rep=use_rep,
        knn=True,
        random_state=0,
        method='umap',
        metric='euclidean',
        key_added=None,
        copy=False
    )


def harmony(counts, batch_name):
    """
    Wrapper function for sc.external.pp.harmony_integrate
    """
    sc.external.pp.harmony_integrate(
        counts,
        key=batch_name,
        basis='X_pca',
        adjusted_basis='X_pca_harmony'
    )


def umap(counts):
    """
    Wrapper function for sc.tl.umap()
    """
    sc.tl.umap(
        counts,
        min_dist=0.5,
        spread=1.0,
        n_components=2,
        maxiter=None,
        alpha=1.0,
        gamma=1.0,
        negative_sample_rate=5,
        init_pos='spectral',
        random_state=0,
        a=None,
        b=None,
        copy=False,
        method='umap',
        neighbors_key=None
    )


def leiden(counts, resol=1.2):
    """
    Wrapper function for sc.tl.leiden
    """
    sc.tl.leiden(
        counts,
        resolution=resol,
        restrict_to=None,
        random_state=0,
        key_added='cluster',
        adjacency=None,
        directed=True,
        use_weights=True,
        n_iterations=-1,
        partition_type=None,
        neighbors_key=None,
        obsp=None,
        copy=False
    )


def find_marker_genes(counts):
    """
    Wrapper function for sc.tl.rank_genes_groups()
    """
    sc.tl.rank_genes_groups(
        counts,
        "cluster",
        reference='rest',
        pts=True,
        method="wilcoxon",
        use_raw=False,
        tie_correct=True,
        layer='normalised'
    )


def output_10x_matrix(data, matrix_dir):
    """
    Function for output 10X-format matrix
    """
    os.makedirs(f'{matrix_dir}', exist_ok=True)

    sparse_mtx = scipy.sparse.coo_matrix(data.X.T)
    with gzip.open(f'{matrix_dir}/{MATRIX_FILE_NAME}', 'wb') as f:
        io.mmwrite(f, sparse_mtx)

    feature_types = ['Gene Expression' for _ in range(data.shape[1])]
    gene_ids = data.var['gene_ids'].tolist() \
        if 'gene_ids' in data.var else data.var_names
    gene_names = data.var_names
    features = pd.DataFrame(
        {'gene_ids': gene_ids, 'gene_names': gene_names, 'feature_types': feature_types}
    )
    features.to_csv(
        f'{matrix_dir}/{FEATURE_FILE_NAME}',
        index=False, sep='\t', header=False
    )

    pd.DataFrame(data.obs_names).to_csv(
        f'{matrix_dir}/{BARCODE_FILE_NAME}',
        index=False, sep='\t', header=False
    )


def cells_cluster(counts: AnnData, resol: float = 1.2):
    """Wrapper function from scanpy"""

    print(f"Fetch clustering info from data...")
    norm_log(counts)
    hvg(counts)
    scale(counts)
    pca(counts)
    neighbors(counts)
    umap(counts)
    leiden(counts, resol)
    find_marker_genes(counts)
