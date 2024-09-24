from anndata import AnnData, logging
from typing import List, Tuple, Optional
from collections import Counter
from sklearn.decomposition import non_negative_factorization
from statsmodels.stats.multitest import multipletests
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from scipy.optimize import nnls
from scipy import sparse
from sklearn.metrics import roc_curve, auc
from CellClear.correct_expression.utils import _load_data, cells_cluster
from itertools import chain
from scanpy._utils import select_groups
from scanpy.preprocessing._utils import _get_mean_var
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
import scanpy as sc
import numpy as np
import pandas as pd
import re
import warnings

logging.anndata_logger.addFilter(
    lambda r: not r.getMessage().startswith("storing")
              and r.getMessage().endswith("as categorical.")
              and r.getMessage().endswith(", copying.")
)
warnings.filterwarnings("ignore")


def _preprocess_data(
        filtered_mtx_path: str,
        raw_mtx_path: str,
        resolution: float = 1.2,
        min_background_counts_num: int = 5000,
        min_cluster_num: int = 50,
        max_umi_count: int = 200,
        min_umi_count: int = 50,
        environ_range: List[int] = [60, 100],
        black_gene_list: List[str] = None,
) -> Tuple[AnnData, AnnData]:
    """\
    load filtered mtx and raw mtx, preprocessing and get rank genes info, etc

    Parameters
    ----------
    filtered_mtx_path
        refers to a filtered matrix file or a path.
    raw_mtx_path
        refers to a raw matrix file or a path.
    min_background_counts_num
        minimum barcodes from background, Too few may result in a lack of background information
    min_cluster_num
        minimum cluster number
    max_umi_count
        maximum umi count for selecting background barcodes
    min_umi_count
        minimum umi count for selecting background barcodes
    environ_range
        umi range for selecting background barcodes
    black_gene_list
        genes needed to be excluded to denoise

    Returns
    -------
    Preprocessed filter counts anndata object and background counts anndata object
    """

    # create anndata object
    filtered_counts = _load_data(filtered_mtx_path)
    raw_counts = _load_data(raw_mtx_path)

    # extract background barcodes
    common_barcodes = np.intersect1d(filtered_counts.obs_names, raw_counts.obs_names)
    if common_barcodes.shape[0] / filtered_counts.shape[0] <= 0.8:
        raise Exception('raw matrix and filtered matrix may not come from a same sample.')

    background_counts = raw_counts[~raw_counts.obs_names.isin(common_barcodes)]
    background_counts = _exclude_genes(background_counts)
    background_counts.obs['umis'] = np.array(background_counts.X.sum(axis=1)).squeeze()

    # prevent getting stuck in an infinite loop
    background_counts_num = 0
    while (background_counts_num < min_background_counts_num) and \
            (environ_range[1] <= max_umi_count or environ_range[0] > min_umi_count):
        col_indices = background_counts.obs['umis'].between(
            environ_range[0], environ_range[1], inclusive='neither'
        )
        background_counts_num = col_indices.sum()
        if environ_range[1] >= max_umi_count and environ_range[0] <= min_umi_count:
            print("Cannot adjust environ_range any further, breaking the loop.")
            break
        if environ_range[1] < max_umi_count:
            environ_range[1] += 10
        elif environ_range[0] > min_umi_count:
            environ_range[0] -= 2

    background_counts = background_counts[col_indices, :]
    print(f"Final umis range used: [{environ_range[0]}, {environ_range[1]}]")

    # fetch the clustering info
    filtered_counts = _exclude_genes(filtered_counts, black_gene_list=black_gene_list)
    cells_cluster(filtered_counts, resol=resolution)
    filtered_counts = filtered_counts[
        filtered_counts.obs['cluster'].astype('str').map(
            filtered_counts.obs['cluster'].astype('str').value_counts()
        ) > min_cluster_num
        ]

    # filter genes, Only keep genes in both Anndata objects
    keep_genes = sorted(list(set(filtered_counts.var_names) & set(background_counts.var_names)))
    filtered_counts = filtered_counts[:, keep_genes]
    background_counts = background_counts[:, keep_genes]
    background_counts.layers['counts'] = background_counts.X.copy()

    return filtered_counts, background_counts


def identify_module(
        counts: AnnData,
        cluster_slot: str = 'cluster',
        rank_genes_slot: str = 'rank_genes_groups',
        mtx_slot: str = 'counts',
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """\
    identify gene expression module based on clustering and rank genes info

    Parameters
    ----------
    counts
        an AnnData object that has been clustered and has rank genes info
    cluster_slot
        setting for which slot in the AnnData object to be used as ident
    rank_genes_slot
        setting for which slot in the AnnData object stored rank genes info
    mtx_slot
        setting for which slot in the AnnData object to be scaled

    Returns
    -------
    usage, spectra from NMF analysis,  parameters used for NMF analysis
    """
    # select highly-exp genes for nmf analysis
    rank_genes = sc.get.rank_genes_groups_df(counts, key=rank_genes_slot, group=None)
    rank_genes = rank_genes.query(
        f'logfoldchanges>=0 & (pct_nz_group>=0.1 | pct_nz_reference>=0.1)'
    ).sort_values(
        'logfoldchanges', ascending=False
    ).reset_index(
        drop=True
    )

    # extract expression matrices and scale to unit variance
    X = _scale_data(counts, slot=mtx_slot)

    # set model n_topic to number of cluster
    n_topic = len(set(counts.obs[cluster_slot]))
    if counts.obs[cluster_slot].dtype != 'category':
        counts.obs[cluster_slot] = counts.obs[cluster_slot].astype('category')
    topic_name = ['Topic_' + i for i in counts.obs[cluster_slot].cat.categories]

    # define nmf parameters
    _nmf_kwargs = _init_nmf(counts=counts, rank_genes=rank_genes, n_components=n_topic)

    # run nmf
    spectra, usages = _nmf(X, _nmf_kwargs)
    usages = pd.DataFrame(usages, index=counts.obs_names, columns=topic_name)
    spectra = pd.DataFrame(spectra.T, index=counts.var_names, columns=topic_name)

    return usages, spectra, _nmf_kwargs


def calculate_nonzero_percentage(row: np.array) -> float:
    """\
    Calculate the percentage of non-zero elements in a given row.

    Parameters
    ----------
    row
        A one-dimensional array containing numerical elements.

    Returns
    -------
    nonzero_percentage
        The percentage of non-zero elements in the row, represented as a float between 0 and 1.
    """
    nonzero_elements = row[row != 0].count()
    total_elements = len(row)
    nonzero_percentage = nonzero_elements / total_elements

    return nonzero_percentage


def calculate_expression_percentage(counts: AnnData, gene_sets: List, slot: str) -> pd.DataFrame:
    """\
    Calculate the expression percentage of genes in different clusters of cells.

    Parameters
    ----------
    counts
        An AnnData object containing gene expression data.
    gene_sets
        A list of gene sets to calculate expression percentages for.
    slot
        The name of the data slot within the AnnData object to use for calculations.

    Returns
    -------
    output_df
        A DataFrame containing the expression percentages of genes in different clusters. 
        Columns represent clusters, and rows represent gene sets.
    """
    # Determine the cluster levels from the 'cluster' column in counts.obs
    if counts.obs['cluster'].dtype == 'category':
        levels = counts.obs['cluster'].cat.categories
    else:
        levels = counts.obs['cluster'].unique()

    # Try to access the specified data slot and copy it to counts.X
    try:
        counts.X = counts.layers[slot].copy()
    except ValueError:
        del counts.X
        counts.X = counts.layers[slot].copy()

    # Calculate the expression percentage for each gene set and store it in the output DataFrame
    output_df = pd.DataFrame(0, columns=levels, index=gene_sets)
    for clu in levels:
        counts_tmp = counts[counts.obs['cluster'] == clu]
        gene_data_dat = counts_tmp.to_df()[gene_sets]
        gene_exp_percent = gene_data_dat.apply(calculate_nonzero_percentage, axis=0)
        output_df[clu] = gene_exp_percent

    return output_df


def calculate_average_expression(
        counts: AnnData,
        gene_sets: List[str],
        slot: str
):
    """\
    Calculate the average expression of genes in different clusters of cells.

    Parameters
    ----------
    counts
        An AnnData object containing gene expression data.
    gene_sets
        A list of gene sets to calculate average expression for.
    slot
        The name of the data slot within the AnnData object to use for calculations.

    Returns
    -------
    output_df
        A DataFrame containing the average expression of genes in different clusters.
        Columns represent clusters, and rows represent gene sets.
    """

    adata = counts[:, gene_sets].copy()
    groups_order, groups_masks_obs = select_groups(
        adata, 'all', 'cluster'
    )

    X = adata.layers[slot]
    var_names = adata.var_names

    n_genes = X.shape[1]
    n_groups = groups_masks_obs.shape[0]

    means = np.zeros((n_groups, n_genes))

    for imask, mask in enumerate(groups_masks_obs):
        X_mask = X[mask]
        means[imask], _ = _get_mean_var(X_mask)

    means = pd.DataFrame(
        means,
        index=groups_order,
        columns=var_names
    ).T

    return means


def nnls_regression(x: pd.DataFrame, W: pd.DataFrame) -> np.array:
    """
    Perform Non-Negative Least Squares (NNLS) regression on input data.

    Parameters
    ----------
    x
        The input data for regression with cells as columns and genes as rows.
    W
        The matrix of regression coefficients.

    Returns
    -------
    y
        The result of NNLS regression, with samples as columns and coefficients as rows.
    """
    n_cols = x.shape[1]
    y = np.zeros((W.shape[1], n_cols))
    residuals = np.zeros(n_cols)

    W = W.to_numpy()
    x = x.to_numpy()

    for i in range(n_cols):
        col = x[:, i]
        try:
            coeffs, residual = nnls(W, col)
            y[:, i] = coeffs
        except RuntimeError:
            y[:, i] = 0
        residuals[i] = residual

    return y


def deconvolution(
        background_counts: AnnData,
        spectra: pd.DataFrame,
        ref: pd.DataFrame,
        min_prop: Optional[float] = 0.1,
        similarity: Optional[float] = 0.9,
        min_cell_num: Optional[float] = 10
) -> pd.DataFrame:
    """\
    Perform deconvolution of data using Non-Negative Least Squares (NNLS) regression.

    Parameters
    ----------
    background_counts
        An AnnData object containing background counts data.
    spectra
        The spectra data for deconvolution.
    ref
        The reference data for deconvolution.
    min_prop
        The minimum proportion of components to consider. Default is 0.1.
    similarity
        The similarity threshold for component selection. Default is 0.9.
    min_cell_num
        The minimum number of cells to consider for a component. Default is 10.

    Returns
    -------
    result
        The deconvolution result, filtered based on similarity and minimum cell count criteria.
    """
    # Scale the background counts data
    # Transpose the background counts to match dimensions 
    sc.pp.scale(background_counts, zero_center=False)
    x = background_counts.to_df().T

    # Perform NNLS regression to obtain deconvolution coefficients
    y = nnls_regression(x, spectra)
    y = pd.DataFrame(y, columns=background_counts.obs_names, index=spectra.columns)

    # Initialize result matrix
    n = y.shape[1]
    result = np.zeros((n, len(ref.columns)))

    # Perform deconvolution for each real cell
    ref_np = ref.to_numpy()
    for i in range(n):
        try:
            pred = nnls(ref_np, y.iloc[:, i])
            prop = pred[0] / np.sum(pred[0])
            prop[prop < min_prop] = 0
            prop = prop / np.sum(prop)
            result[i, :] = prop
        except RuntimeError:  # Maximum number of iterations reached
            result[i, :] = 0

    # Create a DataFrame from the result matrix, drop rows with NaN values
    # Filter results based on similarity threshold
    result = pd.DataFrame(result, index=background_counts.obs_names, columns=ref.columns).dropna()
    failed_bc = result.T.sum()[result.T.sum() == 0].shape[0]
    print(f'{failed_bc} out of {result.shape[0]} cell barcodes failed for deconvolution...')
    max_col_indices = result.idxmax(axis=1)
    max_col_values = result.apply(lambda row: row[max_col_indices[row.name]], axis=1)
    result['max_indices'] = max_col_indices
    result['max_values'] = max_col_values
    result = result[result['max_values'] >= similarity]

    # Count the number of occurrences of each max_index
    # Filter results based on minimum cell count
    max_indices_number = Counter(result['max_indices'])
    est_cluster = [clu for clu, num in max_indices_number.items() if num >= min_cell_num]
    est_cluster = sorted(est_cluster, key=lambda x: max_indices_number[x], reverse=True)
    result = result[result['max_indices'].isin(est_cluster)]

    return result


def curve_fit(data, _filter=0.01, select_factor=0.8):
    """\
    Perform curve fitting and statistical analysis on the input data.

    Parameters
    ----------
    data
        input data containing 'mean_expr' and 'bg_mean_expr' columns.
    _filter
        a threshold for filtering data based on adjusted p-value (default is 0.01).
    select_factor
        a factor to determine the number of data points to select (default is 0.8).

    Returns
    -------
    tmp
        the result of curve fitting and analysis with added columns 'fit', 'distance', 'p.value', and 'p.adj'.
    """
    # Determine the number of data points to select based on the select_factor
    # Calculate the maximum value of 'mean_expr' in the input data
    # Generate a set of discrete x-values for smoothing
    number = round(data.shape[0] * select_factor)
    max_number = data['mean_expr'].max()
    discrete_x = np.arange(0, max_number + 0.001, 0.001)
    discrete_x = np.round(discrete_x, 3)

    # Randomly sample data points
    # Perform smoothing using the 'smooth_spline' function
    # Calculate the 'distance' between 'fit' and 'bg_mean_expr' 
    result = []
    for _ in range(10):
        tmp = data.sample(number, replace=False)
        tmp = tmp.sort_values(by='mean_expr')
        tmp['fit'] = smooth_spline(tmp['mean_expr'], tmp['bg_mean_expr'], tmp['mean_expr'])
        tmp['distance'] = tmp['fit'] - tmp['bg_mean_expr']
        tmp = tmp[tmp['distance'].notnull()]
        tmp['adjusted_p'] = 1 - norm.cdf(tmp['distance'], loc=tmp['distance'].mean(), scale=tmp['distance'].std())
        tmp = tmp[tmp['adjusted_p'] > _filter]
        tmp['fit'] = smooth_spline(tmp['mean_expr'], tmp['bg_mean_expr'], tmp['mean_expr'])
        tmp['distance'] = tmp['fit'] - tmp['bg_mean_expr']
        tmp = tmp[tmp['distance'].notnull()]
        tmp['adjusted_p'] = 1 - norm.cdf(tmp['distance'], loc=tmp['distance'].mean(), scale=tmp['distance'].std())
        tmp = tmp[tmp['adjusted_p'] > _filter]
        pred = smooth_spline(tmp['mean_expr'], tmp['bg_mean_expr'], discrete_x)
        result.append(pred)

    # Calculate the mean of the predictions from multiple iterations
    result = pd.DataFrame(result).mean(axis=0)

    # Merge the 'mean_expr', 'fit', and 'bg_mean_expr' DataFrames
    result = pd.DataFrame({'fit': list(result), 'mean_expr': discrete_x})
    mean_expr = pd.DataFrame(np.round(data['mean_expr'], 3))
    mean_expr['Gene'] = mean_expr.index
    bg_mean_expr = pd.DataFrame(data['bg_mean_expr'])
    bg_mean_expr['Gene'] = bg_mean_expr.index
    tmp_fit = pd.merge(mean_expr, result, on='mean_expr')
    tmp = pd.merge(tmp_fit, bg_mean_expr, on='Gene')

    # Calculate 'distance', 'p.value', and 'p.adj' values 
    tmp['distance'] = tmp['fit'] - tmp['bg_mean_expr']
    tmp['p.value'] = 1 - norm.cdf(tmp['distance'], loc=tmp['distance'].mean(), scale=tmp['distance'].std())
    tmp['p.adj'] = multipletests(tmp['p.value'], method='fdr_bh')[1]
    tmp = tmp.sort_values(by='distance', ascending=False)

    return tmp


def calculate_distance(counts_ave: pd.DataFrame, background_ave: pd.DataFrame) -> List[pd.DataFrame]:
    """\
    Perform data analysis using a smooth spline method (UnivariateSpline).

    Parameters
    ----------
    counts_ave
        A DataFrame containing gene average expression from counts data.
    background_ave
        A DataFrame containing gene average expression from background data.

    Returns
    -------
    distance_list
       A list of DataFrames containing analysis results for each cluster.
    """
    # Get the cluster names and Find common genes between counts_ave and background_ave
    est_cluster = background_ave.columns
    genes = set(counts_ave.index) & set(background_ave.index)
    counts_ave = counts_ave.loc[list(genes)]
    background_ave = background_ave.loc[list(genes)]

    # Apply log transformation to the data
    counts_ave = np.log2(counts_ave + 1).round(3)
    background_ave = np.log2(background_ave + 1).round(3)

    # calculate distance between actual background gene exp and expected gene exp
    distance_list = []
    for clu in est_cluster:
        gene_ave = pd.DataFrame({
            'Gene': list(genes),
            "mean_expr": counts_ave[clu],
            "bg_mean_expr": background_ave[clu]
        })
        gene_ave = curve_fit(gene_ave, 0.01, 0.8)
        distance_list.append(gene_ave)

    return distance_list


def smooth_spline(x: np.array, y: np.array, z: np.array):
    """\
    Perform a smooth spline interpolation of data.

    Parameters
    ----------
    x
        The x-values of the data points.
    y
        The y-values of the data points.

    z
        The values for prediction.

    Returns
    -------
    prd
        The smoothed spline interpolation of the data.
    """
    fit = UnivariateSpline(x, y)
    prd = fit(z)

    return prd


def contaminated_genes_detection(
        counts: AnnData,
        background_counts: AnnData,
        usages: pd.DataFrame,
        spectra: pd.DataFrame,
        min_prop: float = 0.1,
        similarity: float = 0.9,
        min_cell_num: float = 10,
        exp_pct: float = 0,
        top_cont_genes: int = 20,
        slot: str = 'counts'
) -> object:
    """\
    Detect contaminated genes in gene expression data.

    Parameters
    ----------
    counts
        An AnnData object containing gene expression data.
    background_counts
        An AnnData object containing background counts data.
    usages
        A DataFrame containing gene usage data.
    spectra
        A DataFrame containing spectra data.
    min_prop
        The minimum proportion of components to consider in deconvolution. Default is 0.1.
    similarity
        The similarity threshold for component selection in deconvolution. Default is 0.9.
    min_cell_num
        The minimum number of cells to consider for a component. Default is 10.
    exp_pct
        The minimum percentage of genes expressed in potential ambient clusters. Default is 0.
    top_cont_genes
        The number of ambient genes shown and to calculate ambient level. Default is 20.
    slot
        The slot name for data extraction. Default is 'counts'.

    Returns
    -------
    contaminated_genes
        A list of contaminated genes.
    distance_result
        A DataFrame containing distance results.
    """
    # do the deconvolution and filter background barcodes 
    df = pd.concat([counts.obs['cluster'], usages], axis=1)
    ref = df.groupby('cluster').median().T
    print('Find background barcodes that exhibit similar expression patterns to the foreground cluster...')
    deconv_result = deconvolution(background_counts, spectra, ref, min_prop, similarity, min_cell_num)
    if len(deconv_result) == 0:
        raise Exception('No barcodes in background similar to cluster from real cell barcodes')
    background_counts = background_counts[deconv_result.index.tolist(), :]
    background_counts.obs['cluster'] = deconv_result['max_indices'].astype('category')
    print(f'{background_counts.shape[0]} background cells will be used to perform smooth spline...')

    # calculate average expression of each cluster in real cells and background
    background_ave = calculate_average_expression(background_counts, list(background_counts.var_names), slot)
    counts_ave = calculate_average_expression(counts, list(counts.var_names), slot)

    print('Perform smooth spline...')
    all = calculate_distance(counts_ave, background_ave)

    # filter p.adj of genes in each cluster if larger than 0.05
    contaminated_genes = list({gene for tmp in all for gene in tmp.loc[tmp['p.adj'] <= 0.05, 'Gene']})

    # only need genes existed in N background clusters
    exp_percent = calculate_expression_percentage(counts, contaminated_genes, slot)
    filtered_contaminated_genes = [g for g in contaminated_genes if exp_percent.loc[g].min() > exp_pct]

    # evacuate contamination level
    distance_result = pd.concat(all)
    distance_result = (
        distance_result
        .loc[
            distance_result['Gene'].isin(filtered_contaminated_genes) &
            (distance_result['p.adj'] <= 0.05) &
            (distance_result['bg_mean_expr'] > 0)
            ]
        .assign(cont_pct=lambda df: df['distance'] / df['mean_expr'])
        .groupby('Gene')
        .agg(average_distance=('distance', 'mean'), average_cont_pct=('cont_pct', 'mean'))
    )

    sorted_average_distances = distance_result.sort_values(by='average_distance', ascending=False)
    sorted_average_distances['is_top'] = [True] * top_cont_genes + \
                                         [False] * (len(sorted_average_distances) - top_cont_genes)
    top_contaminated_genes = sorted_average_distances.head(top_cont_genes).index.tolist()
    contaminated_percent = sorted_average_distances.loc[top_contaminated_genes, 'average_cont_pct'].mean()

    contamination_metric = {'Top Contamination_Genes': ",".join(top_contaminated_genes),
                            'Contamination_Level': contaminated_percent}
    print(f'Contamination Genes: {",".join(top_contaminated_genes)}')
    print(f'Contamination Level: {round(contaminated_percent, 3)}')

    return sorted_average_distances, contamination_metric


def calculate_jaccard_similarity(set1: set, set2: set) -> float:
    """\
    Calculate the Jaccard similarity between two sets.

    Parameters
    ----------
    set1
        The first set for Jaccard similarity calculation.
    set2
        The second set for Jaccard similarity calculation.

    Returns
    -------
        The Jaccard similarity score between the two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union


def find_optimal_cutoff(TPR, FPR, threshold):
    """\
    Find the optimal cutoff based on true positive rate and false positive rate.

    Parameters
    ----------
    TPR
        true positive rates.
    FPR
        false positive rates.
    threshold
        threshold values.

    Returns
    -------
    A tuple containing the optimal threshold value and the corresponding point.
    """
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]

    return optimal_threshold, point


def cal_ambient_threshold(
        exp: pd.Series,
        neg_cells: list,
        pos_cells: list,
        gene: str,
):
    pos_cells_length = len(pos_cells)
    neg_cells_length = len(neg_cells)
    label = np.concatenate((np.ones(pos_cells_length), np.zeros(neg_cells_length)))
    exp_class = pd.concat([exp.loc[pos_cells], exp.loc[neg_cells]])
    exp_class = pd.DataFrame(exp_class).T
    exp_class.columns = label
    a = exp_class.T.sort_values(by=gene)[gene].tolist()
    b = exp_class.T.sort_values(by=gene).index.tolist()
    fpr, tpr, thresholds = roc_curve(b, a)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = find_optimal_cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return optimal_th, roc_auc


def process_topic(topic, position, common_topic_dict, topic_dict, exp, counts_ave, cluster_dict, roc_threshold):
    pbar = tqdm(total=len(common_topic_dict[topic]), desc=f"Processing {topic}", position=position, leave=True)
    corrected_exp = pd.DataFrame()
    cont_pos_cluster = topic_dict[topic]

    # correct for each gene
    for gene in common_topic_dict[topic]:
        tmp_exp = exp[gene]
        tmp_pos_cluster, tmp_pos_exp = [counts_ave.loc[gene, cont_pos_cluster].idxmax()], \
            counts_ave.loc[gene, cont_pos_cluster].max()

        # need to remove some cluster
        for i in cont_pos_cluster:
            pos_cells = cluster_dict[tmp_pos_cluster[0]]
            neg_cells = cluster_dict[i]
            ambient_threshold, roc_auc = cal_ambient_threshold(
                exp=tmp_exp,
                pos_cells=pos_cells,
                neg_cells=neg_cells,
                gene=gene)
            if roc_auc <= roc_threshold:
                tmp_pos_cluster.append(i)
        tmp_pos_cluster = list(set(tmp_pos_cluster).union(counts_ave.columns[counts_ave.loc[gene] > tmp_pos_exp]))
        tmp_neg_cluster = [i for i in counts_ave.columns if i not in tmp_pos_cluster]

        # start roc
        pos_cells = list(chain.from_iterable(cluster_dict[i] for i in tmp_pos_cluster))
        neg_cells = list(chain.from_iterable(cluster_dict[i] for i in tmp_neg_cluster))
        ambient_threshold, roc_auc = cal_ambient_threshold(
            exp=tmp_exp,
            pos_cells=pos_cells,
            neg_cells=neg_cells,
            gene=gene)
        gene_exp = exp[gene] - (0 if np.isinf(ambient_threshold) else ambient_threshold)
        corrected_exp = pd.concat([corrected_exp, gene_exp], axis=1)
        pbar.update(1)

    pbar.close()

    return topic, corrected_exp


def contaminated_genes_correction(
        counts: AnnData,
        usages: pd.DataFrame,
        spectra: pd.DataFrame,
        average_distance: pd.DataFrame,
        filtered_mtx_path: str,
        roc_threshold: float = 0.6,
        num_processes: int = 4,
        raw_counts_slot: str = 'counts'
):
    # extract contaminated genes identified in previous step
    cont_genes = list(average_distance.index)

    # find topics that have at least two clusters contributing the most to them
    df = pd.concat([counts.obs['cluster'], usages], axis=1)
    ref = df.groupby('cluster').median().T
    norm_ref = ref / ref.sum(0)

    common_topic = list(norm_ref.idxmax(axis=0).value_counts().index)
    print(f'Potential ambient topic include {",".join(common_topic)}...')

    # find clusters which support each topic identified
    topic_dict = {}
    for topic in common_topic:
        topic_dict[topic] = norm_ref.idxmax(axis=0)[norm_ref.idxmax(axis=0) == topic].index.tolist()

    # contaminated genes always express higher in the related source contaminated cluster
    counts_ave = calculate_average_expression(counts, cont_genes, raw_counts_slot)
    gene_exp_high_df = counts_ave[
        [cluster for clusters in topic_dict.values() for cluster in clusters]
    ].apply(lambda row: row.nlargest(2).index.tolist(), axis=1)

    gene_exp_high_df = pd.DataFrame(gene_exp_high_df.tolist(), index=counts_ave.index, columns=['Top_1', 'Top_2'])
    gene_exp_high_df = gene_exp_high_df.applymap(
        lambda cluster: {c: t for t, c in topic_dict.items() for c in c}.get(cluster))

    # create gene family
    gene_family = gene_exp_high_df[gene_exp_high_df['Top_1'] == gene_exp_high_df['Top_2']]['Top_1'].to_dict()
    cont_genes_2 = gene_exp_high_df[gene_exp_high_df['Top_1'] != gene_exp_high_df['Top_2']]
    spectra_filtered_cor = spectra.loc[cont_genes].T.corr().loc[cont_genes_2.index]

    for gene, row in spectra_filtered_cor.iterrows():
        similar_genes = row[row >= 0.95].index.difference([gene])
        if similar_genes.empty or not set(similar_genes).intersection(gene_family):
            gene_family[gene] = cont_genes_2.loc[gene, 'Top_1']
        else:
            gene_family[gene] = gene_family[next(iter(set(similar_genes).intersection(gene_family)))]

    # find more contaminated genes
    spectra_cor = spectra.T.corr()
    spectra_cor = spectra_cor.loc[[gene for gene in spectra_cor.index if gene not in cont_genes]]
    rescue_genes = {}
    for gene, row in spectra_cor.iterrows():
        similar_genes = row[row >= 0.95].index.difference([gene])
        existing_genes = set(similar_genes).intersection(gene_family)
        if similar_genes.empty or not existing_genes:
            pass
        else:
            related_gene = next(iter(existing_genes))
            rescue_genes[gene] = gene_family[related_gene]

    gene_family = {**gene_family, **rescue_genes}
    gene_family_df = pd.DataFrame(gene_family, index=[0]).T
    common_topic_dict = defaultdict(list)
    for gene, topic in gene_family.items():
        common_topic_dict[topic].append(gene)

    rescue_cont_genes = list(set(gene_family))
    print(f'Totally find {len(rescue_cont_genes)} ambient genes...')

    try:
        counts.X = counts.layers[raw_counts_slot].copy()
    except ValueError:
        del counts.X
        counts.X = counts.layers[raw_counts_slot].copy()

    exp = counts.to_df()
    counts_ave = calculate_average_expression(counts, rescue_cont_genes, raw_counts_slot)

    # start remove ambient expression
    print('Start removing ambient expression...')
    counts.obs['barcodes'] = counts.obs_names
    cluster_dict = counts.obs.groupby('cluster')['barcodes'].apply(list).to_dict()

    positions = list(range(num_processes))
    extended_positions = positions * (len(common_topic_dict) // num_processes) + positions[:len(
        common_topic_dict) % num_processes]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_topic,
                               [(
                                   topic, pos, common_topic_dict, topic_dict, exp, counts_ave, cluster_dict,
                                   roc_threshold)
                                   for topic, pos in zip(common_topic_dict.keys(), extended_positions)]
                               )

    corrected_exp_dict = {topic: corrected_exp for topic, corrected_exp in results}
    corrected_exp = pd.concat(corrected_exp_dict.values(), axis=1).applymap(lambda x: max(x, 0))
    raw_counts = _load_data(filtered_mtx_path, verbose=False)
    raw_counts_exp = raw_counts.to_df()
    raw_counts_exp = pd.concat([raw_counts_exp.drop(rescue_cont_genes, axis=1), corrected_exp], axis=1)
    raw_counts_exp = raw_counts_exp.dropna()
    clear_data = AnnData(X=raw_counts_exp.astype("int"))
    print('Finish...')

    return clear_data, gene_family_df, common_topic


def _init_nmf(
        counts: AnnData,
        rank_genes: pd.DataFrame,
        n_components: int,
        beta_loss: str = 'frobenius',
        init: str = 'random',
        solver: str = 'cd',
        alpha_usage: float = 0.00,
        alpha_spectra: float = 0.00,
        l1_ratio: float = 0.00,
        tol: float = 1e-4,
        max_iter: int = 1000,
        random_state: int = 123
) -> dict:
    """\
    Define a dictionary with parameters for Non-negative Matrix Factorization (NMF).

    Parameters
    ----------
    counts
        input annotated data containing gene expression counts.
    rank_genes
        a DataFrame with ranked genes and their properties.
    n_components
        the number of components or topics to extract using NMF.
    beta_loss
        the loss function used in NMF (default is 'frobenius').
    init
        the initialization method for NMF (default is 'random').
    solver
        the solver algorithm for NMF (default is 'cd').
    alpha_usage
        the regularization parameter for the usage matrix (default is 0.00).
    alpha_spectra
        the regularization parameter for the spectra matrix (default is 0.00).
    l1_ratio
        the L1 regularization ratio (default is 0.00).
    tol
        the tolerance value for convergence (default is 1e-4).
    max_iter
        the maximum number of iterations for NMF (default is 1000).
    random_state
        seed for sklearn random state

    Returns
    -------
    _nmf_kwargs
        a dictionary containing the NMF parameters.
    """
    ks = counts.obs['cluster'].cat.categories
    rank_genes = rank_genes[rank_genes['names'].isin(counts.var_names)]

    # define W value by logfoldchanges {Gene Number (row) x Topic Number (col)}
    tmp = rank_genes.set_index('names')
    tmp = tmp[tmp['group'].isin(ks)][['group', 'logfoldchanges']]
    W = tmp.pivot(columns='group', values='logfoldchanges')
    W = W.reindex(counts.var_names)[ks].fillna(1e-12)
    W.columns = ['Topic_' + i for i in W.columns]

    # define H value by ident number {Topic Number (row) x Cell Number (col)}
    H = pd.crosstab(index=counts.obs['cluster'], columns=counts.obs_names)
    H = H.replace(0, 1e-12)
    H.index = ['Topic_' + i for i in H.index]

    # define nmf parameters
    _nmf_kwargs = dict(
        alpha_W=alpha_usage,
        alpha_H=alpha_spectra,
        l1_ratio=l1_ratio,
        beta_loss=beta_loss,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
        init=init,
        n_components=n_components,
        W=W,
        H=H,
        random_state=random_state
    )

    return _nmf_kwargs


def _scale_data(counts: AnnData, slot: str = 'counts') -> sparse.spmatrix:
    """
    Scale the gene expression data.
    """

    counts_tmp = counts.copy()
    sc.pp.scale(counts_tmp, zero_center=False, layer=slot)
    X = counts_tmp.layers[slot]

    return X


def _nmf(X, nmf_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Non-Negative Matrix Factorization (NMF).
    """

    (usages, spectra, niter) = non_negative_factorization(X, **nmf_kwargs)

    return spectra, usages


def _exclude_genes(
        counts: AnnData,
        black_gene_list: List[str] = None
) -> AnnData:
    """\
    exclude genes that are not expressed or
    out of concern in order to reduce the technical noise.
    """
    counts.var['counts'] = np.array(counts.X.sum(axis=0)).squeeze()
    keep_genes = counts.var[counts.var['counts'] > 0]
    counts = counts[:, keep_genes.index]

    excluded_patterns = "^(Rp[s|l]|MT|mt|Gm)"
    white_gene_list = [
        name for name in counts.var_names
        if not re.search(
            excluded_patterns,
            name,
            flags=re.IGNORECASE)
    ]

    if black_gene_list:
        white_gene_list = [i
                           for i in white_gene_list
                           if i not in black_gene_list]

    counts = counts[:, white_gene_list]

    return counts
