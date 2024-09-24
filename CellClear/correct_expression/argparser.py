import argparse


def add_subparser_args(subparsers: argparse) -> argparse:
    """Add tool-specific arguments for correct_expression.

    Args:
        subparsers: Parser object before addition of arguments specific to
            correct_expression.

    Returns:
        parser: Parser object with additional parameters.

    """

    subparser = subparsers.add_parser("correct_expression",
                                      description="correct ambient gene expression",
                                      help="Identify ambient genes and "
                                           "correct their expression in counts matrix",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser.add_argument("--filtered_mtx_path", nargs=None, type=str,
                           dest='filtered_matrix',
                           required=True,
                           help="matrix after cell calling, "
                                "support `filtered_feature_bc_matrix` from cellranger"
                                "or a normal matrix (row as genes, column as barcodes")
    subparser.add_argument("--raw_mtx_path", nargs=None, type=str,
                           dest='raw_matrix',
                           required=True,
                           help="matrix before cell calling, "
                                "support `raw_feature_bc_matrix` from cellranger"
                                "or a normal matrix (row as genes, column as barcodes")
    subparser.add_argument("--min_environ_umi", nargs=None,
                           dest='min_environ_umi', default=60,
                           required=False,
                           help="CellClear needs to extract information from background,"
                                "low UMI barcodes will contain pure ambient gene expression")
    subparser.add_argument("--max_environ_umi", nargs=None,
                           dest='max_environ_umi', default=100,
                           required=False,
                           help="CellClear needs to extract information from background,"
                                "low UMI barcodes will contain pure ambient gene expression")
    subparser.add_argument("--min_background_counts_num", nargs=None, type=int,
                           dest='min_bg_num', default=5000,
                           required=False,
                           help="Minimum number of background barcodes extract for analysis")
    subparser.add_argument("--black_gene_list", nargs=None, type=str,
                           dest='black_gene_list', default=None,
                           required=False,
                           help="genes wanted to exclude from analysis, "
                                "those genes will not be identified as ambient genes")
    subparser.add_argument("--roc_threshold", nargs=None, type=float,
                           dest='roc_threshold', default=0.6,
                           required=False,
                           help="roc threshold to separate ambient and clean cluster")
    subparser.add_argument("--resolution", nargs=None, type=float,
                           dest='resolution', default=1.2,
                           required=False,
                           help="resolution for clustering")
    subparser.add_argument("--output", nargs=None, type=str,
                           dest='output_dir', default=None,
                           required=True,
                           help="Output file location.")
    subparser.add_argument("--prefix", nargs=None, type=str,
                           dest='prefix', default=None,
                           required=True,
                           help="Prefix of output results.")

    return subparsers
