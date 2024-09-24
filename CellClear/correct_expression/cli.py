"""Command-line tool functionality for correct_expression."""

from CellClear.correct_expression.correct_expression import (
    _preprocess_data,
    identify_module,
    contaminated_genes_detection,
    contaminated_genes_correction,
)
from CellClear.correct_expression.utils import output_10x_matrix
from CellClear.correct_expression.plot import plot_usages
from CellClear.base_cli import AbstractCLI
import json
import os


class CLI(AbstractCLI):
    """CLI implements AbstractCLI from the CellClear package."""

    def __init__(self):
        self.name = 'correct_expression'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def validate_args(self, args):
        """Validate parsed arguments."""

        try:
            args.filtered_matrix = os.path.expanduser(args.filtered_matrix)
            args.raw_matrix = os.path.expanduser(args.raw_matrix)
        except TypeError:
            raise ValueError(
                "Problem with provided input paths."
            )

        self.args = args

        return args

    def run(self, args):
        """Run the main tool functionality on parsed arguments."""

        # Run the tool.
        main(args)


def correct_expression(args):
    """The full script for the command line tool to run correct_expression.
    Args:
        args: Inputs from the command line, already parsed using argparse.
    Note: Returns nothing, but writes output to a file(s) specified from
        command line.
    """
    filtered_counts, background_counts = _preprocess_data(
        filtered_mtx_path=args.filtered_matrix,
        raw_mtx_path=args.raw_matrix,
        resolution=args.resolution,
        min_background_counts_num=args.min_bg_num,
        environ_range=[int(args.min_environ_umi), int(args.max_environ_umi)])
    usages, spectra, _nmf_kwargs = identify_module(
        counts=filtered_counts)
    sorted_average_distances, contamination_metric = contaminated_genes_detection(
        counts=filtered_counts,
        background_counts=background_counts,
        usages=usages,
        spectra=spectra,
    )
    clear_data, gene_family_df, common_topic = contaminated_genes_correction(
        counts=filtered_counts,
        usages=usages,
        spectra=spectra,
        average_distance=sorted_average_distances,
        filtered_mtx_path=args.filtered_matrix,
        roc_threshold=args.roc_threshold,
        num_processes=4,
        raw_counts_slot='counts'
    )

    # output
    output_10x_matrix(clear_data, f'{args.output_dir}/matrix')
    with open(f'{args.output_dir}/metric.json', 'w') as json_file:
        json.dump(contamination_metric, json_file)
    filtered_counts.write(f'{args.output_dir}/counts.h5ad', compression='lzf')
    usages.to_csv(f'{args.output_dir}/usages.csv', sep=',')
    spectra.to_csv(f'{args.output_dir}/spectra.csv', sep=',')
    sorted_average_distances.to_csv(f'{args.output_dir}/distance.csv', sep=',')
    plot_usages(filtered_counts, usages, spectra, common_topic, list(gene_family_df.index), f'{args.output_dir}/topic')


def main(args):
    """Take command-line input, parse arguments, and run tests or tool."""

    # Run the tool.
    correct_expression(args)
