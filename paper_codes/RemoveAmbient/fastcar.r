suppressMessages({
library(Matrix)
library(Seurat)
library(qlcMatrix)
library(FastCAR)
})

run_fastcar <- function(toc, tod, emptyDropletCutoff=100, contaminationChanceCutoff=0.05, prefix, outdir) {
    cellMatrix <- Read10X_h5(toc)
    fullMatrix <- Read10X_h5(tod)

    ambientProfile <- determine.background.to.remove(fullMatrix, emptyDropletCutoff, contaminationChanceCutoff)
    cellMatrix <- remove.background(cellMatrix, ambientProfile)

    write.table(cellMatrix, paste0(outdir, "/", prefix, "_fastcar_matrix.tsv"), sep='\t')
}

argv <- arg_parser('fastcar remove contamination')
argv <- add_argument(argv, "--raw_mtx", help = "raw matrix file")
argv <- add_argument(argv, "--filter_mtx", help = "filtered matrix file")
argv <- add_argument(argv, "--emptyDropletCutoff", default = 100, help = "emptyDropletCutoff")
argv <- add_argument(argv, "--contaminationChanceCutoff", default = 0.05, help = "contaminationChanceCutoff")
argv <- add_argument(argv, "--prefix", help = "output prefix")
argv <- add_argument(argv, "--outdir", help = "the output dir")
argv <- parse_args(argv)

run_fastcar(argv$filter_mtx, argv$raw_mtx, argv$emptyDropletCutoff, argv$contaminationChanceCutoff, argv$prefix, argv$outdir)
