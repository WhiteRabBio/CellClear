suppressMessages({
library(celda)
library(Seurat)
library(DropletUtils)
library(argparser)
library(SingleCellExperiment)
})

run_decontx <- function(toc, tod, prefix, outdir, delta = NULL) {
toc <- Read10X_h5(toc)
tod <- Read10X_h5(tod)
tod <- tod[rownames(toc), ]
raw <- tod
raw <- CreateSeuratObject(raw)
raw1 <- as.SingleCellExperiment(raw)

all <- toc
all <- CreateSeuratObject(all)
all <- NormalizeData(all, normalization.method = "LogNormalize", scale.factor = 10000)
all <- FindVariableFeatures(all, selection.method = "vst", nfeatures = 3000)
all.genes <- rownames(all)
all <- ScaleData(all, features = all.genes)
all <- RunPCA(all, features = VariableFeatures(all), npcs = 50, verbose = FALSE)
all <- FindNeighbors(all, dims = 1:30)
all <- FindClusters(all, resolution = 1.0)
all <- RunUMAP(all, dims = 1:30)
matx <- all@meta.data
all1 <- as.SingleCellExperiment(all)

if (delta == "None") {
all1 <- decontX(all1, background = raw1, z = all@meta.data$seurat_clusters)
}else{
all1 <- decontX(all1, background = raw1, z = all@meta.data$seurat_clusters, delta = c(9, as.numeric(delta)), estimateDelta = FALSE)    
}

out <- round(all1@assays@data$decontXcounts)
pdf(paste0(outdir, "/", prefix, "_decontx_contamination.pdf"))
pic <- plotDecontXContamination(all1)
print(pic)
dev.off()
DropletUtils:::write10xCounts(paste0(outdir, "/", prefix, "_decontx_matrix"), out, version = "3")
}


argv <- arg_parser('decontx remove contamination')
argv <- add_argument(argv, "--raw_mtx", help = "raw matrix file")
argv <- add_argument(argv, "--filter_mtx", help = "filtered matrix file")
argv <- add_argument(argv, "--delta", default = "None", help = "priors to influence contamination estimates, set larger than 15 if use")
argv <- add_argument(argv, "--prefix", help = "output prefix")
argv <- add_argument(argv, "--outdir", help = "the output dir")
argv <- parse_args(argv)

run_decontx <- run_decontx(argv$filter_mtx, argv$raw_mtx, argv$prefix, argv$outdir, argv$delta)
