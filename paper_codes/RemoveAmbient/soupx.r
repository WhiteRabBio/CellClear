suppressMessages({
library(SoupX)
library(Seurat)
library(DropletUtils)
library(argparser)
})

run_soupx <- function(toc, tod, prefix, outdir, rho = NULL, geneset = NULL) {
toc <- Read10X_h5(toc)
tod <- Read10X_h5(tod)
tod <- tod[rownames(toc), ]

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

sc <- SoupChannel(tod, toc)
sc <- setClusters(sc, setNames(matx$seurat_clusters, rownames(matx)))

if (geneset == "None") {
if (rho == "None") {
tryCatch(
{sc <- autoEstCont(sc)},
error = function(e) {
print("autoEstCont Error !")
sc <- setContaminationFraction(sc, 0.2)}
)
}else{
sc <- setContaminationFraction(sc, as.numeric(rho))
}
}else{
geneset <- unlist(strsplit(geneset, split = ","))
geneset <- intersect(geneset, rownames(toc))
print(geneset)
nonExpressedGeneList <- list(Genes = geneset)
useToEst <- estimateNonExpressingCells(sc, nonExpressedGeneList = nonExpressedGeneList)
sc <- calculateContaminationFraction(sc, nonExpressedGeneList, useToEst = useToEst)
}

out <- adjustCounts(sc, roundToInt = TRUE)
DropletUtils:::write10xCounts(paste0(outdir, "/", prefix, "_soupX_matrix"), out, version = "3")
return (out)
} 

argv <- arg_parser('soupx remove contamination')
argv <- add_argument(argv, "--raw_mtx", help = "raw matrix file")
argv <- add_argument(argv, "--filter_mtx", help = "filtered matrix file")
argv <- add_argument(argv, "--rho", default = "None", help = "contamination fraction if set manually")
argv <- add_argument(argv, "--geneset", default = "None", help = "soup specific genes if set maunally")
argv <- add_argument(argv, "--prefix", help = "output prefix")
argv <- add_argument(argv, "--outdir", help = "the output dir")
argv <- parse_args(argv)

run_soupx <- run_soupx(argv$filter_mtx, argv$raw_mtx, argv$prefix, argv$outdir, argv$rho, argv$geneset)
