library(dplyr)
library(ggplot2)
library(reshape2)
library(ggpubr)
library(scales)



clustcol<-c("OrangeRed","SlateBlue3","DarkOrange","GreenYellow","Purple","DarkSlateGray","Gold","DarkGreen","DeepPink2","Red4","#4682B4","#FFDAB9","#708090","#836FFF","#CDC673","#CD9B1D","#FF6EB4","#CDB5CD","#008B8B","#43CD80","#483D8B","#66CD00","#CDC673","#CDAD00","#CD9B9B","#FF8247","#8B7355","#8B3A62","#68228B","#CDB7B5","#CD853F","#6B8E23","#696969","#7B68EE","#9F79EE","#B0C4DE","#7A378B","#66CDAA","#EEE8AA","#00FF00","#EEA2AD","#A0522D","#000080","#E9967A","#00CDCD","#8B4500","#DDA0DD","#EE9572","#EEE9E9","#8B1A1A","#8B8378","#EE9A49","#EECFA1","#8B4726","#8B8878","#EEB4B4","#C1CDCD","#8B7500","#0000FF","#EEEED1","#4F94CD","#6E8B3D","#B0E2FF","#76EE00","#A2B5CD","#548B54","#BBFFFF","#B4EEB4","#00C5CD","#008B8B","#7FFFD4","#8EE5EE","#43CD80","#68838B","#00FF00","#B9D3EE","#9ACD32","#00688B","#FFEC8B","#1C86EE","#CDCD00","#473C8B","#FFB90F","#EED5D2","#CD5555","#CDC9A5","#FFE7BA","#FFDAB9","#CD661D","#CDC5BF","#FF8C69","#8A2BE2","#CD8500","#B03060","#FF6347","#FF7F50","#CD0000","#F4A460","#FFB5C5","#DAA520","#CD6889","#32CD32","#FF00FF","#2E8B57","#CD96CD","#48D1CC","#9B30FF","#1E90FF","#CDB5CD","#191970","#E8E8E8","#FFDAB9")


cont_genes <- read.table("metric/cont_genes.csv", header=F)$V1 %>%
    as.character()


difflist <- list(
    cellclear = "metric/CellClear_SRR21882339/CellClear_SRR21882339_diffgenes.list",
    uncorrected = "metric/Uncorrected_SRR21882339/Uncorrected_SRR21882339_diffgenes.list"
)


difftables = lapply(difflist, function(x) {
    path_table = read.table(x, header=F, stringsAsFactors=F)
    celltypes = path_table$V1
    tables = lapply(celltypes, function(c) {
        path = subset(path_table, V1 == c)[,'V2']
        diffgenes = read.table(path, header=T, row.names=1, stringsAsFactors=F)
        return(diffgenes)
    })
    names(tables) = celltypes
    return(tables)
})


numbers = lapply(difftables, function(diff) {
    stat = lapply(diff, function(c) {
        up_df = c[c$logfoldchanges > 0,]
        down_df = c[c$logfoldchanges < 0,]
        out = list(
            n_up = nrow(up_df),
            n_down = nrow(down_df),
            up_genes = rownames(up_df),
            down_genes = rownames(down_df)
        )
        return(out)
    })
    return(stat)
})


outdir = 'figure/DEG_UP/'
outdir_unique = file.path(outdir, 'unique_list')
dir.create(outdir_unique)
setwd(outdir_unique)

genesets = lapply(numbers, function(x) {
    lapply(x, function(y) y[['up_genes']])
})

celltypes <- c('RSC-PyNs', 'SubC-PyNs', 'CA1-PyNs', 'CA3-PyNs', 'DG-GCs', 'NSCs',
               'Interneurons', 'IPs', 'CR', 'Endo', 'Erythrocytes', 'Microglial')
for (i in celltypes) {
    cc_name = paste0(i, '_cc')
    uc_name = paste0(i, '_uc')

    prefix_cc = paste0(cc_name, '.specific')
    prefix_uc = paste0(uc_name, '.specific')

    prefix_common = paste0(i, '.common')
    glist_cc = genesets[['cellclear']][[i]]
    glist_uc = genesets[['uncorrected']][[i]]
    df_uc = data.frame(genelist = glist_uc[!glist_uc %in% glist_cc])
    df_cc = data.frame(genelist = glist_cc[!glist_cc %in% glist_uc])
    df_common = data.frame(genelist = intersect(glist_uc, glist_cc))
    write.table(df_uc, prefix_uc, row.names=F, col.names=F, quote=F, sep='\t')
    write.table(df_cc, prefix_cc, row.names=F, col.names=F, quote=F, sep='\t')
    write.table(df_common, prefix_common, row.names=F, col.names=F, quote=F, sep='\t')

}


################ upupup
outdir = 'figure/DEG_UP/'
dir.create(outdir)
setwd(outdir)
genesets = lapply(numbers, function(x) {
    lapply(x, function(y) y[['up_genes']])
})

batches = names(genesets)


df_bar = sapply(batches, function(x){
    n = sapply(celltypes, function(y) length(genesets[[x]][[y]]))
    return(n)
})

counts_data = melt(df_bar)
colnames(counts_data) = c('celltype', 'method', 'n_total')
intersected_counts = sapply(rownames(counts_data), function(x){
    ct = counts_data[x, 'celltype'] %>% as.character()
    met = counts_data[x, 'method'] %>% as.character()
    n_total = counts_data[x, 'n_total'] %>% as.integer()
    other_met = batches[batches != met]
    met_gs = genesets[[met]][[ct]]
    other_met_gs = genesets[[other_met]][[ct]]

    n_intersect = length(intersect(met_gs, other_met_gs))
    n_unique = n_total - n_intersect
    pct_intersect = round(n_intersect/n_total, 4)
    pct_unique = 1 - pct_intersect
    pct_total = 1

    n_intersect_conta = length(intersect(met_gs, cont_genes))
    n_not_conta = n_total - n_intersect_conta
    pct_intersect_conta = round(n_intersect_conta/n_total, 4)
    pct_not_conta = 1 - pct_intersect_conta

    pct_total_conta = 1

    out = c(n_intersect, n_unique, pct_intersect, pct_unique, pct_total, -n_intersect_conta, -n_not_conta, -pct_intersect_conta, -pct_not_conta, -pct_total_conta)
    names(out) = c('n_intersect', 'n_unique', 'pct_intersect', 'pct_unique', 'pct_total','n_intersect_conta', 'n_not_conta', 'pct_intersect_conta', 'pct_not_conta', 'pct_total_conta')
    return(out)
})
intersected_counts_use = as.data.frame(intersected_counts) %>% t
counts_data = cbind(counts_data, intersected_counts_use)


p1 = ggplot(counts_data, aes(x = method)) +
        geom_bar(
            aes(y = pct_total_conta),
            fill= '#fdae61',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) + 
        geom_bar(
            aes(y = pct_intersect_conta),
            fill= '#d73027',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) + 
        geom_bar(
            aes(y = pct_total),
            fill= '#abd9e9',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) +
        geom_bar(
            aes(y = pct_unique),
            fill= '#4575b4',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) +
        geom_point(
            aes(y = rescale(n_total, c(0,1))),
            size = 5,
            shape=19
        ) +
        scale_y_continuous(
            "total counts",
            sec.axis = sec_axis(~ . * max(counts_data$n_total))
        ) +
        facet_grid( ~ celltype,  space  = "free", scales = "free", drop = TRUE, shrink = TRUE)

p1 <- p1 + theme_classic() + theme(
  axis.text.y = element_text(size = 12),  # 主坐标轴 x
  axis.text.x = element_text(size = 12, angle = 45, hjust = 1),  # 主坐标轴 y
  axis.text.y.right = element_text(size = 12),  # 次坐标轴 y
  axis.title.y.right = element_text(size = 12),  # 次坐标轴标题
)


png(file.path(outdir, 'test.bar.2.png'), width=900)
print(p1)
dev.off()
pdf(file.path(outdir, 'test.bar.2.pdf'), width=12)
print(p1)
dev.off()


p1 = ggplot(counts_data, aes(x = method)) +
        geom_bar(
            aes(y = pct_total_conta),
            fill= '#fdae61',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) + 
        geom_bar(
            aes(y = pct_intersect_conta),
            fill= '#d73027',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) + 
        geom_point(
            aes(y = rescale(n_intersect_conta, c(0,1))),
            size = 3,
            shape=2
        ) +
        geom_bar(
            aes(y = pct_total),
            fill= '#abd9e9',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) +
        geom_bar(
            aes(y = pct_unique),
            fill= '#4575b4',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) +
        geom_point(
            aes(y = rescale(n_unique, c(0,-1))),
            size = 3,
            shape=6
        ) +
        facet_grid( ~ celltype,  space  = "free", scales = "free", drop = TRUE, shrink = TRUE)

p1 = p1 + theme_classic()

png(file.path(outdir, 'test.bar.2.png'), width=1000)
print(p1)
dev.off()
pdf(file.path(outdir, 'test.bar.2.pdf'), width=10)
print(p1)
dev.off()


p1 = ggplot(counts_data, aes(x = method)) +
        geom_bar(
            aes(y = n_total),
            fill= '#E69F00',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) + 
        geom_bar(
            aes(y = n_intersect_conta),
            fill= 'red',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) + 
        geom_point(
            aes(y = rescale(pct_intersect_conta, c(0,500))),
            size = 3,
            shape=2
        ) +
        geom_bar(
            aes(y = n_unique),
            fill= '#56B4E9',
            stat = "identity",
            position = position_dodge(0.8),
            width = 0.7
        ) +
        geom_point(
            aes(y = rescale(pct_unique, c(0,-500))),
            size = 3,
            shape=6
        ) +
        facet_grid( ~ celltype,  space  = "free", scales = "free", drop = TRUE, shrink = TRUE)

p1 = p1 + theme_classic()

png(file.path(outdir, 'test.bar.png'), width=1000)
print(p1)
dev.off()
pdf(file.path(outdir, 'test.bar.pdf'), width=10)
print(p1)
dev.off()

#######################################################intersection_barcode#############################################
library(Seurat)
library(dplyr)
library(ggplot2)
library(reshape2)
library(pheatmap)

cal_shared= function(obj_raw, obj_clean) {
    barcodes_raw = data.frame(barcodes=colnames(obj_raw), celltypes=Idents(obj_raw))
    barcodes_raw$barcodes2 = sapply(barcodes_raw$barcodes, function(x) {
        unlist(strsplit(x, split='_'))[-1]
    })
    barcodes_clean = data.frame(barcodes=colnames(obj_clean), celltypes=Idents(obj_clean))
    barcodes_clean$barcodes2 = sapply(barcodes_clean$barcodes, function(x) {
        unlist(strsplit(x, split='_'))[-1]
    })
    ct_raw = rev(levels(obj_raw))
    ct_clean = levels(obj_clean)
    stat = sapply(ct_clean, function(x){
        sub_clean = subset(barcodes_clean, celltypes==x)$barcodes2
        stat_raw = sapply(ct_raw, function(y) {
            sub_raw = subset(barcodes_raw, celltypes==y)$barcodes2
            ratio = length(intersect(sub_clean, sub_raw))/length(sub_clean)
            return(ratio)
        })
        return(stat_raw)
    })
    df = data.frame(stat)
    rownames(df) = paste0('raw_', rownames(df))
    colnames(df) = paste0('clean_', colnames(df))
    return(df)
}

outroot = 'figure/intersection_barcode_v2'

rds_raw = "filtered_h5ad/Uncorrected_SRR21882339.rds"
rds_clean = "filtered_h5ad/CellClear_SRR21882339.rds"

obj_raw = readRDS(rds_raw)
obj_clean = readRDS(rds_clean)

Idents(obj_raw) = obj_raw@meta.data$celltype
Idents(obj_clean) = obj_clean@meta.data$celltype

Idents(obj_raw) = factor(Idents(obj_raw), levels=c('RSC-PyNs', 'SubC-PyNs', 'CA1-PyNs', 'CA3-PyNs', 'DG-GCs', 'NSCs', 'Interneurons', 'IPs', 'CR','Endo', 'Erythrocytes', 'Microglial'))
Idents(obj_clean) = factor(Idents(obj_clean), levels=c('RSC-PyNs', 'SubC-PyNs', 'CA1-PyNs', 'CA3-PyNs', 'DG-GCs', 'NSCs', 'Interneurons', 'IPs', 'CR','Endo', 'Erythrocytes', 'Microglial'))

df = cal_shared(obj_raw, obj_clean)
df_labels <- df
df_labels = round(df_labels, 2)
df_labels[df_labels==0] = ''

p = pheatmap::pheatmap(df,cluster_rows = F,cluster_cols = F,
                    breaks = seq(0, 1, length.out = 50),
                   color = colorRampPalette(c("white","black"))(50),
                   display_numbers = df_labels,
                   border_color = "NA")

pdf(file.path(outdir, paste0(uc, '.pdf')))
p
dev.off()
png(file.path(outdir, paste0(uc, '.png')))
p
dev.off()
