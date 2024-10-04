## Read giftwrap-generated h5 files into R objects.

library(rhdf5)
library(Matrix)

read_sparse <- function(file, grp) {
  dat = h5read(file, grp)
  attrs = h5readAttributes(file, grp)

  counts <- sparseMatrix(
    dims = attrs$shape,
    j = as.numeric(dat$indices),
    p = as.numeric(dat$indptr),
    x = as.numeric(dat$data),
    index1=FALSE
  )

  return(counts)
}

read_df <- function(file, grp) {
  return(as.data.frame(h5read(file, grp)))
}

# R version of the read_h5_file function in the Python code
read <- function(file) {
  ## this is in cell x gene format
  counts = read_sparse(file, 'matrix/data')
  total_reads = read_sparse(file, 'matrix/total_reads')
  percent_supporting = read_sparse(file, 'matrix/percent_supporting')

  features_df = as.data.frame(t(h5read(file, 'matrix/probe')))
  colnames(features_df) <- c('probe', 'gapfill')

  barcodes_df = as.data.frame(h5read(file, 'matrix/barcode'))
  colnames(barcodes_df) <- c("barcode")


  probe_metadata = read_df(file, 'probe_metadata')
  probe_metadata$probe = probe_metadata$name
  cell_metadata = list()
  for (colname in h5read(file, 'cell_metadata/columns')) {
    dat = h5read(file, paste("cell_metadata", colname, sep="/"))
    cell_metadata[[colname]] = dat
  }
  cell_metadata = as.data.frame(cell_metadata)

  features_df = merge(features_df, probe_metadata, by=c('probe'))
  features_df$probe_gapfill = paste(features_df$probe, features_df$gapfil, sep= ".")

  rownames(features_df) <- features_df$probe_gapfill


  barcodes_df = merge(barcodes_df, cell_metadata, by=c('barcode'))

  rownames(barcodes_df) <- barcodes_df$barcode

  metadata = h5readAttributes(file, "/")

  ## Adding names to bad boys
  rownames(counts)   <- barcodes_df$barcode
  colnames(counts) <- features_df$probe_gapfill

  rownames(total_reads)   <- barcodes_df$barcode
  colnames(total_reads) <- features_df$probe_gapfill

  rownames(percent_supporting)   <- barcodes_df$barcode
  colnames(percent_supporting) <- features_df$probe_gapfill


  return(list(
    counts=counts,
    total_reads=total_reads,
    percent_supporting=percent_supporting,
    features_df=features_df,
    barcodes_df=barcodes_df,
    metadata=metadata
  ))
}

read_seurat <- function(file) {
  library(Seurat)

  data = read(file)

  obj = CreateSeuratObject(counts=t(data[['counts']]),assay = "gapfill")
  obj[["total_reads"]] <- CreateAssayObject(counts = t(data[['total_reads']]))
  obj[['percent_supporting']] <- CreateAssayObject(counts = t(data[['percent_supporting']]))

  ##add 'gene' metadata
  #https://github.com/satijalab/seurat-object/issues/125
  obj[["gapfill"]] <- AddMetaData(object = obj[["gapfill"]], metadata = data[["features_df"]])
  #obj[["gapfill"]]@meta.data ## To access
  #obj[["total_umis"]] <- AddMetaData(object = obj[["total_umis"]], metadata = data[["features_df"]])
  #obj[["percent_supporting"]] <- AddMetaData(object = obj[["percent_supporting"]], metadata = data[["features_df"]])

  ## add cell metadata
  obj <- AddMetaData(obj, metadata = data[["barcodes_df"]])

  ## put unstructured data in misc
  obj@misc <- data[["metadata"]]

  return(obj)
}

#read("../data/counts.1.h5")
#read_seurat(file ="../data/counts.1.h5")
