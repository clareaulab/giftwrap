---
title: Analyzing GIFT-seq Data with GIFTwrap
summary: A guide to the GIFTwrap package for analysis.
---

# Analyzing GIFT-seq Data with GIFTwrap
In addition to providing a command line interface for working with raw GIFT-seq data, GIFTwrap also provides a Python API designed to integrate well with the scverse ecosystem to enable robust analysis of GIFT-seq data. This tutorial will walk through some basic analyses in Python, however we also offer basic R integration as described [here](./seurat_integration.md). We will also not cover spatial analysis in this tutorial, for information regarding spatial-specific analysis, please refer to the [spatial tutorial](./spatial_giftseq.md).


## Getting Started
To get started, ensure that GIFTwrap is installed in your Python environment, note that analysis requires scanpy and related components to also be installed in your environment, so we recommend [installing GIFTwrap with the `analysis` extra](../installation.md).

We will also be working with the counts.1.filtered.h5 file from the GIFTwrap pipeline, though counts.1.h5 is also acceptable since the file format is the same.
