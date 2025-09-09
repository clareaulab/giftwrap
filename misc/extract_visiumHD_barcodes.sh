#!/usr/bin/env sh

# Extract CB and CR tag from Visium HD BAM files
# Usage: ./extract_visiumHD_barcodes.sh <input.bam> <output.csv>

echo "Barcode,ID,R1" > $2
samtools view $1 | awk '{
  cb=""; cr=""; r1="";
  for (i=1; i<=NF; i++) {
    if ($i ~ /^CB:Z:/) cb=substr($i,6);
    if ($i ~ /^CR:Z:/) cr=substr($i,6);
    if ($i ~ /^1R:Z:/) r1=substr($i,6);
  }
  if (cb != "" && cr != "" && r1 != "") print cb "," cr "," r1
}' | sort -u >> $2
