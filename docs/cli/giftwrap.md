---
name: The `giftwrap` Command
description: The `giftwrap` command is the main entry point for processing GIFT-seq data, running the full pipeline end-to-end.
---

# `giftwrap`
The `giftwrap` command is the main entry point for processing GIFT-seq data, running the full pipeline end-to-end. It automates the entire workflow from raw FastQ files to a processed counts matrix, including basic quality control. This is the typical command you would use to process GIFT-seq data.

Running `giftwrap` is equivalent to running the following commands in sequence:

1. [`giftwrap-count`](giftwrap_count.md): Counts GIFT-seq data from FastQ files.
2. [`giftwrap-correct-umis`](giftwrap_correct_umis.md): Corrects UMIs in GIFT-seq data.
3. [`giftwrap-correct-gapfill`](giftwrap_correct_gapfill.md): Corrects gapfill sequences in GIFT-seq data.
4. [`giftwrap-collect`](giftwrap_collect.md): Collects GIFT-seq data after processing.
5. [`giftwrap-summarize`](giftwrap_summarize.md): Performs basic quality control on GIFT-seq data after collection.

See the [GIFTwrap tutorial](../tutorials/processing_giftseq_data.md) for a walkthrough of using `giftwrap` to process GIFT-seq data.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
