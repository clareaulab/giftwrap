---
name: The `giftwrap-summarize` Command
description: The `giftwrap-summarize` command takes processed GIFT-seq data and generates a basic QC summary report and, if given, filters the data based on an already generated `cellranger` output.
---

# `giftwrap-summarize`
The `giftwrap-summarize` command takes processed GIFT-seq data and generates a basic QC summary report and, if given, filters the data based on an already generated `cellranger` output. This command is useful for summarizing the results of GIFT-seq processing and ensuring that the data meets quality standards.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-summarize --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
