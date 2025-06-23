---
name: The `giftwrap-collect` Command
description: The `giftwrap-collect` command is used to manually collect GIFT-seq data after processing, allowing for further analysis and quality control.
---

# `giftwrap-collect`
The `giftwrap-collect` command is used to manually collect GIFT-seq data after processing, allowing for further analysis and quality control. This command is typically run after all steps of the GIFT-seq pipeline have been completed, including counting, correcting UMIs, and gapfill sequences.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-collect --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
