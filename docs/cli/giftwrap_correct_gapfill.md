---
name: The `giftwrap-correct-gapfill` Command
description: The `giftwrap-correct-gapfill` command is used to manually correct gapfill sequences in GIFT-seq data. This command is typically run after the initial counting and UMI correction steps to ensure that gapfill sequences are accurately processed.
---

# `giftwrap-correct-gapfill`
The `giftwrap-correct-gapfill` command is used to manually correct gapfill sequences in GIFT-seq data. This command is typically run after the initial counting and UMI correction steps to ensure that gapfill sequences are accurately processed.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-correct-gapfill --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
