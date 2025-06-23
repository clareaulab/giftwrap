---
name: The `giftwrap-correct-umis` Command
description: The `giftwrap-correct-umis` command is used to manually correct UMIs (Unique Molecular Identifiers) in GIFT-seq data. This command is typically run after the initial counting step to ensure that UMIs are accurately processed and corrected. It additionally allows for correction of PCR chimeras/artifacts.
---

# `giftwrap-correct-umis`
The `giftwrap-correct-umis` command is used to manually correct UMIs (Unique Molecular Identifiers) in GIFT-seq data. This command is typically run after the initial counting step to ensure that UMIs are accurately processed and corrected. It additionally allows for correction of PCR chimeras/artifacts.

```py exec="md"
import subprocess, textwrap, os
cmd = "giftwrap-correct-umis --help"
out = subprocess.check_output(cmd.split(), text=True, env={**os.environ, "TERM": "xterm-256color"})
print("<!-- termynal -->")
print("```console")
print(f"$ {cmd}")
print(textwrap.dedent(out))
print("```")
```
