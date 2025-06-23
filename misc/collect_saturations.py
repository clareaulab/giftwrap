import os
import sys
from glob import glob

import matplotlib.pyplot as plt

import giftwrap as gw


def main(dirs):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(32, 16))
    fig.set_dpi(600)
    exp2sats = dict()
    # Set colors for each experiment
    for dir in dirs:
        print(f"Searching in {dir}")
        # Find the appropriate file. First look for filtered.h5 files
        files = glob(f'{dir}/counts.*.filtered.h5')
        if len(files) == 0:
            # If not found, look for .h5 files
            files = glob(f'{dir}/counts.*.h5')
        for file in files:
            plex = file.split('.')[-2]
            exp = os.path.basename(dir).replace("/", "") + "_" + plex
            adata = gw.read_h5_file(file)
            # Collapse
            adata = gw.tl.collapse_gapfills(adata)
            # Compute current saturation
            sat = gw.sequencing_saturation(adata.layers['total_reads'])
            exp2sats[exp] = sat
            # Compute the curve
            curve = gw.sequence_saturation_curve(adata.layers['total_reads'])
            # Plot the curve
            axes[0].plot(curve[:,0], curve[:,1], label=exp)

    # Complete the saturation plot
    # Change the line colors
    for i, line in enumerate(axes[0].lines):
        line.set_color(plt.cm.tab20(i))
    axes[0].set_xlabel('Mean reads per cell')
    axes[0].set_ylabel('Saturation')
    axes[0].legend()
    axes[0].set_title('Downsampled Saturation curves')
    axes[0].set_xscale('log')

    # Plot the final saturation values as a bar plot (sorted by value)
    sorted_exps = sorted(exp2sats.keys(), key=lambda x: exp2sats[x])
    axes[1].bar(sorted_exps, [exp2sats[exp] for exp in sorted_exps])
    axes[1].set_title('Final saturation values')
    axes[1].set_ylabel('Saturation')
    axes[1].set_xticklabels(sorted_exps, rotation=90)
    # Save as a pdf
    fig.savefig('saturation.pdf')
    plt.close(fig)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        print('Usage: python collect_saturations.py path/to/giftwrap_output_dir1/ path/to/giftwrap_output_dir2/...')
        sys.exit(1)
    print("Note that this may take awhile depending on the number of experiments.")
    main(args)
