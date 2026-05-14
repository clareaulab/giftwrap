FROM continuumio/miniconda3:latest

# Copy the environment file
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml && conda clean -afy

# Make the environment's binaries available on PATH
ENV PATH=/opt/conda/envs/giftwrap_sc_env/bin:$PATH

# Default command
CMD ["/bin/bash"]