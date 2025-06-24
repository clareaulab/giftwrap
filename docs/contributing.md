---
title: Contribution Guide
summary: Contributing to the project
---
# Contribution Guide
This guide provides information on building and contributing to the `giftwrap` project.

## Required Tooling
This project is managed using the [uv](https://docs.astral.sh/uv/) tooling system, a modern dependency manager for python projects. Please ensure that it is available in your environment.

## Installing Dependencies
To install the project dependencies, run the following command:

```bash
uv sync --all-extras --all-groups
```

Note that when adding/updating packages with `uv add` or updating the `pyproject.toml` file, you should run `uv sync` to ensure that all dependencies are correctly installed and the `uv.lock` file is updated.

## Building the package
To build the package, use the following command:

```bash
uv build
```

You should see a build `.whl` file in the `dist/` directory. This can be directly installed with `pip`. 


## Updating Documentation
The `giftwrap` documentation site is built using [mkdocs](https://www.mkdocs.org/) and [mkdocs-material](https://squidfunk.github.io/mkdocs-material/). To update the documentation, follow these steps:

1. Make your changes to the Markdown files in the `docs/` directory.

2. If adding a new page, ensure it is linked in the `nav` section of the `mkdocs.yml` file.

3. To preview your changes locally, run:
   ```bash
   uv run mkdocs serve
   ```
   This will start a local server where you can view the documentation at `http://localhost:8000`.

4. When committed to Github, the documentation will automatically be build and deployed.


## Releasing a new version (Maintainers only)
To release a new version of the package, follow these steps:

1. Update the version in `pyproject.toml` to the new version.

2. Run the build command to create the new package:
   ```bash
   uv build
   ```

3. Upload the new package to PyPI via uv or through Github Actions (Recommended):
    ```bash
    # Manually via command line
    uv publish --token <your-pypi-token>
    ```
    or simply create a new release on GitHub with a tag in the form of `vX.Y.Z`, and the GitHub Actions workflow will automatically publish the package to PyPI.
