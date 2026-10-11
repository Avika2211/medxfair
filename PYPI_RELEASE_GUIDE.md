# Publication checklist

Review code and metadata, confirm a license choice, run `python -m pytest -q`, `python -m build`, and `python -m twine check dist/*`; verify installation from the built wheel. Set up PyPI Trusted Publishing with the exact GitHub owner/repository and `.github/workflows/publish.yml` workflow; environment `pypi`. Only publish after explicit release review. The workflow is manual; GitHub commits must not automatically publish.
