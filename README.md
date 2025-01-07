# ripple-detection

## environment set up

> [!NOTE]
> Some packages like `_libgcc_mutex`, `_openmp_mutex`, and `ld_impl_linux-64` are Linux-specific and cannot be installed on macOS.

First setup your conda environment:
```
conda env create -f environments.yml
```

Activate environment:
```
conda activate ripple-detection
```

When you modify `environments.yml`, run the following command to update the environment:
```
conda env update --file environment.yml --prune
```

Sometimes, we have problems installing library `ptsa`. We can build it from the source instead. See instructions [here](https://github.com/pennmem/ptsa?tab=readme-ov-file#build-from-source).

If you install `ptsa` from the source, then use poetry to install the remaining requirements:
```
poetry install
```
This will install libraries listed in `pyproject.toml`.
