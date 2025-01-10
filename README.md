# ripple-detection

## environment set up

> [!NOTE]
> Some packages like `_libgcc_mutex`, `_openmp_mutex`, and `ld_impl_linux-64` are Linux-specific and cannot be installed on macOS with M1 chip.

### Option 1. Create an environment directly with conda
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

### Option 2. Use poetry to install packages and build `ptsa` from source

Sometimes, we have problems installing library `ptsa`. We can build it from the source instead. See instructions [here](https://github.com/pennmem/ptsa?tab=readme-ov-file#build-from-source).

First, create your virtual environment and activate it.

Then use poetry to install the remaining requirements:
```
poetry install
```
This will install libraries listed in `pyproject.toml`.

To build `ptsa` from source, See [this](https://github.com/pennmem/ptsa?tab=readme-ov-file#build-from-source)

If you get the error:
```
ld: library not found for -lfftw3
```
Make sure you have library `fftw` installed:

```
conda install -c conda-forge fftw

```
And add the path of the library when building the package:
```
python setup.py build_ext --library-dirs=$CONDA_PREFIX/lib --include-dirs=$CONDA_PREFIX/lib/include
```

This is tested on MacOS with M1 chip.
