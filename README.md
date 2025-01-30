# ripple-detection

## environment set up

> [!NOTE]
> Some packages like `_libgcc_mutex`, `_openmp_mutex`, and `ld_impl_linux-64` are Linux-specific and may not be installed natively on macOS with M1 chip. John reports successful installation on an M2 chip, probably with a x86_64 Conda environment on macOS under Rosetta 2

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

### Set up environments on SGE server:

On Hoffman2, conda is already installed, which can be loaded by
```
module load conda
```

Then, create the environment:
```
conda env create -f environments.yml
conda activate ripple-detection
```

Make sure the conda environment is setup correctly:
```
which python
```

If python does not point to the virtual environment:
```
~/.conda/envs/ripple-detection/bin/python
```
One reason could be a different version of python is loaded when you start your terminal. Check your `~/.bashrc` to see if the default python is loaded.


You may also want to setup a separate jupyter kernel so that you can select on jupyter notebook or vsstudio:

```
python -m ipykernel install --user --name ripple-detection --display-name "ripple-detection"
```

Or you can activate the virtual environment and start jupyter-notebook.

If you are on SEG login node, there could be memory issue. Request an interactive computing node:

```
qlogin -l h_vmem=16G -l mem_free=16G
```



### Option 2. Use poetry to install packages and build `ptsa` from source

Sometimes, we have problems installing library `ptsa`. We can build it from the source instead.

First, create your virtual environment and activate it.

Then use poetry to install the remaining requirements (run command in the root directory of the repo):
```
poetry install
```
This will install libraries listed in `pyproject.toml`.

To build `ptsa` from source, Download [ptsa](https://github.com/pennmem/ptsa) and follow [instructions](https://github.com/pennmem/ptsa?tab=readme-ov-file#build-from-source)

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

The above method only installs `ptsa` locally in the `ptsa` directory, which means you cannot import it from a different path. To install `ptsa` globally in your conda environment:

```
pip install .
```

This is tested on MacOS with an M1 chip.
