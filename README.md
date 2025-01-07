# ripple-detection

## environment set up

First setup your conda environment:
```
conda create --name ripple-detection
```

Activate environment:
```
conda activate ripple-detection
```

Then run the following to install additional libraries:
```
conda env update --file environment.yml --prune
```

Sometimes, we have problems installing library `ptsa`. We can build it from the source instead. See instructions [here](https://github.com/pennmem/ptsa?tab=readme-ov-file#build-from-source).

If you install `ptsa` from the source, then use poetry to install the remaining requirements:
```
poetry install
```
This will install libraries listed in `pyproject.toml`.
