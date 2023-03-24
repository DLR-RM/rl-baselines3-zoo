## RL Zoo3 Documentation

This folder contains documentation for the RL Zoo.


### Build the Documentation

#### Install Sphinx and Theme
Execute this command in the project root:
```
pip install stable_baselines3[docs]
pip install -e .
```

#### Building the Docs

In the `docs/` folder:
```
make html
```

if you want to building each time a file is changed:

```
sphinx-autobuild . _build/html
```
