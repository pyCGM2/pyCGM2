
| Type | Status |
|---|---|
| License | [![License: CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/4.0/)  |
| Continuous integration | [![Build status-python3.9](https://github.com/pyCGM2/pyCGM2/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/pyCGM2/pyCGM2/actions/) |



## pyCGM2 documentation

See pyCGM2's [documentation site](https://pycgm2.readthedocs.io/en/latest/).


## Installation

 * First, *clone or fork* the pycgm2 github folder to have a local version on your computer 
 * install a *virtual python environment* . The following code automatically creates a virtual python environment named 'pycgm39' based on python 3.9 (python version 3.7 and 3.8 are also available.) 

```bash
conda env create -f environment_py39.yml
```

 * install pycgm2 with either the command 

```bash
pip install . 
```

to place pycgm2 in the *site-package* folder

or 
```bash
pip install -e . 
```

to work, as developper, and use  your local pycgm2 folder ( ie the clone/fork folder)


* test your installation with 

```python
import pyCGM2
```

