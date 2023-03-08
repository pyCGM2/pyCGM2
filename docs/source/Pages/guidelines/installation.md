# Installation

pyCGM2 requires, first, to install the [anaconda](www.anaconda.org) python suite. Download either **anaconda** or its light-version named **miniconda**. 


## option 1 - local installation from the github folder

**step1 - download pyCGM2**

 * clone/fork the github package

or 

 * download a release version

**step 2 - install the pyCGM2 python environement**

  1. open an anaconda console
  2. go to your local pyCGM2 folder with the `cd` command.
  3. depending you use python 3.7, 3.8 or 3.9, type the command ( e.g for python3.9)

```
conda env create -f environment_39.yml
```

this create a python environment named `pycgm39` with all pyCGM2 dependancies.

then, 

 4. activate your environement

```
activate pycgm39
```

**step 3 - install the pyCGM2 package**

After activation, of your environment, install the package with :  

```
pip install .
```

for developer who would like to modify pyCGM2 code, you can
install the package with  

```
pip install -e .
```

