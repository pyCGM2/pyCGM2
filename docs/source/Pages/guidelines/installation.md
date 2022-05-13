# Installation


The installation suppose you install the [anaconda] python suite


## Create a virtual environment [ Recommanded]

```{note}
you can also consult the  [python setup](https://pycgm2.netlify.app/code/python-setup/) page of the project website
```


First create a virtual python environment

open the anaconda prompt, then type

```
conda create --name pycgm3 python=3.7
```

and activate your environment

```
activate pycgm3
```


## install pyCGM2

### option 1 : from pip


### option 2 : from a local pyCGM2 folder

```{note}
you can also consult the  [installation](https://pycgm2.netlify.app/code/installation/) page of the project website
```

* download pyCGM2 code from the [github website](https://github.com/pyCGM2/pyCGM2)
* open an anaconda prompt
* go to your local pycgm2 folder
* type

```
pip install .
```



### option 3 : developer mode

consider this option, if your goal is to customize the pyCGM2 package

* first fork the package from github
* download your fork
* open an anaconda prompt
* go to your local pycgm2 folder
* type

```
pip install -e .
```
