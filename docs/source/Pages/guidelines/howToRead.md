# How to read the documentation


The API documentation was designed to satisfy analysts with different experience in python programming.


## Analyst with no experience in python programming


**Non python programmer** would be more interested in the [executables](Pages/Executables.md) scripts.
We invite him to read  the input arguments of each executable.


* Open an anaconda prompt
* activate your virtual environment
* then type the name of the executable.

check out a [Vicon example](https://pycgm2.netlify.app/code/apps/vicon-apps/cgm1) or [QTM example](https://pycgm2.netlify.app/code/apps/qtm-apps) from the project website

To know the executable names, you can  

 * type the command
```
pyCGM2-displayAllScripts.exe
```

in the anaconda prompt.

 * go to the folder `(yourPythonCoreFolder))/envs/(yourVirtualEnvironment)/Scripts`
 * open the file `setup.py` and read the content of `entry_points`



## Novice

By novice, we assume you can :

 * edit a simple script
 * know how to call a function
 * run a python script   


In this case, we encourage novice to check out the [high-level functions](Pages/Lib.md).

 ```{note}
the novice can visit the section [work with the API](https://pycgm2.netlify.app/code/apps/vicon-apps/cgm1) to find out more explicit examples        
 ```

No need to be either a Vicon user or a QTM user, pyCGM2 is not dependant on a mocap system.
it only require c3d files as inputs.    



## Developer

Developer can consult the [modules](Pages/Core.md) section to deep into the content of the pyCGM2 package.
