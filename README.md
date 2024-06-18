# stokes-nc-ill-posed
Reproduction material for the paper 
> Data assimilation nonconforming finite element methods for transient Stokes problem 
>
> * authors: Erik Burman, Deepika Garg and Janosch Preuss
> * University College London

The numerical experiments have been implemented using the Software [ngsolve](https://ngsolve.org/). 
More precisely we used the software at commit `a8b62a566f421f3e7942ed95e7cdc586e326b33c` related to [this](https://github.com/NGSolve/ngsolve) github repository.

The reproduction files are available in terms of `jupyter` notebooks in the folder `ngsolve`. 

## Section 5.1. Convergence Study: Stokes problem

### Fig.2 
The notebook to reproduce these results is `Stokes-non-conforming-clean-data.ipynb`. 
In the second cell of the notebook the discretization parameters have to be set. 
To reproduce the results in the figure the notebook should be run consecutively with the following parameters: 

```
N = 1
hmax = 0.6 

N = 2
hmax = 0.3

N = 4
hmax = 0.15


N = 8
hmax = 0.075

N = 16
hmax = 0.0375
``` 
The corresponding errors are printed when running the notebook. The quantities of interest (shown in the Fig.2) are 

```
L2 error velocity at time step 0 = ...
L2 error velocity at time step N = ... 
L2 error pressure at time step N = ...
delta norm = ...
```

### Fig.3
The notebook to reproduce these results is `Stokes-non-conforming_const_noise.ipynb`. 

### Fig.3 (a) 
To reproduce these results we have to fix the parameters 
```
N = 16 
hamx = 0.0375
```
in the second cell of the notebook. The parameter which has to be varied from 1e-5 to 1e1 is called `constant_noise_data` (it is also in the second cell of the notebook). We run the notebook consecutively changing this parameter and monitor the results. The output shown in the figure is 

```
l2_err at t = [0,T/8,T/4,T/2,T] = ...
delta norm = ...
```
### Fig.3 (b)
To reproduce these results we have to fix `constant_noise_data = 2e-1` and run the script repreatedly with the following discretization parameters:

```
N = 1
hmax = 0.6 

N = 2
hmax = 0.3

N = 4
hmax = 0.15


N = 8
hmax = 0.075

N = 16
hmax = 0.0375
``` 

## Section 5.2. Navier-Stokes equations

### Fig. 4
The notebook to reproduce these results is `Navier-Stokes.ipynb`. The parameters to be changed are in the second cell of the notebook. Run the notebook repeatedly with the follwing discretization parameters:

```
N = 2
hmax = 0.4

N = 4
hmax = 0.2

N = 8 
hmax = 0.1

N = 12 
hmax = 0.075

N = 16
hmax = 0.05

```
Note that a fixed point iteration is run and in each iteration step the errors 

```
L2-error velocity final time: ...
L2-error pressure final time: ...
L2-error velocity initial time: ...
epsilon norm = ...
``` 

The values shown in the Fig.4 are taken from the final step of the fixed point iteration. 


 






