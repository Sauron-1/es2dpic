This is a simple 2D3V electrostatic PIC simulation code runs on single CUDA GPU.

# Installation
The simulation parameters are written in the code, so no initialization is required,
you just need to change parameters in the code, compile and run.

However, the program does require CUDA installation and a CUDA enabled GPU with
[compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus)
equal to or greater than 6.0 for support of 64-bit floating atomic addition.
Also, cublas, curand and cufft, which come along with CUDA toolkit, are required.

NetCDF is optional package for data saving, otherwise csv files would be generated.

For installation of CUDA toolkit, please refer to [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
or [CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

For installation of NetCDF, please refer to its [documantation](https://www.unidata.ucar.edu/software/netcdf/docs/index.html).

# Usage

## Parameters

- `SAVE_PATH`: An existing path to save output data. Note that energy output data will be always
saved to current directory.
- `USE_NETCDF`: If `1`, use NetCDF to save data, else use csv.
- `N_STEPS`: Number of steps to run.
- `N_SAVE`: Save field data every # steps.
- `DT`: Step time, unit is electron plasma frequency.
- `NX`, `NY`: Number of grids in both direction.
- `NPARTX`, `NPARTY`: Number of particles per x/y grid.
- `N_SPECIES`: Number of particles species.
- `AX`, `AY`: Factor used for filter in field solver.
- `ms`: Mass in electron mass.
- `qs`: Charge in elctron charge.
- `vths`: Thermal velocity for each species, each direction.
- `vdrs`: Drift velocity for each species, each direction.
- `B`: Background magnetic field, in 3 directions.

## Compiling

Please use `nvcc` to compile and add following flags: `-lnetcdf -lcufft -lcublas -lcurand -arch sm_XX`
if you want to use NetCDF or just add `-lcufft -lcublas -lcurand -arch sm_XX` if you don't want to use NetCDF.
Where `sm_XX` means your GPU's compute capability, e.g. `sm_75` for compute capability 7.5.

For example, use `nvcc code.cu -o code -O3 -lnetcdf -lcufft -lcublas -lcurand -arch sm_75` to generate executable `code` on 16XX or 20XX Geforce GPU.

## Outputs

In `SAVE_PATH`, `dataXXXXX.nc` or `dataXXXXX.csv` would be found, with charge density and
electric field in every grid point been saved. Look for headers for more information.
In running directory, `energy.dat` would be found,
with four columns means # of step, total kinetic energy, field energy and total energy.
