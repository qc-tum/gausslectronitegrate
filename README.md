[![CI](https://github.com/cmendl/gausslectronitegrate/actions/workflows/ci.yml/badge.svg)](https://github.com/cmendl/gausslectronitegrate/actions/workflows/ci.yml)


Gausslectronitegrate
====================

Computing kinetic, nuclear and electron repulsion integrals for Gausslet orbitals.


Building
--------
The code requires a C compiler, cmake and the HDF5 development library. These can be installed via 
- `sudo apt install build-essential libhdf5-dev` (on Ubuntu Linux)
- `brew install hdf5` (on Apple macOS)

From the project directory, run the following commands in a terminal to build the project:
```bash
mkdir build_gli && cd build_gli
cmake ../
cmake --build .
```
Currently, this will compile the unit tests, which you can run via `./gausslectronitegrate_test`, as well as a performance demo example.


References
----------
- Steven R. White  
  Hybrid grid/basis set discretizations of the Schrödinger equation  
  [J. Chem. Phys. 147, 244102 (2017)](https://doi.org/10.1063/1.5007066)
- Steven R. White, E. Miles Stoudenmire  
  Multisliced gausslet basis sets for electronic structure  
  [Phys. Rev. B 99, 081110(R) (2019)](https://doi.org/10.1103/PhysRevB.99.081110)
- Steven R. White, Michael J. Lindsey  
  Nested gausslet basis sets  
  [J. Chem. Phys. 159, 234112 (2023)](https://doi.org/10.1063/5.0180092)
- Samuel F. Boys  
  Electronic wave functions I. A general method of calculation for the stationary states of any molecular system  
  [Proc. R. Soc. Lond. A 200, 542 (1950)](https://doi.org/10.1098/rspa.1950.0036)
