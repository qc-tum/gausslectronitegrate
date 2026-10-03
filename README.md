[![CI](https://github.com/qc-tum/gausslectronitegrate/actions/workflows/ci.yml/badge.svg)](https://github.com/qc-tum/gausslectronitegrate/actions/workflows/ci.yml)


Gausslectronitegrate
====================

Computing kinetic, nuclear, and electron repulsion integrals for Gausslet orbitals, accompanying [arXiv:2609.39176](https://arxiv.org/abs/2609.39176).

This library is written in C and offers a Python 3 interface for more straightforward accessibility.


Examples
--------
The [examples](examples/) folder contains several demonstrations of the functionalities, using Jupyter notebooks and the Python interface:
- [kinetic_integrals](examples/kinetic_integrals.ipynb)
- [nuclear_integrals](examples/nuclear_integrals.ipynb)
- [eri_integrals](examples/eri_integrals.ipynb)
- [erida_integrals](examples/erida_integrals.ipynb)
- [hydrogen_atom](examples/hydrogen_atom.ipynb)
- [hydrogen_molecule_eri](examples/hydrogen_molecule_eri.ipynb)
- [hydrogen_molecule_erida](examples/hydrogen_molecule_erida.ipynb)


Building
--------
The code requires a C compiler, CMake, and the HDF5 and Python 3 development libraries with NumPy. These can be installed via 
- `sudo apt install build-essential libhdf5-dev python3-dev python3-numpy` (on Ubuntu Linux or similar)
- `brew install hdf5 python3 numpy` (on Apple macOS)

From the project directory, run the following commands in a terminal to build the project:
```bash
mkdir build_gli && cd build_gli
cmake ../
cmake --build .
```
Currently, this will compile the unit tests, which you can run via `./gausslectronitegrate_test`, a performance benchmark demo, and the Python module library.

To build the corresponding Python package directly, ensure that the Python [build](https://pypi.org/project/build/) tool is installed, and run
```bash
python3 -m build . --wheel
pip3 install dist/gausslectronitegrate-...whl
```
The first line should run `cmake` in the background and create a Python "wheel" (.whl file) in the `dist/` subfolder. This package file can then be installed locally via the second line.


Directory structure
-------------------
- **cli**: command-line interface
- **examples**: examples and demonstrations
- **gausslets**: Gausslet coefficients and function evaluation
- **include**: include files of the C code
- **perf**: performance benchmarking
- **python**: Python interface
- **src**: C source code
- **test**: unit tests


Citing
------
Gausslectronitegrate accompanies the following preprint - if it's ever useful for a research project, please consider citing it:

```
@Article{gausslectronitegrate,
  author  = {Yin, Xianrui and Ghasempour, Fereshteh and Mendl, Christian B.},
  title   = {Computing electron overlap integrals for Gausslet orbitals on cubic lattices},
  journal = {arXiv:2609.39176},
  eprint  = {2609.39176},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.39176},
}
```


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
