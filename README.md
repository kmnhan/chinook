# chinook

## Introduction

This is a modified version of the chinook package, which is a Python module for
simulating matrix-elements in ARPES. The original version of chinook can be found
[here](https://github.com/rpday/chinook).

A few of the changes made to the original version of chinook are:

- Proper multiprocessing support for matrix element calculations
- Numba-compiled functions for faster calculations
- Progress and ETA indicators
- Tweaks to plotting functions
- Modern Python packaging and project management

## Installation

The modified version of chinook must be installed from source:

```bash

pip install git+https://github.com/kmnhan/chinook.git

```

## Original README

Python module for simulating matrix-elements in ARPES. Documentation, in addition to
instructions and tutorials can be found at <https://chinookpy.readthedocs.io>. chinook
is registered at <https://www.pypi.org>, and can be installed simply using pip. For
further information regarding the formalism and applications of chinook, please see our
recent publication,

R.P. Day, B. Zwartsenberg, I.S. Elfimov and A. Damascelli, npj Quantum Materials 4, 54 (2019)

For any questions, please send us mail at <rpday7@gmail.com>

Thanks!

-- The chinook team
