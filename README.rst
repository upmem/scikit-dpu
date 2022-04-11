.. -*- mode: rst -*-

PIM-enabled scikit-learn extension

This is a library that provides hardware acceleration for scikit-learn algorithms with UPMEM PIM devices.
Currently supported algorithms are:

- Classification trees (extremely randomized version a.k.a. scikit-learn's extra trees)

Algorithms being integrated are:

- Linear regression
- Logistic regression
- K-Means

.. list-table::
  :header-rows: 1

  * - CI
    - status
  * - pip builds
    - |actions-pip-badge|

.. |actions-pip-badge| image:: https://github.com/upmem/scikit-dpu/workflows/Pip/badge.svg
   :target: https://github.com/upmem/scikit-dpu/actions?query=workflow%3APip