# CellClear: Enhancing scRNA-seq Data Quality via Biologically-Informed Ambient RNA Correction

<img src="https://github.com/WhiteRabBio/CellClear/blob/main/method.png" width="1000">



## Installation

**CellClear** can be installed from the python repository pypi (https://pypi.org/project/CellClear/)



## **Running CellClear**

A typical **CellClear** run with default settings would look like this:

```python
CellClear correct_expression --filtered_mtx_path filtered_feature_bc_matrix --raw_mtx_path raw_feature_bc_matrix --prefix test --output .
```

For full usage details with additional options, see "CellClear correct_expression --help".



## CellClear outputs 

(1) Determine the ambient RNA  expression level

(2) Give the identified ambient genes

(3) Remove the ambient RNA from each cell and output a new matrix

