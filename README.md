# EVAC+
This is a repository to run EVAC+: Multi-scale V-net with Deep Feature CRF Layers for Brain Extraction

## Pre-requisites

This model depends on the implementation of the CRFasRNNLayer, which is from this repository : https://github.com/MiguelMonteiro/CRFasRNNLayer.

Files were copied for convenience.

Please follow the instructions of that repo which is also written below.

===================================================

To compile the code run:
````
sh build.sh
````

See the nested module [permutohedral_lattice](https://github.com/MiguelMonteiro/permutohedral_lattice) for more information on compilation for different image types.

====================================================

Just running the command should be enough to run the model.

Due to the constraints of the implemented CRFasRNNLayer, the model is limited to a Tensorflow version of 2.3.0, which will be edited later to remove this constraint.

## Examples

Please refer to [training.ipynb](training.ipynb) and [testing.ipynb](testing.ipynb).

Specifications of the model can be found in [models.py](models.py)

