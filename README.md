# Cell Instance Segmentation using Z-Stacks in Digital Cytopathology

This repository contains code for the paper:

```
A. Bouyssoux, R. Fezzani and J. -C. Olivo-Marin, "Cell Instance Segmentation Using Z-Stacks In Digital Cytopathology," 2022 IEEE International Symposium on Biomedical Imaging (ISBI), 2022.
```

## Dataset

The Cell Instance Segmentation Dataset can be downloaded on Zenodo: https://zenodo.org/record/5938893
In order to use directly the notebook provided with this package, the dataset
archive should be uncompressed under the `zstacks_cell_instance_segmentation`
folder and next to the `src` folder.

## Instructions

A Jupyter Notebook located at `./src/main.ipynb` contains functions to train the
models and evaluate the performances on the CISD test set.
