## A theory of temporal self-supervised learning in neocortical layers

This repository contains the code to run the experiments for the paper: A theory of temporal self-supervised learning in neocortical layers.

## Dependencies

To install the requrired packages, create a new conda environment using:

```
conda env create -f environment.yaml
```

## Training the neo-ssl model

### Command line

To train run the experiment, simply pass the figure number as an argument for the `run.py`. For example, to run the experiment for figure 2, execute the following command:

```
python run.py --exp fig2
```

### Plot the results
For each experiment, there is a jupyter notebook in `plotting` folder that plots the results.
