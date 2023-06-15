# Adversarials vs Detectors

This repository contains the source code used for the publication "Adversarial Attacks on Leakage Detectors in Water Distribution Networks". The contents are as follows:

- Data: Datasets for which the least sensitive point was determined on the Hanoi and L-Town network as well as data used for threshold validation
- Results: Results for the different datasets
- Writing: tex source files of the publication, including figures
- src: source code in Python
- doc: full documentation of the code
- `water.yml`: file to create the used `conda` environment. This can be installed on other machines using `conda env create -f water.yml`. Note that due to some conda peculiarities, we can give no guarantee that the installation will run smoothly everywhere. If a problem occurs, please install the packages listed there by hand.

## The Code

This section just contains a brief overview. Full documentation can be found in `doc`.
The most important Python file is `find_lsp.py`, containing the `LspFinder` class with implementations of the three search algorithms. You can run `python find_lsp.py` to re-compute the five trials with the genetic algorithm on the larger Hanoi dataset. The block at the end of the file may be changed to run other experiments. Default parameters for the Bisection algorithm can be loaded from `src/Resources/default_bisection_params.yaml`.
`LeakageDetectors.py` contains various leakage detector implementations, of which only the `BetweenSensorInterpolator` is used. This is also the one described in the paper. Feel free to implement your own leakage detector and conduct forther experiments with it.

Note: Before running experiments on L-town, you need to unzip `Data/L-TOWN_Real.zip`
