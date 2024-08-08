# Fuzzy reputation for 5G
This repository contains the code used as a demonstrator for a fuzzy reputation system in 5G network services sharing. 

## Install the project
Poetry is used for package dependency.
Use `poetry install` to install the necessary python packages. 

## Results and analysis 
All the graphs that make it into the paper are located in the notebook located in `./plotter/paper_graphs.ipynb`. 
The cell can be ran individually 

## Re-run the tests


## Hydra usage
Run the project as a python module :
`python -m reput`
`python -m reput --config-name fuzzy_config`

## Repository structure
```
.
├── conf # Hydra configuration files used for the experiments. 
│   ├── peers
│   ├── reputation
│   ├── simulation
│   └── topology
├── plotter # Graph used for the papers and code necessary to diggest the json files into those graphs. 
│   ├── fuzzy # Output containing the different pdf graphs.
│   └── test
│   ├── paper_graphs.ipynb # All the graphs that make it into the paper are in this notebook. 
├── reput # Code used for runing the experiment 
│   ├── tests
│   └── utils
└── results # Row json results that the graph are used from with hydra config files containing parameters used for the run.    
    ├── decay_results
    ├── fuzzy
    ├── no_reput
    ├── reput
    └── reput_no_random
```