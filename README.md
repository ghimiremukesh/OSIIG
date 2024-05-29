## Code for the paper - [State-Constrained Zero-Sum Games with One-Sided Information](https://arxiv.org/pdf/2403.02741)

### Training and Simulation
1. Install the conda environment: `conda env create -f env.yml`

#### Training
2. To train the models, run the following:
   1. first, run `train_uncons.sh` to train the unconstrained game. In the command line, enter `./train_uncons.sh`
   2. next, run `train_cons.sh` to train the constrained game. 

#### Simulation
3. To simulate the games using the pre-trained models, do the following:
   1. Run `validations_scripts/simulate_uncons.py` to simulate unconstrained game.
   2. Run `validations_scripts/simulate_cons.py` to simulate constrained game. 


Optionally, to train the Reachable Tube, navigate to `reachability/experiment_scripts` and run `train_hji_8D.py`


Known Issues: If you get a directory exists error, run the `.sh` file again and it should be ok. 

