# BayesOptPythia


## Setup and Installation Procedure


1. **Insure that you have docker installed successfully on your system, and check that you are comfortable in downloading and running a docker image.**
> If you need more help on this, feel free to check out the docker tutorials that I wrote here.

----

> A docker image containing all the required code (`pythia8`, `rivet`, `yoda`, `jupyterlab`, `hepmc`, etc.) is available. Pull and run it interactively with the command

```
docker run -it alialkadhim/pythia_sbi_tune
```

2. **Install the docker image which has all software prerequisites.**

To run my docker image you could do 

```
docker run -v $PWD:$PWD -w $PWD -p 8890:8890 -it pythia_sbi_tune
```

Then, to start a jupyter server, inside the docker container do

```
jupyter-lab --ip 0.0.0.0 --port 8890 --allow-root &
```

Then copy the url that is displayed in the terminal and paste it in your local browser.

**The preferred way to run the docker image is to simply do**

```
sudo bash docker_run.sh
```

which basically is just this command

```
docker run -v $PWD:$PWD -w $PWD -p 8889:8889 -it alialkadhim/pythia_sbi_tune:latest
```

3. **set up relative path directory inside the docker container by doing**

```
source setup.sh
```

4. **All the source code is available in `BayesOpt/src`. Take a look for the python scripts there!**

- The code is meant to be as modular and efficient as possible. The same code is used in the toy experiments, pythia experiments, etc.
- If you would like to run toy experiments as described in the paper, first define all the Bayesian Optimization hyperparameters in `configs.py`, then you can do something like `python BayesOpt/src/main_toy.py`.
- If you want to run the large hyperparameter analysis study discussed in the paper on toy experiments, you can do something like `python BayesOpt/src/main_toy_all_hyperparams.py`
- If you would like to run a pythia experiment, you can similarly define the configs and then do something like `python BayesOpt/src/main_toy.py`

5. **post-processing, analysis and paper plots**

- If you would like to produce plots for toy experiments like those shown in the paper, you can run the jupyter notebook `BayesOpt/src/post_processing_toy.ipynb`
- If you would like to run the BO-GPy analysis on pythia experiments, you can do something like `python BayesOpt/src/validate_pythia_EI.py`
- If you would like to produce plots for toy experiments like those shown in the paper, you can run the jupyter notebook `BayesOpt/src/post_processing_pythia.ipynb`
