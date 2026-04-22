PYTHON := .venv/bin/python
VENV := .venv

.PHONY: setup run run-all run-ordered run-random figures

setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run: run-all run-ordered run-random

run-all:
	$(PYTHON) train_reach.py --timesteps 500000
	$(PYTHON) train_reach.py --phase hold --load-model models/ppo_reach_all.zip --timesteps 1000000
	$(PYTHON) train_grasp.py --timesteps 5000000
	$(PYTHON) train_pick_place.py --timesteps 7500000

run-ordered:
	$(PYTHON) train_reach.py --timesteps 500000 --curriculum ordered
	$(PYTHON) train_reach.py --phase hold --load-model models/ppo_reach_ordered.zip --timesteps 1000000 --curriculum ordered
	$(PYTHON) train_grasp.py --timesteps 5000000 --curriculum ordered
	$(PYTHON) train_pick_place.py --timesteps 7500000 --curriculum ordered

run-random:
	$(PYTHON) train_reach.py --timesteps 500000 --curriculum random
	$(PYTHON) train_reach.py --phase hold --load-model models/ppo_reach_rand.zip --timesteps 1000000 --curriculum random
	$(PYTHON) train_grasp.py --timesteps 5000000 --curriculum random
	$(PYTHON) train_pick_place.py --timesteps 7500000 --curriculum random

figures:
	cd analysis && ../$(PYTHON) plot_curricula.py
	cd analysis && ../$(PYTHON) plot_cumulative.py
	cd analysis && ../$(PYTHON) plot_scratch_vs_chain.py
	cd analysis && ../$(PYTHON) plot_scratch_vs_chain_full.py
