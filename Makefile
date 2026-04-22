PYTHON := .venv/bin/python
VENV := .venv

.PHONY: setup run run-all run-ordered run-random run-scratch-full run-ordered-match-all figures

setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip uninstall -y opencv-python
	$(PYTHON) -m pip install --force-reinstall --no-deps opencv-python-headless

run: run-all run-ordered run-random run-scratch-full

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

# From-scratch baseline for plot_scratch_vs_chain_full.py. Trains pick-and-place
# only, for the same total timesteps as the full chain (500K + 1M + 5M + 7.5M).
# Creates a new PPO_<n> dir under logs/pick_place_all/; update the `ppo_n` value
# for the scratch run in analysis/plot_scratch_vs_chain_full.py to match.
run-scratch-full:
	$(PYTHON) train_pick_place.py --from-scratch --timesteps 14000000

# Resume ordered pick_place until the cumulative wall-clock hours across all
# phases of the ordered curriculum match the 'all' curriculum's total (~25 h
# per plot_cumulative.py). Override the budget with `make run-ordered-match-all HOURS=30`.
HOURS ?= 25.61
run-ordered-match-all:
	$(PYTHON) train_pick_place.py --resume --curriculum ordered --max-hours $(HOURS) --timesteps 999999999 

figures:
	cd analysis && ../$(PYTHON) plot_curricula.py
	cd analysis && ../$(PYTHON) plot_cumulative.py
	cd analysis && ../$(PYTHON) plot_scratch_vs_chain_full.py
	cd analysis && ../$(PYTHON) plot_ordered_matched_vs_all.py
