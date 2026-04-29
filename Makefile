PYTHON := .venv/bin/python
VENV := .venv

.PHONY: setup run replicate run-all run-ordered run-random run-scratch-full run-ordered-match-all figures

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
	$(PYTHON) train_pick_place.py --timesteps 10000000

run-ordered:
	$(PYTHON) train_reach.py --timesteps 500000 --curriculum ordered
	$(PYTHON) train_reach.py --phase hold --load-model models/ppo_reach_ordered.zip --timesteps 1000000 --curriculum ordered
	$(PYTHON) train_grasp.py --timesteps 5000000 --curriculum ordered
	$(PYTHON) train_pick_place.py --timesteps 10000000 --curriculum ordered

run-random:
	$(PYTHON) train_reach.py --timesteps 500000 --curriculum random
	$(PYTHON) train_reach.py --phase hold --load-model models/ppo_reach_rand.zip --timesteps 1000000 --curriculum random
	$(PYTHON) train_grasp.py --timesteps 5000000 --curriculum random
	$(PYTHON) train_pick_place.py --timesteps 10000000 --curriculum random

# From-scratch baseline for plot_scratch_vs_chain_full.py. Trains pick-and-place
# only, for the same total timesteps as the full chain (500K + 1M + 5M + 10M).
# Creates a new PPO_<n> dir under logs/pick_place_all/; update the `ppo_n` value
# for the scratch run in analysis/plot_scratch_vs_chain_full.py to match.
run-scratch-full:
	$(PYTHON) train_pick_place.py --from-scratch --timesteps 16500000

# Resume ordered pick_place until the cumulative wall-clock hours across all
# phases of the ordered curriculum match the 'all' curriculum's total.
# If HOURS is unset, derive it from 'all''s actual logs at recipe time, so the
# budget is always grounded in the real training time on this machine rather
# than a hardcoded literal. Override with `make run-ordered-match-all HOURS=30`.
HOURS ?=
run-ordered-match-all:
	@HOURS_VAL="$(HOURS)"; \
	if [ -z "$$HOURS_VAL" ]; then \
	    echo "Computing target hours from 'all' curriculum logs..."; \
	    HOURS_VAL=$$($(PYTHON) analysis/compute_match_hours.py all); \
	fi; \
	echo "Resuming ordered pick_place with --max-hours $$HOURS_VAL"; \
	$(PYTHON) train_pick_place.py --resume --curriculum ordered --max-hours $$HOURS_VAL --timesteps 999999999

# End-to-end replication of every result from scratch. Sequences all training
# for the three curricula, then the two auxiliary pick_place baselines, then
# regenerates every figure. This is the single command to reproduce the thesis.
#   - run-ordered-match-all needs run-all (for the hours budget) AND run-ordered
#     (to resume its checkpoint), so it runs after both.
#   - run-scratch-full is independent of the others but must precede figures.
# Uses recursive $(MAKE) so each step gets its own recipe invocation; a failure
# halts the pipeline instead of silently poisoning later stages.
replicate:
	$(MAKE) run-all
	$(MAKE) run-ordered
	$(MAKE) run-random
	$(MAKE) run-ordered-match-all
	$(MAKE) run-scratch-full
	$(MAKE) figures

figures:
	cd analysis && ../$(PYTHON) plot_curricula.py
	cd analysis && ../$(PYTHON) plot_cumulative.py
	cd analysis && ../$(PYTHON) plot_scratch_vs_chain_full.py
	cd analysis && ../$(PYTHON) plot_ordered_matched_vs_all.py
