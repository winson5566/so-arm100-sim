.PHONY: setup baseline collect collect50 train eval clean

PYTHON := .venv/bin/python
LEROBOT_TRAIN := .venv/bin/lerobot-train

setup: ## Create venv and install dependencies
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e .

baseline: ## Run the scripted pick & place baseline once
	PYTHONPATH=src $(PYTHON) -m so_arm100_sim.scripts.run_baseline

collect: ## Collect demonstrations (default: 20 episodes)
	PYTHONPATH=src $(PYTHON) -m so_arm100_sim.scripts.collect_demos --num-episodes 20

collect50: ## Collect 50 demonstrations (ACT paper scale) with +/-1cm cube jitter
	PYTHONPATH=src $(PYTHON) -m so_arm100_sim.scripts.collect_demos --num-episodes 50 --cube-jitter 0.01 --video-codec h264_videotoolbox

train: ## Train ACT on the collected dataset
	$(LEROBOT_TRAIN) --config_path=configs/train_act.yaml

eval: ## Evaluate the trained ACT checkpoint in simulation
	PYTHONPATH=src $(PYTHON) -m so_arm100_sim.scripts.eval_act

clean: ## Remove generated data and outputs
	$(PYTHON) -c "import shutil; [shutil.rmtree(p) for p in ('data','outputs') if __import__('pathlib').Path(p).exists()]"
