.PHONY: install benchmark convert swarm clean

install:
	pip install -e .[dev]

benchmark:
	python3 benchmarks/sustained_load.py

convert:
	python3 scripts/convert_model.py $(MODEL_PATH)

swarm:
	./scripts/ignite_swarm.sh $(NODES) $(SCALE)

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info build dist
