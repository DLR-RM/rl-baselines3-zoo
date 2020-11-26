LINT_PATHS = *.py tests/ scripts/ utils/

# Run pytest and coverage report
pytest:
	./scripts/run_tests.sh

# check all trained agents (slow)
check-trained-agents:
	python -m pytest -v tests/test_enjoy.py -k trained_agent

# Type check
type:
	pytype -j auto ${LINT_PATHS}

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --count --exit-zero --statistics


format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l 127 ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check -l 127 ${LINT_PATHS}

commit-checks: format type lint

docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

.PHONY: docker lint type pytest
