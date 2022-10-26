LINT_PATHS = *.py tests/ scripts/ rl_zoo3/

# Run pytest and coverage report
pytest:
	./scripts/run_tests.sh

# check all trained agents (slow)
check-trained-agents:
	python -m pytest -v tests/test_enjoy.py -k trained_agent --color=yes

# Type check
type:
	pytype -j auto rl_zoo3/ tests/ scripts/ -d import-error

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

# PyPi package release
release:
	# rm -r build/* dist/*
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload dist/*

# Test PyPi package release
test-release:
	# rm -r build/* dist/*
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: lint format check-codestyle commit-checks doc spelling  docker type pytest
