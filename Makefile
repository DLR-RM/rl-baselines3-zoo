LINT_PATHS = *.py tests/ scripts/ rl_zoo3/ hyperparams/python/*.py docs/conf.py

# Run pytest and coverage report
pytest:
	./scripts/run_tests.sh

# check all trained agents (slow)
check-trained-agents:
	python -m pytest -v tests/test_enjoy.py -k trained_agent --color=yes

mypy:
	mypy ${LINT_PATHS} --install-types --non-interactive

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff ${LINT_PATHS} --select=E9,F63,F7,F82 --show-source
	# exit-zero treats all errors as warnings.
	ruff ${LINT_PATHS} --exit-zero

format:
	# Sort imports
	ruff --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	ruff --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint

doc:
	cd docs && make html

spelling:
	cd docs && make spelling

clean:
	cd docs && make clean

docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

# PyPi package release
release:
	# rm -r build/* dist/*
	python -m build -s
	python -m build -w
	twine upload dist/*

# Test PyPi package release
test-release:
	# rm -r build/* dist/*
	python -m build -s
	python -m build -w
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: lint format check-codestyle commit-checks doc spelling docker type pytest
