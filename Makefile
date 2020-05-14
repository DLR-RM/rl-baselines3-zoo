# Run pytest and coverage report
pytest:
	./scripts/run_tests.sh

# Type check
type:
	pytype .

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 . --count --exit-zero --statistics

docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

.PHONY: docker lint type pytest
