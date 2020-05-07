# Run pytest and coverage report
pytest:
	./scripts/run_tests.sh

# Type check
type:
	pytype .

docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

.PHONY: docker
