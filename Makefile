# Create the local env
install:
	#conda create -n mlfpyt python=3.8 -y; conda activate mlfpyt
	pip install numpy
	pip install mlflow
	pip install torch
	pip install torchvision
	pip install flake8

# Setup infra
infra:
	./setup/create-resources.sh

# Run the model training
local_run:
	python ./scripts/main.py

# Copy the model contents to the root of the directory
# Trigger the model registration, endpoint & deployment creation
aml_deploy:
	rm -rf ./model
	python ./scripts/find_model_files.py
	./inference/deploy.sh

# Create fake test data, and manually upload in the 'Test' GUI
get_prediction:
	python ./scripts/make_test_data.py
	cat ./data/request.json
