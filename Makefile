.PHONY: setup data train run clean

setup:
	pip install -r requirements.txt

data:
	python3 data/mock_generator.py

train:
	python3 src/model.py

run:
	python3 src/router.py

clean:
	rm -f data/historical_logs.csv
	rm -f models/latency_predictor.pkl
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
