# CatBoost Inference Container

A containerized inference service for CatBoost classification models with support for numerical, categorical, and text features.

## Overview

This project provides a production-ready inference container for serving CatBoost classification models. It includes:

- REST API endpoints for predictions
- Comprehensive metrics calculation and visualization
- Support for multiple input types (CSV, JSON)
- Health check endpoint
- AWS SageMaker compatibility

## Requirements

- Python 3.12+
- Docker
- AWS CLI (for ECR deployment)

Core dependencies:

```bash
catboost==1.2.8
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.2
flask
gunicorn
loguru
```

## Installation

### Using Poetry (Recommended)

1. Install Poetry if you haven't already:
2. 
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

1. Clone the repository:
```bash
git clone https://github.com/yourusername/catboost_inference.git
cd catboost_inference
```

1. Install dependencies using Poetry:
```bash
poetry install
poetry shell  # Activates the virtual environment
```

### Using pip (Alternative)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/catboost_inference.git
cd catboost_inference
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Development Installation

For development, install additional dependencies:
```bash
poetry install --with dev
```

This will install development dependencies like:
- pytest
- pytest-cov
- black
- isort
- mypy
- flake8

## Project Structure

```
catboost_inference/
├── src/
│   ├── inference/          # Core inference code
│   │   └── predictor.py    # Model serving logic
│   └── utils/
│       └── metrics.py      # Metrics calculation
├── tests/                  # Test suites
├── Dockerfile             # Container definition
└── build_and_push.sh     # ECR deployment script
```

## Usage

### Local Development

1. Run tests:
```bash
pytest tests/ -v
```

2. Start the Flask development server:
```bash
python src/inference/serve
```

### Docker Build and Deploy

Build and push to Amazon ECR:
```bash
./src/build_and_push.sh <account_id> <ecr_repo_name> <region> [tag_name]
```

### API Endpoints

#### Health Check
```http
GET /ping
```

#### Prediction
```http
POST /invocations
Content-Type: text/csv

<csv-data>
```

## Metrics and Visualization

The service generates:
- ROC curves
- Confusion matrices
- Precision-recall metrics
- Classification thresholds analysis

Generated plots are saved to the configured image directory.

## Testing

The project includes comprehensive test suites:
- Unit tests for core functionality
- Integration tests for API endpoints
- Metrics calculation validation
- Plot generation verification

Run tests with coverage:
```bash
pytest tests/ -v --cov=src
```

## AWS SageMaker Deployment

1. Deploy using SageMaker Python SDK:

```python
from sagemaker.model import Model

model = Model(
    image_uri='<ecr-image-uri>',
    model_data='s3://<bucket>/<path>/model.tar.gz',
    role='<role-arn>'
)

predictor = model.deploy(
    instance_type='ml.c5.xlarge',
    initial_instance_count=1
)
```