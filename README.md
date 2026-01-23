# Machine learning group
### Project structure
```
CAR-IMAGE-CLASSIFICATION-USING-CNN
├── README.md                 <- Top-level README for developers and reviewers
├── AGENTS.md                 <- AI agent instructions and guidelines
├── LICENSE                   <- Project license
│
├── pyproject.toml            <- Python project metadata and dependencies (uv)
├── uv.lock                   <- Locked dependencies for reproducibility
├── tasks.py                  <- Invoke task automation commands
│
├── .devcontainer/            <- Development container configuration
├── .dvc/                     <- DVC metadata for data and model versioning
├── .github/                  <- GitHub configuration
│   ├── workflows/            <- GitHub Actions CI/CD workflows
│   │   ├── gcs-sync.yaml     <- Docker build and GCR push workflow
│   │   └── tests.yaml        <- Automated testing workflow
│   └── dependabot.yaml       <- Dependency update configuration
│
├── configs/                  <- Hydra configuration files
│   ├── config.yaml           <- Main configuration
│   ├── data/                 <- Data-related configs
│   ├── experiment/           <- Experiment configs (customcnn, resnet variants)
│   ├── model/                <- Model architecture configs
│   └── training/             <- Training hyperparameter configs
│
├── dockerfiles/              <- Dockerfiles for containerized execution
│   ├── api.dockerfile        <- FastAPI service container
│   ├── evaluate.dockerfile   <- Model evaluation container
│   └── train.dockerfile      <- Training container
│
├── backend.dockerfile        <- Backend service Dockerfile
├── frontend.dockerfile       <- Frontend service Dockerfile
├── docker-compose.yml        <- Multi-container Docker setup
├── cloudbuild.yaml           <- Google Cloud Build configuration
│
├── docs/                     <- Project documentation (MkDocs)
│   ├── mkdocs.yaml           <- MkDocs configuration
│   ├── README.md             <- Documentation overview
│   └── source/               <- Documentation source files
│
├── data/                     <- Processed data directory
│   └── processed/            <- Cleaned and transformed data
│
├── raw/                      <- Raw data directory (DVC-tracked)
│   ├── train/                <- Training images by class
│   └── test/                 <- Test images by class
├── raw.dvc                   <- DVC file for raw data versioning
│
├── models/                   <- Trained models and checkpoints
│   ├── best_model.pth        <- Best model checkpoint
│   └── best_model.pth.dvc    <- DVC file for model versioning
│
├── notebooks/                <- Jupyter notebooks for exploration
├── outputs/                  <- Training outputs (logs, metrics, predictions)
│
├── reports/                  <- Project report and analysis
│   ├── README.md             <- **EXAM REPORT**
│   ├── figures/              <- Figures and visualizations
│   └── Alerts.png            <- Monitoring alerts dashboard
│
├── scripts/                  <- Utility scripts
│   ├── batch_test.py         <- Batch prediction testing
│   ├── export_onnx.py        <- ONNX model export
│   └── manual_api_client.py  <- API testing client
│
├── src/                      <- Source code for the project
│   ├── main.py               <- FastAPI application (prediction service)
│   └── car_image_classification_using_cnn/
│       ├── __init__.py
│       ├── data.py           <- Dataset loading and preprocessing
│       ├── data_transform.py <- Image transformations and augmentation
│       ├── drift_detection.py <- Data drift detection and monitoring
│       ├── model.py          <- CNN model architectures (ResNet, Custom)
│       ├── train.py          <- Model training with Typer CLI
│       ├── train_hydra.py    <- Training with Hydra configuration
│       ├── evaluate.py       <- Model evaluation script
│       ├── visualize.py      <- Results visualization
│       └── logger.py         <- Centralized logging (Loguru)
│
├── tests/                    <- Unit and integration tests (205 tests, 82% coverage)
│   ├── __init__.py
│   ├── test_data.py          <- Dataset and preprocessing tests
│   ├── test_data_transform.py <- Transform function tests
│   ├── test_model.py         <- Model architecture tests
│   ├── test_training.py      <- Training loop tests
│   ├── test_train_hydra.py   <- Hydra training tests
│   ├── test_evaluate.py      <- Evaluation function tests
│   ├── test_logger.py        <- Logging utility tests
│   ├── test_drift_detection.py <- Drift detection tests
│   └── integrationtests/     <- API integration tests
│       ├── conftest.py       <- Test fixtures
│       └── test_apis.py      <- FastAPI endpoint tests
│
├── frontend.py               <- Gradio web interface
├── client_bento.py           <- BentoML client
├── service.py                <- Service management script
│
├── dist/                     <- Built distribution packages
├── htmlcov/                  <- Test coverage HTML reports
│
├── .dockerignore             <- Files ignored by Docker
├── .gitignore                <- Files ignored by Git

```
# Project Description

## Goal
This is the project description for Group 140 in the 02476 Machine Learning Operations course at DTU. The overall goal of the project is to apply the material learned throughout the course to a realistic machine learning problem, with a strong focus on MLOps principles such as reproducibility, experiment tracking, configuration management, and deployment. The project will serve as the basis for the final exam and will be documented through code and a written report.

## Framework
The framework chosen for this project is PyTorch, with the TIMM (PyTorch Image Models) library used for computer vision. TIMM provides access to a wide range of pretrained convolutional neural networks (CNNs), which allows us to efficiently fine-tune existing models instead of training from scratch. The framework will be integrated into a structured codebase with version control and configuration management.

## Data
The dataset used in this project is the Cars Image Dataset from Kaggle. The dataset consists of labeled images of cars belonging to different car brands, Audi, Hyundai Creta, Rolls Royce, Swift, Tata Safari, Toyota Innova. Each image is associated with a single car brand, making the task a supervised multi-class image classification problem. The dataset provides a realistic and diverse set of car images suitable for training and evaluating convolutional neural networks.

Size: 37.52 MB


[Kaggle Cars Image Dataset](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset/data)

## Models
The project focuses on performing an image classification task on the car brand dataset using convolutional neural networks. We expect to use pretrained models from the TIMM framework, such as ResNet, EfficientNet, MobileNet, and ConvNeXt, as baseline and comparison models. The models will be fine-tuned on the dataset, and their performance will be evaluated using standard classification metrics.

## Docker Support

This project includes Docker support for reproducible training and evaluation. See [docs/DOCKER.md](docs/DOCKER.md) for comprehensive usage instructions.

### Quick Start

Build and run training:
```bash
docker build -f dockerfiles/train.dockerfile . -t car-classifier-train:latest
docker run --rm -v $(pwd)/models:/app/models car-classifier-train:latest
```

Build and run evaluation:
```bash
docker build -f dockerfiles/evaluate.dockerfile . -t car-classifier-eval:latest
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/raw/test:/app/raw/test car-classifier-eval:latest models/best_model.pth
```
