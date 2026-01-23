# Exam report for 02476 Machine Learning Operations group 140

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

We are group 140.

### Question 2
> **Enter the study number for each member in the group**

> Answer:

s205430, s211150, s215784, s215810, s215813


### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
> Answer:

We used PyTorch as the main framework, extended with timm (PyTorch Image Models) library for pretrained CNNs (ResNet, EfficientNet, MobileNet, ConvNeXt). TIMM helped tremendously by providing high-quality ImageNet-pretrained weights, enabling fast and effective transfer learning on our small Kaggle Cars dataset (37.52 MB, 6 classes: Audi, Hyundai Creta, Rolls Royce, Swift, Tata Safari, Toyota Innova). This shifted focus to MLOps (Hydra configs, DVC data versioning, Docker reproducibility, GitHub Actions + Cloud Build CI) rather than low-accuracy scratch training.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words

> Answer:

Dependencies managed with Poetry (pyproject.toml defines packages like torch, timm, hydra-core, dvc[gcs], pytest, ruff, pre-commit; locked via poetry.lock or uv.lock) and also git.
New team member process:

Clone repo: git clone https://github.com/Klakawaka/Car-Image-Classification-using-CNN
Install Poetry (pip install poetry)
Run poetry install → creates venv + installs deps
Activate: poetry shell
For GPU-based training or cloud deployment, we rely on Docker images that encapsulate the same dependencies, eliminating host-specific issues such as CUDA mismatches. This combination of Poetry for local development and Docker for deployment ensured consistent environments, reduced onboarding friction, and minimized “works on my machine” problems.

This ensures identical environments for everyone.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words

> Answer:

We initialized the project using the DTU MLOps cookiecutter data science template, which provided a standardized and well-organized project structure. The core logic resides in src/car_image_classification_using_cnn/, containing modules for data loading, preprocessing, model definition, training with Hydra, evaluation, and visualization utilities. Configuration files are stored in the configs/ directory and managed using Hydra, allowing flexible experiment configuration.

We used tests/ for unit and integration testing, models/ for trained checkpoints, raw/ for DVC-tracked datasets, and reports/figures/ for generated plots. Dockerfiles were separated into a dockerfiles/ directory to clearly distinguish training, evaluation, and API containers.

We deviated slightly from the original template by minimizing the use of notebooks after initial exploration and by adding MLOps-specific components such as .pre-commit-config.yaml, cloudbuild.yaml, and drift detection utilities. These changes were made to better support automation, reproducibility, and deployment, while still adhering to the spirit of the cookiecutter structure.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.

> Answer:

We enforced code quality using several automated tools. Ruff was used for linting and formatting, ensuring consistent coding style and catching common programming errors. Mypy was used for static type checking, helping us detect type mismatches and logical issues early in development. These checks were integrated into pre-commit hooks, meaning code could not be committed unless it passed linting and type checks.

We also added docstrings and inline comments for key functions, particularly in the data pipeline, training loop, and API code. This improved readability and made the codebase easier to understand for all team members.

These practices are especially important in larger projects because they reduce technical debt, prevent subtle bugs from propagating, and make collaboration more efficient. Consistent formatting reduces merge conflicts, typing catches errors before runtime, and documentation improves onboarding and long-term maintainability. In a team of five, these tools significantly reduced debugging time and improved overall code reliability.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Answer:

We implemented approximately 45 tests across multiple test files. The tests cover:

- **Data loading and transforms** (`test_data.py`) - verifying dataset creation, data augmentation, and preprocessing
- **Model architecture** (`test_model.py`) - checking model output shapes, parameter counts, and forward passes
- **Training functions** (`test_training.py`) - validating training/validation loops, gradient flow, and optimizer behavior
- **Drift detection** (`test_drift_detection.py`) - testing feature extraction and drift report generation
- **API endpoints** (`test_apis.py`) - ensuring prediction and monitoring endpoints work correctly

The focus was on the most failure-prone components: the data pipeline, model definition, and API functionality.


### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Total code coverage is 82% (pytest-cov, src/ only, excluding tests and configs).

Even 100% coverage would not mean the code is error-free. Coverage only shows which lines were executed not whether the logic is correct, edge cases are handled, numerical stability holds, or the model generalizes to new data. ML-specific issues like data drift, preprocessing mismatches, or GPU/CPU differences often lie outside unit tests, so high coverage is good but must be combined with meaningful tests, experiment tracking, and monitoring at all time for it to be reliable.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we actively used branches and pull requests throughout the project. Each team member worked on separate feature branches (eg. datatranform, test_api, model_train ), such as data preprocessing, model training, API development, or testing. This allowed parallel development without interfering with the stability of the main branch.

All changes were merged into main through pull requests. Each pull request required at least one team member’s approval and had to pass all CI checks or mostly all, including linting, formatting, and unit tests. This ensured that new code met quality standards before being integrated.

This workflow improved code quality through peer review, helped catch bugs early, and encouraged knowledge sharing within the group. It also provided a clear history of changes and made it easier to revert or debug problematic commits. Using branches and pull requests kept the main branch stable and production-ready throughout the project.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Using DVC significantly improved reproducibility and collaboration. All team members could pull the exact same version of the dataset and models using dvc pull, ensuring consistency across experiments. This prevented issues where different dataset versions could lead to inconsistent results.

Using DVC significantly improved reproducibility and collaboration. All team members could pull the exact same version of the dataset and models using dvc pull, ensuring consistency across experiments. This prevented issues where different dataset versions could lead to inconsistent results.

dvc add raw/, dvc add models/
Remote storage on GCS bucket

Additionally, DVC allowed us to trace which data version was used to train each model checkpoint, improving experiment traceability. It also kept large files out of Git, keeping the repository lightweight. Overall, DVC played a crucial role in ensuring reproducible experiments, reliable collaboration, and clean version control of both data and models.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:
We have organized our continuous integration into two main GitHub Actions workflows:

1. **tests.yaml** - Runs on every push and pull request:
   - Executes pytest test suite with coverage reporting
   - Runs ruff for linting (`ruff check .`)
   - Runs ruff for code formatting validation (`ruff format --check .`)
   - Tests on Ubuntu (Linux) with Python 3.12
   - Uses caching for `uv` dependencies to speed up workflow execution
   - Link: [tests.yaml](https://github.com/Klakawaka/Car-Image-Classification-using-CNN/blob/main/.github/workflows/tests.yaml)

2. **gcs-sync.yaml** - Runs on push to main branch:
   - Builds Docker images for training and API
   - Pushes images to Google Container Registry (GCR)
   - Optionally deploys to Google Cloud Run
   - Link: [gcs-sync.yaml](https://github.com/Klakawaka/Car-Image-Classification-using-CNN/blob/main/.github/workflows/gcs-sync.yaml)

We use pre-commit hooks locally (`.pre-commit-config.yaml`) that run the same checks before commits, catching issues early. The CI setup ensures that all code merged to main is tested, properly formatted, and passes linting. While we currently only test on one OS/Python version, this covers our deployment target environment and keeps CI runtime reasonable.


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Answer:

We configured experiments using Hydra with YAML config files in the `configs/` directory. Example run command:

```bash
uv run python src/car_image_classification_using_cnn/train_hydra.py \
    model=resnet \
    training.num_epochs=20 \
    training.batch_size=32 \
    training.learning_rate=0.001
```

Hydra automatically saves the full configuration to `outputs/<timestamp>/.hydra/config.yaml`, making every run reproducible. We can override any parameter from the command line while keeping defaults in YAML files.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured experiment reproducibility through a combination of configuration management, version control, and logging. All experiments were configured using Hydra, which automatically saves the full configuration (including command-line overrides) to the output directory of each run. This ensures that no hyperparameter or setting is lost.

We also logged the Git commit hash, random seeds, and Weights & Biases run IDs for every experiment, creating a strong link between code, configuration, and results. Data and model versions were pinned using DVC, guaranteeing that the same dataset and checkpoints could be retrieved later.

To reproduce an experiment, one simply needs to check out the corresponding Git commit, run dvc pull to retrieve the correct data and models, and rerun the exact Hydra command stored in the output directory or W&B dashboard. This setup ensured full traceability and reproducibility across environments and over time.

To reproduce any run: checkout the commit, dvc pull, run the exact command from Hydra output or W&B.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

We used Weights & Biases to track our experiments. Key metrics tracked include:

- **Training Loss**: Monitors how well the model is learning from the training data. A decreasing trend indicates the model is improving.
- **Validation Loss**: Critical for detecting overfitting - if validation loss increases while training loss decreases, the model is memorizing rather than generalizing.
- **Training/Validation Accuracy**: Percentage of correctly classified images. We track both to ensure the model generalizes well to unseen data.
- **Learning Rate**: Logged to understand the optimization dynamics and correlate performance with different LR schedules.
- **Epoch Time**: Helps identify performance bottlenecks and compare efficiency across different model architectures.

We also logged:
- **System metrics** (GPU utilization, memory usage) to optimize resource usage
- **Hyperparameters** (batch size, model architecture, pretrained weights) to compare configurations
- **Git commit hash** to link each run to specific code versions

This comprehensive tracking allowed us to quickly identify the best-performing model (ResNet with pretrained weights achieved ~92% validation accuracy), debug training issues (like learning rate being too high), and make informed decisions about model selection. The W&B dashboard made it easy to compare multiple runs side-by-side and share results with team members.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We developed three Docker images for different purposes:

1. **Training image** ([train.dockerfile](https://github.com/Klakawaka/Car-Image-Classification-using-CNN/blob/main/dockerfiles/train.dockerfile)):
```bash
docker build -f dockerfiles/train.dockerfile -t train:latest .
docker run --rm -v $(pwd)/models:/app/models train:latest --num_epochs=10
```

2. **Evaluation image** ([evaluate.dockerfile](https://github.com/Klakawaka/Car-Image-Classification-using-CNN/blob/main/dockerfiles/evaluate.dockerfile)):
```bash
docker build -f dockerfiles/evaluate.dockerfile -t eval:latest .
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/raw/test:/app/raw/test eval:latest models/best_model.pth
```

3. **API image** ([api.dockerfile](https://github.com/Klakawaka/Car-Image-Classification-using-CNN/blob/main/dockerfiles/api.dockerfile)):
```bash
docker build -f dockerfiles/api.dockerfile -t api:latest .
docker run -p 8000:8000 api:latest
```

All images use `uv` for dependency management and are built with `--platform linux/amd64` for cloud deployment compatibility. Volume mounts (`-v`) ensure models and data persist across container restarts.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Answer:

When encountering bugs during experiments, we followed a systematic debugging approach. First, we examined error messages and stack traces to identify the source. We used Python's built-in debugger (`pdb`) and VS Code's debugging tools to step through code and inspect variables. When errors persisted, we consulted online resources like Stack Overflow and GitHub issues, where similar problems were often documented with solutions.

For profiling, we used PyTorch's built-in profiler to identify performance bottlenecks in our training loop. We discovered that data loading was a bottleneck and addressed it by increasing the number of workers in our DataLoader. We also used the `--profile_run` flag in [`train.py`](src/car_image_classification_using_cnn/train.py) to profile specific training runs.

While our code works well for this project, it's far from perfect. We identified areas for improvement such as more efficient data preprocessing, better error handling, and optimization of model inference speed. Continuous profiling and optimization would be necessary for production deployment.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:

> Answer:

We used the following GCP services:

- **Compute Engine**: Virtual machines for GPU-accelerated model training. We used n1-standard-4 instances with NVIDIA Tesla T4 GPUs to train our models significantly faster than on local CPUs.

- **Cloud Storage (GCS)**: Object storage for data and model versioning. We configured it as our DVC remote storage, storing raw datasets (~37.52 MB), processed data, and trained model checkpoints.

- **Container Registry / Artifact Registry**: Docker image repository where we stored our training, evaluation, and API container images. These images were built automatically via Cloud Build and deployed to Compute Engine and Cloud Run.

- **Cloud Build**: Automated CI/CD service that builds our Docker images whenever code is pushed to the main branch on GitHub, following our [`cloudbuild.yaml`](cloudbuild.yaml) configuration.

- **Cloud Run** (optional/experimental): Serverless container platform where we tested deploying our FastAPI backend for auto-scaling inference.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used Compute Engine to run GPU-accelerated training jobs in the cloud. Specifically:

**VM Configuration:**
- **Machine type**: n1-standard-4 (4 vCPUs, 15 GB memory)
- **GPU**: NVIDIA Tesla T4 (16 GB GPU memory)
- **OS**: Container-Optimized OS (for running Docker containers)
- **Region**: europe-west1 (Belgium) for lower latency

**Workflow:**
1. Created VM instance with GPU attached
2. SSH into the instance and pulled our training Docker image from GCR
3. Ran training: `docker run --gpus all gcr.io/[PROJECT]/train:latest`
4. Model checkpoints automatically saved to mounted Cloud Storage bucket via DVC
5. Stopped VM after training to minimize costs

This setup provided ~10x speedup compared to CPU-only training while keeping costs manageable through on-demand GPU usage.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![GCP Bucket Overview](figures/bucket.png)

Our GCS bucket `dtu-mlops-car-classification_bucket` contains:
- `raw/` - DVC-tracked training and test datasets (6 car classes)
- `models/` - Trained model checkpoints (.pth files)
- `.dvc/cache/` - DVC cache for data versioning

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![GCP Artifact Registry](figures/registry.png)

Our Artifact Registry contains:
- `train:latest` - Training container (~2.5 GB)
- `api:latest` - FastAPI inference container (~1.8 GB)
- Multiple tagged versions corresponding to different commits

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Cloud Build History](figures/cloud_build.png)

Our Cloud Build history shows:
- Automated builds triggered by pushes to main branch
- Build duration typically 5-8 minutes
- Successful builds result in images pushed to GCR

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we successfully trained models in the cloud using Compute Engine with GPU instances and Docker containers. We did not use Vertex AI because we preferred direct control over our training script, environment, and resource allocation.

**Training workflow:**
1. Created a GPU-enabled VM instance (n1-standard-4 with Tesla T4)
2. Installed Docker and authenticated with Google Container Registry
3. Pulled our training Docker image: `docker pull gcr.io/[PROJECT]/train:latest`
4. Mounted Cloud Storage bucket for model checkpoints
5. Ran training: `docker run --gpus all -v /mnt/gcs/models:/app/models train:latest`
6. Monitored training via Weights & Biases remotely
7. Stopped the VM after training completed to avoid unnecessary costs

This approach gave us full control over dependencies (via Docker), ensured reproducibility, and allowed us to leverage GPU acceleration without expensive local hardware.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we implemented a comprehensive inference API using FastAPI in [`src/main.py`](src/main.py). The API provides several endpoints:

- **`/predict`**: Accepts single image uploads, performs preprocessing (resize to 224×224, normalize), runs inference with our trained CNN, and returns the predicted class with confidence scores and probability distribution.

- **`/predict_batch`**: Handles batch predictions for up to 10 images simultaneously, improving throughput for bulk inference.

- **`/monitoring`**: Generates drift detection reports by comparing recent predictions against training data distribution using Evidently.

- **`/metrics`**: Exposes Prometheus metrics for monitoring request latency, error rates, and throughput.

- **`/health`** and **`/classes`**: Utility endpoints for health checks and class information.

We implemented background tasks to log predictions to `prediction_database.csv` for monitoring purposes. The API is containerized using [`dockerfiles/api.dockerfile`](dockerfiles/api.dockerfile) and can be deployed locally or on Google Cloud Run for auto-scaling.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.

> Answer:

Yes, we successfully deployed our API both locally and in the cloud. Locally, the API can be run directly using Uvicorn or via Docker. Running locally allowed us to rapidly iterate and debug endpoints during development.

For cloud deployment, we containerized the FastAPI application using Docker and pushed the image to Google Container Registry via our CI pipeline. The container was then deployed to Google Cloud Run, which provides serverless execution with automatic scaling. This allowed the API to handle varying request loads without manual infrastructure management.

The deployed service can be invoked via HTTP requests using tools such as curl, Postman, or through our Streamlit frontend. For example, users can upload an image to the /predict endpoint and receive a classification result in JSON format. This deployment approach ensured scalability, reproducibility, and ease of access.

**Local deployment:**
 bash
 Using uvicorn directly
 uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

Or using Docker
docker build -f [api.dockerfile](http://_vscodecontentref_/0) -t api:latest .
docker run -p 8000:8000 api:latest


### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

**Unit Testing:**
Yes, we implemented unit tests for the API in [`tests/integrationtests/test_apis.py`](tests/integrationtests/test_apis.py) using FastAPI's `TestClient`. Tests cover:
- `/predict` endpoint with valid images
- `/predict` endpoint rejects non-image files
- `/classes` endpoint returns correct class information
- `/health` endpoint returns healthy status

**Load Testing:**
We performed basic load testing using [`scripts/batch_test.py`](scripts/batch_test.py), which sends 20 concurrent prediction requests to test throughput. Results showed:
- Average response time: ~200-300ms per request
- Successfully handled 20 concurrent requests without errors
- Memory usage remained stable around 2GB

For production deployment, we would implement more comprehensive load testing with tools like Locust or Apache JMeter to:
- Test with 100+ concurrent users
- Measure request latency at different percentiles (p50, p95, p99)
- Identify breaking point and optimal scaling parameters

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Yes, we implemented monitoring for our deployed model using metrics, logging, and drift detection. The API exposes a /metrics endpoint compatible with Prometheus, providing insights into request counts, error rates, latency, and confidence score distributions. These metrics help monitor system health and performance over time. Its also shown below:

**Metrics** (exposed at `/metrics` endpoint):
- Request counters (total requests, errors, batch requests)
- Latency histograms (prediction duration)
- Confidence score distributions
- Image feature summaries (brightness, contrast )

**Prediction Logging**:
All predictions are logged to [`prediction_database.csv`](prediction_database.csv) with timestamps, predicted classes, confidence scores, and image features. This creates an audit trail for analyzing model behavior over time.

**Drift Detection** (via `/monitoring` endpoint):
The API generates drift reports using Evidently by comparing recent predictions (last N samples) against the training data distribution. It tracks:
- Feature drift (brightness, contrast distributions)
- Class distribution shifts
- Statistical summaries

All predictions are logged to prediction_database.csv, including timestamps, predicted classes, confidence scores, and extracted image features. This creates a historical record that can be analyzed for anomalies or performance degradation.
We also implemented drift detection using Evidently, accessible through the /monitoring endpoint. It compares recent prediction data with the training data distribution to detect feature and class distribution drift. Monitoring is essential for long-term model reliability, as it helps identify when retraining or data updates are required due to changing input distributions.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We used approximately $30-40 in total GCP credits across all team members:

- **Compute Engine with GPU**: ~$15-20 (most expensive, ~$0.35/hour for T4 GPU)
- **Cloud Storage**: ~$2-3 for data and model storage
- **Artifact Registry**: ~$1-2 for Docker image storage
- **Cloud Build**: Minimal cost (free tier covered most builds)

**Reflections on cloud computing:**
Working in the cloud was initially challenging due to the learning curve (IAM permissions, networking, quotas) but ultimately very beneficial. The ability to access GPU resources on-demand was crucial for experimentation. However, costs can escalate quickly if resources aren't managed properly - we learned to:
- Always stop VMs when not in use
- Use preemptible instances for non-critical workloads
- Monitor billing alerts closely

For this course project, cloud resources were appropriate. For larger production systems, careful cost optimization and resource planning would be essential.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Answer:

We implemented a simple frontend for our API to make the system easier to use and demonstrate its functionality. The frontend allows users to interact with the API without needing to send requests manually, providing a more intuitive way to test and visualize the model's predictions.

The frontend was designed to be lightweight and focused on usability rather than advanced features. It communicates directly with the backend API to submit input data and display the returned results. Implementing a frontend helped us validate that the API worked correctly in an end-to-end setup and made the project more accessible for users who are not familiar with API tools.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![System Architecture](figures/architecture_diagram.png)

**System Architecture Overview:**

Our MLOps pipeline starts with **local development** where developers write code using VS Code, organize it using the cookiecutter template, and manage dependencies with `uv`. Code is version controlled with Git and data/models are tracked with DVC (stored in GCS bucket).

When code is pushed to **GitHub**, it triggers **GitHub Actions CI/CD**:
- `.github/workflows/tests.yaml` runs pytest, ruff linting, and coverage checks
- `.github/workflows/gcs-sync.yaml` builds Docker images and pushes to Google Container Registry

**Training workflow:**
1. Pull data from GCS using `dvc pull`
2. Run training locally or on **Compute Engine GPU** VMs using Docker containers
3. Experiments tracked in **Weights & Biases** (loss, accuracy, hyperparameters)
4. Best models saved to `models/` and pushed to GCS with DVC
5. Hydra configs ensure reproducibility

**Inference/Deployment:**
1. **FastAPI backend** (`src/main.py`) loads trained model and exposes endpoints:
   - `/predict` - single image classification
   - `/predict_batch` - batch predictions
   - `/monitoring` - drift detection reports
2. **Streamlit frontend** (`frontend.py`) provides user interface
3. Deployed on **Google Cloud Run** (containerized with api.dockerfile)
4. Prometheus metrics exposed at `/metrics` for monitoring

**Monitoring:**
- Predictions logged to `prediction_database.csv`
- Drift detection compares current predictions vs. training data distribution
- Evidently generates HTML reports for data quality assessment

This architecture ensures reproducibility (Docker, DVC, Hydra), automation (GitHub Actions), scalability (Cloud Run), and maintainability (monitoring, drift detection).

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Answer:

The project involved several challenges, with the majority of our time spent on setting up a robust and reproducible MLOps pipeline rather than on model development itself. One of the main struggles was integrating multiple tools—such as Hydra, DVC, Docker, GitHub Actions, and cloud services—into a single coherent workflow. While each tool is powerful on its own, making them work together correctly required significant experimentation, debugging, and iteration. Another major challenge was ensuring reproducibility across different environments. Differences between local machines, Docker containers, and cloud GPU instances sometimes led to unexpected behavior, especially related to dependencies and file paths. We addressed this by standardizing our setup using Poetry for dependency management, Docker for containerization, and DVC for data and model versioning. This significantly reduced environment-related issues over time.

Overall, the project required balancing model development with engineering and infrastructure work. While challenging, overcoming these obstacles deepened our understanding of real-world MLOps practices and highlighted the importance of reproducibility, automation, and systematic debugging in machine learning projects.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
>
> Answer:

**Individual Contributions:**

- **Student s211150** was in charge of setting up the initial cookiecutter project structure, implementing the data pipeline ([`data.py`](src/car_image_classification_using_cnn/data.py), [`data_transform.py`](src/car_image_classification_using_cnn/data_transform.py)), and establishing DVC for data versioning with GCS.

- **Student s215784** focused on model architecture implementation ([`model.py`](src/car_image_classification_using_cnn/model.py)), integrating TIMM pretrained models, and setting up the training pipeline with Hydra configuration ([`train_hydra.py`](src/car_image_classification_using_cnn/train_hydra.py)).

- **Student s215810** developed the Docker containerization strategy ([dockerfiles/](dockerfiles/)), set up GitHub Actions CI/CD workflows ([`.github/workflows/`](.github/workflows/)), and managed GCP Compute Engine deployments.

- **Student s205430** was responsible for the testing infrastructure ([`tests/`](tests/)), writing unit and integration tests, setting up coverage reporting, implemented the FastAPI backend ([`src/main.py`](src/main.py)),  and developed drift detection functionality ([`drift_detection.py`](src/car_image_classification_using_cnn/drift_detection.py)).

- **Student s215813** Implemented API integration and load tests and set up continuous integration using GitHub Actions to ensure reliability and code quality. Developed a specialized ML deployment API using BentoML and ONNX for efficient and scalable model inference. Created a frontend interface for the API to enable user interaction and demonstrate the deployed machine learning system. ([`frontend.py`](frontend.py)). Implementing pre-commit hooks for code quality ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

**Shared Responsibilities:**
All members contributed to code reviews through pull requests, participated in debugging sessions, configured Weights & Biases experiment tracking, and ran local training experiments. We collaborated on documentation and the final report.

**Generative AI Usage:**
We used GitHub Copilot for code suggestions, and ChatGPT for debugging error messages and understanding unfamiliar concepts. All AI-generated suggestions were manually reviewed, tested, and adapted to fit our project requirements.
