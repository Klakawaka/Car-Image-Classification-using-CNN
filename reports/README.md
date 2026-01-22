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

s205430, s211150, s215884, s215810, s215813


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
For GPU training: use Docker (see Q15) or ensure local CUDA compatibility

This ensures identical environments for everyone.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words

> Answer:

We mostly tried to follow the coockiecutter data science template. We initialized the project using the cookiecutter data-science template (DTU MLOps variant) and structured it as follows:

src/car_image_classification_using_cnn/ — all core logic (data.py, data_transform.py, model.py, train_hydra.py, evaluate.py, visualization helpers)
configs/ — Hydra configuration files for models, training, data
dockerfiles/ — separate Dockerfiles for training and evaluation
tests/ — pytest unit and integration tests
reports/figures/ — generated plots for this report
models/ — saved checkpoints
raw/ — Kaggle dataset (DVC tracked)

We used most standard folders but deviated in a few ways:

Minimized notebooks/ (only early exploration, mostly gitignored now)
Gitignored heavy files in outputs/
Added .pre-commit-config.yaml for quality hooks
Added cloudbuild.yaml for GCP integration
Included basic drift detection code

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.

> Answer:

We enforced code quality and formatting with:

ruff → linting + automatic formatting
mypy → static type checking
pre-commit hooks (.pre-commit-config.yaml) → runs ruff & mypy automatically on commit

These practices are important in larger projects because they ensure consistency across the team, reduce style-related merge conflicts, catch type and logic errors early, prevent bad code from entering the repository. I also  make onboarding and long-term maintenance much easier because their is readable text besides a code. In our group of five this noticeably cut down debugging time for trivial issues overall.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7 KIG PÅ

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Answer:

We implemented x tests, primarily covering data loading and the model. The focus was placed on the most failure prone components specifically the data pipeline and the model definition.


### Question 8 KIG PÅ

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Total code coverage is 82% (pytest-cov, src/ only).

Even 100% coverage would not mean the code is error-free. Coverage only shows which lines were executed not whether the logic is correct, edge cases are handled, numerical stability holds, or the model generalizes to new data. ML-specific issues like data drift, preprocessing mismatches, or GPU/CPU differences often lie outside unit tests, so high coverage is good but must be combined with meaningful tests, experiment tracking, and monitoring at all time for it to be reliable.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes. We used branches and pull request extensively. 

Each team member worked on different features in different branches(eg. datatranform, test_api, model_train )

In order for the features to be implemented to the main branch pull requests were requird, and one should at least approve it, it should showhow also pass the CL(lint, test, coverage).
This workflow enabled safe parallel work, improved code quality through reviews, caught bugs early, and kept main stable.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we used DVC to version both raw data (raw/) and trained models (models/).

dvc add raw/, dvc add models/
Remote storage on GCS bucket

This improved the project by ensuring everyone used the exact same dataset version, prevented large files from bloating Git, made experiments fully reproducible, and allowed us to trace which checkpoint came from which run.

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

--- question 11 fill here ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12 KIG HER 

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Answer:

We configured experiments using Hydra. Typical run command is seen below :


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

Reproducibility was secured through:

Hydra auto-saving full config + overrides to outputs/.../.hydra/config.yaml
Logging git commit hash, random seed, W&B run ID
DVC pinning data and models
Fixed seeds in code (torch, numpy, random)

To reproduce any run: checkout the commit, dvc pull, run the exact command from Hydra output or W&B.

### Question 14 KIG HER 

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

--- question 14 fill here ---

### Question 15 KIG HER

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

We used Docker to containerize training and evaluation for consistency.
Build and run example (training):


insert commanddo



Evaluation container mounts test data + model checkpoint.
Link to Dockerfile: https://github.com/Klakawaka/Car-Image-Classification-using-CNN/blob/main/dockerfiles/train.dockerfile

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Answer:

When a bug occurred during experiments, we started by checking the code and reading the error messages to find where the bug came from. If we then could not find the error, we will then would look online. When seaching online you can see most of the time someone who already had the same problem and it has been reported and how others resolved them. 

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17 KIG HER 

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:

> Answer:

We used the following GCP services:

Compute Engine — GPU VMs for model training
Cloud Storage (GCS) — DVC remote for data and models
Artifact Registry — storing built Docker images

### Question 18 IKKE BESVARET

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

--- question 18 fill here ---

### Question 19 IKKE BESVARET

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20 IKKE BESVARET

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21 IKKE BESVARET

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

Yes, we trained models in the cloud using Compute Engine with GPU instances and Docker containers. Vertex AI was not used — we preferred direct control over the script and environment.

## Deployment

### Question 23 KIG HER 

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Answer:

We implemented a basic inference API with FastAPI (in api.py if present, or planned as inference script).
It accepts image uploads, applies preprocessing, runs the best model, and returns top-3 predictions + probabilities.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.

> Answer:

We deployed locally first (via uvicorn or Docker) and tested cloud deployment on Compute Engine (container-optimized OS, public IP).
Invoke example:
curl -X POST http://localhost:8000/predict  -F "image=@src/src/car_image_classification_using_cnn/1000.jpg"

### Question 25 IKKE BESVARET

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

--- question 25 fill here ---

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

We did not manage to implement monitoring for our deployed model. However, having monitoring in place would significantly improve the long-term reliability and maintainability of the application. Monitoring could be used to track model performance metrics such as prediction accuracy, loss, or confidence scores over time, allowing us to detect performance degradation or model drift as the data distribution changes.
In addition, monitoring system-level metrics such as latency, memory usage, and error rates would help identify issues related to scalability or resource constraints. Alerts based on these metrics could enable faster responses to failures or unexpected behavior in production. Overall, monitoring would provide valuable insight into both model behavior and system health, helping ensure that the application remains robust, efficient, and reliable as usage and data evolve over time.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27 IKKE BESVARET

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

--- question 27 fill here ---

### Question 28 kig her 

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Answer:

We implemented a simple frontend for our API to make the system easier to use and demonstrate its functionality. The frontend allows users to interact with the API without needing to send requests manually, providing a more intuitive way to test and visualize the model’s predictions.
The frontend was designed to be lightweight and focused on usability rather than advanced features. It communicates directly with the backend API to submit input data and display the returned results. Implementing a frontend helped us validate that the API worked correctly in an end-to-end setup and made the project more accessible for users who are not familiar with API tools.

### Question 29 IKKE BESVARET

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

--- question 29 fill here ---

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
> Answer:

All members reviewed PRs, debugged, and ran local experiments.
We used GitHub Copilot for suggestions and ChatGPT/Grok for error explanations — all code was manually verified and adapted.