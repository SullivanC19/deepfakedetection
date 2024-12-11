# DeepFakeDetction

Hi CS231N TAs! Thank you for taking the time to evaluate our project. Please follow the instructions below to replicate the results from our final report and poster presentation.

We've excluded our trained model weights to keep this repository light. If you'd like them, please email me at colins26@stanford.edu, and I'd be happy to send them over.

## Setup and Execution
- Make sure you have conda installed
- Create the environment with `conda env create -f environment.yml`
- Activate the environment with `conda activate deepfakesonly`
- Run `python download.py` to download data (you may need to enter your Kaggle username and API key)
    - To get an API key, go to the 'Account' tab of your user profile and select 'Create New Token'. This will trigger the download of kaggle.json, a file containing your API credentials.
- Run `python train.py` to train all models
    - If on a slurm cluster, you can run `sbatch script.slurm` to train models in parallel
- Run `python analyze.py` to evaluate all models and reproduce all plots

## Tensorboard (optional)
- To view intermediate training results, run `tensorboard --logdir=logs --port=6006` and open up `https://localhost:6006` in a browser
- If on a gcloud compute cluster, you can run `gcloud compute ssh [INSTANCE_NAME] -- -NfL 6006:localhost:6006` to forward the tensorboard output to your local machine
