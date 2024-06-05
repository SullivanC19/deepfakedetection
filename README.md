# DeepFakesOnly

## Setup and Run Training
    curl -sL \
      "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
      "Miniconda3.sh"
    bash "Miniconda3.sh" -b
    rm "Miniconda3.sh"
    conda init bash
    sudo chown 1002:1003 /opt/conda/pkgs/cache/*
    conda env create -f environment.yml
    conda update -n base conda
    conda activate deepfakesonly
    python main.py & tensorboard --logdir=logs --port=6006 && fg
Enter your Kaggle username and password to download the dataset

## View Tensorboard Locally
In cloud, run...
    tensorboard --logdir=logs --port=6006 &
Locally, run...
    gcloud compute ssh [INSTANCE_NAME] -- -NfL 6006:localhost:6006
