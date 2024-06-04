# DeepFakesOnly

## Setup and Run Training
    chmod +x ./setup_and_train.sh
    ./setup_and_train.sh
Enter your Kaggle username and password to download the dataset

## View Tensorboard Locally
    gcloud compute ssh [INSTANCE_NAME] -- -NfL 6006:localhost:6006
