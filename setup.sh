# curl -sL \
#   "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" > \
#   "Miniconda3.sh"
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"
bash "Miniconda3.sh" -b
rm "Miniconda3.sh"
conda init bash
sudo chown 1002:1003 /opt/conda/pkgs/cache/*
conda env create -f environment.yml
conda update -n base conda