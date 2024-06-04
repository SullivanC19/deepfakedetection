curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" > \
  "Miniconda3.sh"
# curl -sL \
#   "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
#   "Miniconda3.sh"
bash "Miniconda3.sh" -b
rm "Miniconda3.sh"
conda update -y -n base -c conda-forge conda
conda env create -y -f environment.yml
source activate deepfakesonly
python main.py