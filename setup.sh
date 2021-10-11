apt install -y libsndfile1-dev libopenblas-base
conda install -y jupyter ruamel.yaml

# conda environment for Montreal-Forced-Aligner
# MFA should run in a separate venv
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p miniconda

# MFA
source miniconda/bin/activate
conda install -y -c conda-forge openblas openfst pynini ngram baumwelch
pip install montreal-forced-aligner
mfa thirdparty download
wget https://github.com/MontrealCorpusTools/mfa-models/raw/main/g2p/english_g2p.zip
wget https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip
conda deactivate

pip install -r requirements.txt

