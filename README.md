# Perseus
Inspired by Perseus’s clever and strategic approach to challenges, this repository meticulously analyzes information to expose the mastermind behind crypto market manipulation.


### Set up the repo

#### Create a python virtual environment

- macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r dev-requirements.txt
pip3 install -e .
export $(cat .env | xargs)
```

- Windows

```bash
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -r dev-requirements.txt
pip install -e .
export $(cat .env | xargs)
```

### For running 
Download the dataset COSS_data.pkl, DDINA_data.pkl, DDM_data.pkl, test_signals.pkl, train_signals.pkl, validate_signals.pkl.

Prepare Dataset

```
cd src/perseus/dataset
python dataset_preparation

```

Run experiments

```
cd src/perseus/evaluation
python run_fusion_experiments.py
```



