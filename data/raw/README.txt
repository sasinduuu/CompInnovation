Datasets go here.

1) malicious_phish.csv (base, Kaggle):
   https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
2) phiusiil_urls.csv.gz (extra, already provided) - slim copy of the UCI
   PhiUSIIL dataset: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

The processed features used by the model are in ../processed/final_dataset.csv.gz
Run 'python src/data_prep.py' to rebuild the full processed dataset (with URLs).
