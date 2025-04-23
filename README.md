
# Running CC12M dataset creation

pip install datasets termcolor


# need to change the number of samples generated here!
PYTHONPATH='.' python tld/sr_data_laion.py 

# update the configs here:
PYTHONPATH='.' python tld/configs_cc12m.py 

# train model:

# coco configs
PYTHONPATH='.' python tld/sr_train_tatitok.py  --config=configs

PYTHONPATH='.' python tld/sr_train_tatitok.py  --config=config_cc12m

