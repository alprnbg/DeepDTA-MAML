#!/bin/sh
set -e
export GPU_ID=0

echo $GPU_ID

cd ..
export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$GPU_ID


python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_0_test_cold_prot.csv --test_model 09-02-2021-split0-DeepdtaInit/saved_models/train_model_15
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_0_test_cold_both.csv --test_model 09-02-2021-split0-DeepdtaInit/saved_models/train_model_15
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_0_test_cold_lig.csv --test_model 09-02-2021-split0-DeepdtaInit/saved_models/train_model_15
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_0_test_cold_pfam.csv --test_model 09-02-2021-split0-DeepdtaInit/saved_models/train_model_15
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_0_test_warm.csv --test_model 09-02-2021-split0-DeepdtaInit/saved_models/train_model_15

python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_1_test_cold_prot.csv --test_model 09-02-2021-split1-DeepdtaInit/saved_models/train_model_29
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_1_test_cold_both.csv --test_model 09-02-2021-split1-DeepdtaInit/saved_models/train_model_29
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_1_test_cold_lig.csv --test_model 09-02-2021-split1-DeepdtaInit/saved_models/train_model_29
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_1_test_cold_pfam.csv --test_model 09-02-2021-split1-DeepdtaInit/saved_models/train_model_29
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_1_test_warm.csv --test_model 09-02-2021-split1-DeepdtaInit/saved_models/train_model_29

python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_2_test_cold_prot.csv --test_model 09-02-2021-split2-DeepdtaInit/saved_models/train_model_70
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_2_test_cold_both.csv --test_model 09-02-2021-split2-DeepdtaInit/saved_models/train_model_70
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_2_test_cold_lig.csv --test_model 09-02-2021-split2-DeepdtaInit/saved_models/train_model_70
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_2_test_cold_pfam.csv --test_model 09-02-2021-split2-DeepdtaInit/saved_models/train_model_70
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_2_test_warm.csv --test_model 09-02-2021-split2-DeepdtaInit/saved_models/train_model_70

# python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_3_test_cold_prot.csv --test_model 
# python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_3_test_cold_both.csv --test_model 
# python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_3_test_cold_lig.csv --test_model 
# python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_3_test_cold_pfam.csv --test_model 
# python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_3_test_warm.csv --test_model 

python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_4_test_cold_prot.csv --test_model 09-02-2021-split4-DeepdtaInit/saved_models/train_model_69
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_4_test_cold_both.csv --test_model 09-02-2021-split4-DeepdtaInit/saved_models/train_model_69
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_4_test_cold_lig.csv --test_model 09-02-2021-split4-DeepdtaInit/saved_models/train_model_69
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_4_test_cold_pfam.csv --test_model 09-02-2021-split4-DeepdtaInit/saved_models/train_model_69
python test_maml_system.py --name_of_args_json_file experiment_config/config_test.json --test_path ../data/bdb_pfam/bdb_pfam_4_test_warm.csv --test_model 09-02-2021-split4-DeepdtaInit/saved_models/train_model_69


