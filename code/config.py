# Number of Epochs
EPOCHS = 2

# batch size
batchsize = 64

# Fraction of data to be used for constructing training and test set            
frac = 0.002

# stride for generating onehot_seq_dict & chip-seq hdf5 file
stride = 50

# prediction region length
window_len = 400

# context around prediction region
context_window_len = 10_000

# stride used for chopping genome during training set construction
chop_genome_stride=50

# number of times to oversample chip-seq (positive) regions in training set
num_oversample_chip_seq_peaks = 5

# output path
out_path = "../example_output"

# input data path
in_path = "../example_input" #"../input/raw_data-Ascl1_12hr_real"

# experiment name (directory with this name will be create in out_path)
exp_name="experiment_name"

# training data directory name (directory with this name will be created in out_path/exp_name)
training_data_dir_name="training_data"

# train out directory name (directory with this name will be created in out_path/exp_name)
train_out_dir_name="train_out"

# test results directory path (directory with this name will be created in out_path/exp_name/train_out)
test_out_dir_name="test_set_performance"

# path to genome sizes file
info = f"{in_path}/mm10.info"

# path to genome fasta file
fa = f"{in_path}/mm10.fa"

# path to chip seq bigwig file in a list
bigwig_tracks_path_list =[
    f"{in_path}/Ascl1_R1_R2_R3_rep_avg.bw"
]

# path to MultiGPS .event file
peaks = f"{in_path}/multigps_2023-03-30-05-43-08_ES.events"

# path to blacklist regions .bed file
blacklist = f"{in_path}/mm10_blacklist.bed"

# list training chromosomes
training_chrom_list = ["chr1"]

# list of validation chromosomes
val_chroms = ["chr17"]

# list of test chromosomes
test_chroms = ["chr10"]

# path to cool .file
cool_file_path = f"{in_path}/GSE130275_mESC_WT_combined_1.3B_400_normalized.cool"

# resolution of .cool file
resolution = window_len

# model params (Note: Not all parameters are update using this dictionary. You may need to modify most model parameters inside the seq_and_bimodal_networks_ResNet.py)
model_params_dict = {"dense_layers": 3,
                     "n_filters": 240,
                     "filter_size": 20,
                     "pooling_size": 15,
                     "pooling_stride": 10,
                     "dropout": 0.5,
                     "dense_layer_size": 512,
                     "lstm_out": 32,
                     "dilation": 1,
                     "nnodes": (context_window_len+window_len)//resolution,
                     "out_features": 32
                    }

# dataloader threads
data_loader_num_workers = 4



# INTERNAL PARAMETERS (MODIFY ONLY IF REQUIRED): These are the default paths to various directories and files used during training and testing. You many not need to modify this

# number of 100 bp bins 
nbins = (context_window_len + window_len)//100

exp_path = f"{out_path}/{exp_name}"
data_path = f"{exp_path}/{training_data_dir_name}"
train_out_path = f"{exp_path}/{train_out_dir_name}"
test_out_path = f"{train_out_path}/{test_out_dir_name}"
seq_onehot_dict_path = f"{out_path}/common_data/onehot_seq_dict/onehot_seq_dict_res_{stride}.pickle"
hdf5_file_path = f"{out_path}/common_data/hdf5_chip_seq/hdf5_chip_seq_{stride}.h5"

train_data_paths_seq = {"bed_file_training_df": f"{data_path}/training_df_seq.bed"}
    
train_data_paths_bimodal = {"bed_file_bound_sample_all_df": f"{data_path}/training_df_bimodal_bound.bed",
                            "bed_file_unbound_genome_df": f"{data_path}/training_df_bimodal_unbound.bed"
                           }

val_data_path = f"{data_path}/data_val.bed"
internal_test_data_path = f"{data_path}/test_df_internal.bed"
external_test_data_path = f"{data_path}/test_df_external.bed"

config_file_path = "config.py"
