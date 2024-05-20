EPOCHS = 2
batchsize = 64


# Construct data                        
frac = 0.1
stride = 50
window_len = 400
context_window_len = 10_000
chop_genome_stride=50
num_oversample_chip_seq_peaks = 5
out_path = "../output"

in_path = "../input/raw_data-Ascl1_12hr_real"
exp_path = f"{out_path}/experiment_name"
data_path = f"{exp_path}/training_data"
train_out_path = f"{exp_path}/train_out"
test_out_path = f"{train_out_path}/test_set_performance"
seq_onehot_dict_out_path = f"{out_path}/common_data/onehot_seq_dict"

info = f"{in_path}/mm10_random_removed.info"
fa = f"{in_path}/mm10.fa"
acc_domains = f"{in_path}/ATAC_peaks.narrowPeak"

hdf5_chromatin_tracks_out_path = f"{out_path}/common_data/hdf5_Ascl1"
chromatin_tracks_path = f"{in_path}"
chromatin_tracks_path_list =[
    f"{chromatin_tracks_path}/Ascl1_R1_R2_R3_rep_avg.bw"
]

peaks = f"{in_path}/multigps_2023-03-30-05-43-08_ES.events"
nbins = (context_window_len + window_len)//100
blacklist = f"{in_path}/mm10_blacklist.bed"
training_chrom_list = ["chr1"]
val_chroms = ["chr17"]
test_chroms = ["chr10"]

# HiC & GAT
cool_file_path = f"{in_path}/GSE130275_mESC_WT_combined_1.3B_400_normalized.cool"
resolution = window_len

# model params
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

data_loader_num_workers = 4


train_data_paths_seq = {"bed_file_training_df": f"{data_path}/training_df_seq.bed"}
    
train_data_paths_bimodal = {"bed_file_bound_sample_all_df": f"{data_path}/training_df_bimodal_bound.bed",
                            "bed_file_unbound_genome_df": f"{data_path}/training_df_bimodal_unbound.bed"
                           }

val_data_path = f"{data_path}/data_val.bed"
internal_test_data_path = f"{data_path}/test_df_internal.bed"
external_test_data_path = f"{data_path}/test_df_external.bed"

config_file_path = "config_one_time_train.py"
