import numpy as np
import pandas as pd
import utils
import config
import os
import pickle
from torch.utils.data import Dataset, DataLoader
import h5py
import seq_and_bimodal_networks_ResNet as seq_and_bimodal_networks
import config
import torch
import pyBigWig
from subprocess import call

"""
This script predicts chip-seq bigwig track using a trained seq-net for a given chromosome.
If you need tracks for multiple chromosomes, you will need to run it on each chromosome separately.
"""

# chromosome for which to predict bigwig track
_chr="chr10"

# trained seq-net model path
seq_model_path=f"{config.train_out_path}/seqnet/best_seq_model_epoch_1_train_loss_0.32.pt"

# output directory path
out_path=f"{config.exp_path}/predict_chip_seq_track_from_seqnet"

# window length of each sample. This shold be equal to total window length that seq-net was train on (prediction window length + context window length)
chop_genome_window_size=config.window_len+config.context_window_len

class test_dataset_class_seq(Dataset):
    """
    Custom dataset class for reading seq-net test data
    """

    def __init__(self, bed_df, nbins):
        sizes = pd.read_csv(config.info, names=['chrom', 'chrsize'], sep='\t')
        self.data_df=bed_df.copy()
        chrom_sizes_dict = (dict(zip(sizes.chrom, sizes.chrsize)))
        self.data_df['chr_limits_upper'] = self.data_df['chrom'].map(chrom_sizes_dict)
        self.data_df = self.data_df[self.data_df['end'] <= self.data_df['chr_limits_upper']]
        self.data_df = self.data_df[self.data_df['start'] >= 0]
        self.data_df = self.data_df[['chrom', 'start', 'end', "label", "id"]]
        print(self.data_df.shape)
        print(self.data_df["label"].value_counts())
        
        self.h5 = h5py.File(config.hdf5_file_path, 'r')
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        item=self.data_df.iloc[idx,:]
        data_point=utils.get_datapoint_features_test_seq(item, onehot_seq_dict, config.stride, config.bigwig_tracks_path_list, self.h5, nbins)
        return data_point

def getTestDatasetGenerator_seq(bed_df, nbins):
    """
    Test batch generator for seq-net
    """

    test_dataset_seq=test_dataset_class_seq(bed_df, nbins)
    print(f"  Data size = {len(test_dataset_seq)}")
    test_dataloader_seq = DataLoader(test_dataset_seq, batch_size=config.batchsize, num_workers=config.data_loader_num_workers, shuffle=False)
    return test_dataloader_seq


# MAIN

# Initial setup and data loading
print("Loading seq_onehot_dict.pickle")
onehot_seq_dict=None
print(config.seq_onehot_dict_path)
print(os.path.isfile(config.seq_onehot_dict_path))
with open(config.seq_onehot_dict_path, 'rb') as handle:
    onehot_seq_dict = pickle.load(handle)

seq_data_df = utils.chop_genome(config.info, [_chr], None, stride=chop_genome_window_size, l=chop_genome_window_size)
seq_data_df["label"]=np.nan
seq_data_df["id"]=list(np.arange(len(seq_data_df)))

print(seq_data_df.shape)
print(seq_data_df.head())

call(["mkdir","-p",out_path])

# GPU
device = utils.getDevice()
print(device)

model_seq = seq_and_bimodal_networks.seq_network(config.model_params_dict, config.bigwig_tracks_path_list)
model_seq.load_state_dict(torch.load(seq_model_path,map_location=device))
# model_seq.load_state_dict(torch.load(seq_model_path,map_location=torch.device('cpu')))
model_seq.to(device)
model_seq.train(False)
model_seq.eval()

h5 = h5py.File(config.hdf5_file_path, 'r')

nbins = chop_genome_window_size//100
true_bigwig_tracks={}
pred_bigwig_tracks={}
for bigwig_track_path in config.bigwig_tracks_path_list:
    bigwig_track_name = bigwig_track_path.split("/")[-1].split(".")[0]
    true_bigwig_tracks[bigwig_track_name]={"chroms":[],"starts":[],"ends":[],"values":[]}
    pred_bigwig_tracks[bigwig_track_name]={"chroms":[],"starts":[],"ends":[],"values":[]}

test_dataset_loader_seq = getTestDatasetGenerator_seq(seq_data_df, nbins)

# Iterate over batches and predict bigwig track values for each
for batch_idx, test_batch in enumerate(test_dataset_loader_seq):

    seq = test_batch["seq"].to(device)
    seq=torch.permute(seq,(0,2,1))
    seq = seq.to(torch.float)
    output = model_seq(seq)

    _batchsize=test_batch["seq"].shape[0]
    print(_batchsize)
    for i in range(_batchsize):
        # print(i)
        start=test_batch["start"][i]
        end=test_batch["end"][i]
        start_array=np.arange(start,end,100)
        end_array=start_array+100
        start_list=start_array.tolist()
        end_list=end_array.tolist()
        chr_list=[_chr]*len(start_array)
        for track_idx,bigwig_track_path in enumerate(config.bigwig_tracks_path_list):
            bigwig_track_name = bigwig_track_path.split("/")[-1].split(".")[0]
            true_value=test_batch[bigwig_track_name][i].tolist()
            pred=output[track_idx][i].tolist()

            true_bigwig_tracks[bigwig_track_name]["chroms"].extend(chr_list)
            true_bigwig_tracks[bigwig_track_name]["starts"].extend(start_list)
            true_bigwig_tracks[bigwig_track_name]["ends"].extend(end_list)
            true_bigwig_tracks[bigwig_track_name]["values"].extend(true_value)
            pred_bigwig_tracks[bigwig_track_name]["chroms"].extend(chr_list)
            pred_bigwig_tracks[bigwig_track_name]["starts"].extend(start_list)
            pred_bigwig_tracks[bigwig_track_name]["ends"].extend(end_list)
            pred_bigwig_tracks[bigwig_track_name]["values"].extend(pred)

df=pd.read_csv(config.info,header=None,sep="\t")
chr_sizes_list=[]
for i,row in df.iterrows():
    chr_sizes_list.append((row[0],row[1]))

# Create predicted and true bigwig track files
for bigwig_track_path in config.bigwig_tracks_path_list:
    bigwig_track_name = bigwig_track_path.split("/")[-1].split(".")[0]
    bw = pyBigWig.open(f"{out_path}/true_{bigwig_track_name}.bw","w")
    bw.addHeader(chr_sizes_list)
    bw.addEntries(true_bigwig_tracks[bigwig_track_name]["chroms"], 
                  true_bigwig_tracks[bigwig_track_name]["starts"], 
                  ends=true_bigwig_tracks[bigwig_track_name]["ends"], 
                  values=true_bigwig_tracks[bigwig_track_name]["values"])
    bw.close()

    bw = pyBigWig.open(f"{out_path}/pred_{bigwig_track_name}.bw","w")
    bw.addHeader(chr_sizes_list)
    bw.addEntries(pred_bigwig_tracks[bigwig_track_name]["chroms"],
                  pred_bigwig_tracks[bigwig_track_name]["starts"],
                  ends=pred_bigwig_tracks[bigwig_track_name]["ends"], 
                  values=pred_bigwig_tracks[bigwig_track_name]["values"])
    bw.close()
