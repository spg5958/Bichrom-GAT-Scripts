import argparse
import yaml
from subprocess import call
from json import load
import numpy as np
import pandas as pd
import predict_on_external_data
import torch
import os
import utils
import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import pyfasta
import pyBigWig
from sklearn.metrics import roc_auc_score,average_precision_score
import seq_and_bimodal_networks_ResNet as seq_and_bimodal_networks
import config
import random
import inspect
import cooler
from tqdm import tqdm
import pickle
import h5py


bound_sample_all_id_list_seq=[]
unbound_all_id_list_seq=[]
bound_sample_all_id_list=[]
unbound_genome_id_list=[]
chopped_genome_id_list_seq=[]

print("Loading seq_onehot_dict.pickle")
onehot_seq_dict=None
print(config.seq_onehot_dict_path)
print(os.path.isfile(config.seq_onehot_dict_path))
with open(config.seq_onehot_dict_path, 'rb') as handle:
    onehot_seq_dict = pickle.load(handle)


class dataset_class_seq(Dataset):
    """
    Custom dataset class for reading seq-net training data
    """

    def __init__(self, bed_file):
        bed_file_name_prefix = bed_file.split("/")[-1].split(".")[0]
        print(bed_file_name_prefix)
        
        sizes = pd.read_csv(config.info, names=['chrom', 'chrsize'], sep='\t')
        
        
        self.data_df = pd.read_table(bed_file, header=None, names=["chrom", "start", "end", "type", "label", "id"])
        print(self.data_df)
        print(self.data_df.shape)
        
        self.data_df['end'] = self.data_df['end'] + config.context_window_len//2
        self.data_df['start'] = self.data_df['start'] - config.context_window_len//2
        chrom_sizes_dict = (dict(zip(sizes.chrom, sizes.chrsize)))
        self.data_df['chr_limits_upper'] = self.data_df['chrom'].map(chrom_sizes_dict)
        self.data_df = self.data_df[self.data_df['end'] <= self.data_df['chr_limits_upper']]
        self.data_df = self.data_df[self.data_df['start'] >= 0]
        self.data_df = self.data_df[['chrom', 'start', 'end', "label", "id"]]
        print(self.data_df)
        print(self.data_df.shape)
        print(self.data_df["label"].value_counts())
        
        self.h5 = h5py.File(config.hdf5_file_path, 'r')
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        item=self.data_df.iloc[idx,:]
        data_point=utils.get_datapoint_features_seq(item, onehot_seq_dict, config.stride, config.bigwig_tracks_path_list, self.h5, config.nbins)
        return data_point
 

class dataset_class_bimodal(Dataset):
    """
    Custom dataset class for reading GAT-net/Bimodal-net training data
    """

    def __init__(self, bed_file):
        bed_file_name_prefix = bed_file.split("/")[-1].split(".")[0]
        print(bed_file_name_prefix)
        
        sizes = pd.read_csv(config.info, names=['chrom', 'chrsize'], sep='\t')
        
        self.data_df = pd.read_table(bed_file, header=None, names=["chrom", "start", "end", "type", "label", "id"])
        self.data_df['end'] = self.data_df['end'] + config.context_window_len//2
        self.data_df['start'] = self.data_df['start'] - config.context_window_len//2
        chrom_sizes_dict = (dict(zip(sizes.chrom, sizes.chrsize)))
        self.data_df['chr_limits_upper'] = self.data_df['chrom'].map(chrom_sizes_dict)
        self.data_df = self.data_df[self.data_df['end'] <= self.data_df['chr_limits_upper']]
        self.data_df = self.data_df[self.data_df['start'] >= 0]
        self.data_df = self.data_df[['chrom', 'start', 'end', "label", "id"]]
        print(self.data_df.shape)
        print(self.data_df["label"].value_counts())
        
        self.cooler_obj=cooler.Cooler(config.cool_file_path)
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        item=self.data_df.iloc[idx,:]
        data_point=utils.get_datapoint_features_bimodal(item, self.cooler_obj, onehot_seq_dict, config.stride, config.window_len)
#         print("DONE")
        return data_point


def getTrainDatasetGenerator_seq(data_path):
    """
    Training batch generator for seq-net
    """

    bed_file_chopped_genome_df=data_path["bed_file_training_df"]
    
    train_dataset=dataset_class_seq(bed_file_chopped_genome_df)
    print(f"  train_dataset = {len(train_dataset)}")
    
    train_dataset_loader_seq = DataLoader(train_dataset, 
                                         batch_size=config.batchsize, 
                                         num_workers=config.data_loader_num_workers, 
                                         shuffle=True)

    return train_dataset_loader_seq


def getTrainDatasetGenerator_bimodal(data_path):
    """
    Training batch generator for GAT-net/Bimodal-net
    """

    bed_file_bound_sample_all_df=data_path["bed_file_bound_sample_all_df"]
    bed_file_unbound_genome_df=data_path["bed_file_unbound_genome_df"]
    
    bound_sample_all_dataset=dataset_class_bimodal(bed_file_bound_sample_all_df)
    unbound_genome_dataset=dataset_class_bimodal(bed_file_unbound_genome_df)
    print(f"  bound_sample_all_dataset data size = {len(bound_sample_all_dataset)}")
    print(f"  unbound_genome_dataset data size = {len(unbound_genome_dataset)}")
    bound_sample_all_dataset_loader = DataLoader(bound_sample_all_dataset, 
                                                 batch_size=config.batchsize//2, 
                                                 num_workers=config.data_loader_num_workers, 
                                                 shuffle=True)
    unbound_genome_dataset_loader = DataLoader(unbound_genome_dataset,
                                               batch_size=config.batchsize//2, 
                                               num_workers=config.data_loader_num_workers, 
                                               shuffle=True)

    unbound_genome_dataset_loader_iter=iter(unbound_genome_dataset_loader)
    return bound_sample_all_dataset_loader, unbound_genome_dataset_loader_iter

"""
def getValDatasetGenerator_seq(data_path):
    val_dataset=dataset_class_seq(data_path)
    print(f"  Val data size = {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=config.batchsize, num_workers=config.data_loader_num_workers, shuffle=True)
    return val_loader


def getValDatasetGenerator_bimodal(data_path):
    val_dataset=dataset_class_bimodal(data_path)
    print(f"  Val data size = {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=config.batchsize, num_workers=config.data_loader_num_workers, shuffle=True)
    return val_loader
"""

def train_one_epoch_seq(epoch, chopped_genome_dataset_loader, model, loss_fn, optimizer, scheduler, device):
    """
    Seq-net one epoch training function 
    """
    
    print("  Training (seq):")
    running_loss = 0.
    batch_avg_vloss = 0.

    print()
    print("  #### Num of Batches ####")
    print(f"  New chopped_genome_dataset_loader = {len(chopped_genome_dataset_loader)}")
    print()

    running_loss = 0.
    train_loss_batch_avg = 0.

    for i, batch in enumerate(chopped_genome_dataset_loader):
            
        # IDs
        chopped_genome_id_list_seq.extend(batch["id"].tolist())

        seq=batch["seq"]
        print(f"  {seq.size()}")
        bigwig_track_dict={}
        for bigwig_track_path in config.bigwig_tracks_path_list:
            bigwig_track_name = bigwig_track_path.split("/")[-1].split(".")[0]
            bigwig_track_dict[bigwig_track_name]=batch[bigwig_track_name]
            print(f"  {bigwig_track_dict[bigwig_track_name].size()}")
        
        seq=seq.type(torch.float)
        seq=torch.permute(seq,(0,2,1))
        
        # transfer data to GPU
        seq = seq.to(device)
        for bigwig_track_path in config.bigwig_tracks_path_list:
            bigwig_track_name = bigwig_track_path.split("/")[-1].split(".")[0]
            bigwig_track_dict[bigwig_track_name] = bigwig_track_dict[bigwig_track_name].to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        seq_out = model(seq)

        loss = 0
        loss_avg = 0
        loss_zero = 0
        loss_ideal = 0
        for j,bigwig_track_path in enumerate(config.bigwig_tracks_path_list):
            bigwig_track_name = bigwig_track_path.split("/")[-1].split(".")[0]
            true_value = bigwig_track_dict[bigwig_track_name].type(torch.float)
            seq_out[j] = seq_out[j].type(torch.float)
            
            _seq_out = seq_out[j].cpu().detach().numpy()
            _true_value = true_value.cpu().detach().numpy()
            print("pred",
                  round(np.mean(_seq_out),4),
                  round(np.min(_seq_out),4),
                  round(np.max(_seq_out),4),
                  round(np.quantile(_seq_out,0.9),4)
                 )
            print("true",
                  round(np.mean(_true_value),4),
                  round(np.min(_true_value),4),
                  round(np.max(_true_value),4),
                  round(np.quantile(_true_value,0.9),4)
                 )
            loss += loss_fn(seq_out[j], true_value)
            loss_avg += loss_fn(torch.mean(true_value), true_value)
            loss_zero += loss_fn(torch.zeros_like(seq_out[j]), true_value)
            loss_ideal += loss_fn(true_value, true_value)
            
        factor = len(config.bigwig_tracks_path_list)
        loss *= (1/factor)
        loss_avg *= (1/factor)
        loss_zero *= (1/factor)
        loss_ideal *= (1/factor)
        
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        batch_avg_loss = running_loss / (i + 1) # loss per batch
        print('  batch {} batch_avg_loss: {}'.format(i + 1, batch_avg_loss))
        print('  batch {} loss: {}'.format(i + 1, loss))
        print('  batch {} loss_avg: {}'.format(i + 1, loss_avg))
        print('  batch {} loss_zero: {}'.format(i + 1, loss_zero))
        print('  batch {} loss_ideal: {}'.format(i + 1, loss_ideal))
        
        # if i>10:
        #     break
            
    return batch_avg_loss


def train_one_epoch_bimodal(epoch, bound_sample_all_dataset_loader, unbound_genome_dataset_loader_iter, model, loss_fn, optimizer, device):
    """
    GAT-net/Bimodal-net one epoch training function
    """

    print("  Training (bimodal):")
    
    running_loss = 0.
    train_loss_batch_avg = 0.

    for i, (bound_sample_all_batch, unbound_batch) in enumerate(zip(bound_sample_all_dataset_loader, unbound_genome_dataset_loader_iter)):
        
        # IDs
        bound_sample_all_id_list.extend(bound_sample_all_batch["id"].tolist())
        unbound_genome_id_list.extend(unbound_batch["id"].tolist())
        
        print(f"\nBatch {i} (bimodal)")
        print(f"  {bound_sample_all_batch['seq'].size(),unbound_batch['seq'].size()}")
        print(f"  {bound_sample_all_batch['adj'].size(),unbound_batch['adj'].size()}")
        print(f"  {len(bound_sample_all_batch['label']),len(unbound_batch['label'])}")
        
        if bound_sample_all_batch["seq"].size()[0] != unbound_batch["seq"].size()[0]:
            print("  Bound & Unbound sizes are not equal")
            idx=torch.multinomial(torch.ones(unbound_batch["seq"].size()[0]), bound_sample_all_batch["seq"].size()[0], replacement=False) 
            unbound_batch["seq"]=unbound_batch["seq"][idx]
            unbound_batch["adj"]=unbound_batch["adj"][idx]
            unbound_batch["label"]=unbound_batch["label"][idx]
            print(f"  {bound_sample_all_batch['seq'].size(),unbound_batch['seq'].size()}")
            print(f"  {bound_sample_all_batch['adj'].size(),unbound_batch['adj'].size()}")
            print(f"  {len(bound_sample_all_batch['label']),len(unbound_batch['label'])}")
        
        seq_cat=torch.cat((bound_sample_all_batch["seq"], 
                           unbound_batch["seq"]), 0)
        adj_cat = torch.cat((bound_sample_all_batch["adj"], 
                             unbound_batch["adj"]), 0)
        label_cat=torch.cat((bound_sample_all_batch["label"],
                             unbound_batch["label"]), 0)
        indices = torch.randperm(seq_cat.size()[0])
        seq_cat=seq_cat[indices]
        adj_cat=adj_cat[indices]
        label_cat=label_cat[indices]
        
        seq=seq_cat.type(torch.float)
        seq=torch.permute(seq,(0,2,1))
        adj=adj_cat.type(torch.float)
        labels=label_cat.type(torch.float)
        labels=torch.reshape(labels,(len(labels),1))

        print(f"  {seq.size()}")
        print(f"  {adj.size()}")
        print(f"  {labels.size()}")
        
        # transfer data to GPU
        seq, adj, labels = seq.to(device), adj.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(seq,adj)
        
        # Compute the loss and its gradients
#         loss = loss_fn(labels, outputs) # mistake
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        train_loss_batch_avg = running_loss / (i+1) # loss per batch
        print(f'  batch {i} batch_avg_train_loss: {train_loss_batch_avg}')
        
        # if i>10:
        #     break

    return train_loss_batch_avg


def build_and_train_seq(data_path, val_data_path, out_path):
    """
    Main function for building & training seq-net.
    Save model at each epoch & identify best performing model based on training loss.
    """
    
    # Create an output directory for saving models + per-epoch logs.
    out_path_seq = f"{out_path}/seqnet"
    call(['mkdir', "-p", out_path_seq])
    
    hist_dict={"epoch":[],"avg_train_loss":[]}
    
    # GPU
    device = utils.getDevice()
    
    # Model
    model=seq_and_bimodal_networks.seq_network(config.model_params_dict, config.bigwig_tracks_path_list)
    print(model)
    
    model = nn.DataParallel(model)
    
    # transfer model to GPU
    model.to(device)

    train_dataset_loader_seq = getTrainDatasetGenerator_seq(data_path)
    
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-6)   # 0.02

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(config.EPOCHS):
        
        print(f"\nEPOCH {epoch}/{config.EPOCHS-1}")
   
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_train_loss=train_one_epoch_seq(epoch,
                                           train_dataset_loader_seq,
                                           model,
                                           loss_fn,
                                           optimizer,
                                           scheduler,
                                           device)

        # Adjust learning rate
        scheduler.step()
        
        # We don't need gradients on to do reporting
        model.train(False)
        model.eval()
        
        print(f'  LOSS train = {avg_train_loss}')
        torch.save(model.module.state_dict(), f"{out_path_seq}/seq_model_epoch_{epoch}_train_loss_{round(avg_train_loss,2)}.pt")  
        hist_dict["epoch"].append(epoch)
        hist_dict["avg_train_loss"].append(avg_train_loss)

    hist_df=pd.DataFrame.from_dict(hist_dict)
    hist_df.to_csv(f"{out_path_seq}/train_hist_seq.csv",index=False)
    print("  Finding best model:")
    best_model_idx = np.argmin(hist_df["avg_train_loss"])
    best_model_train_loss=round(hist_df.iloc[best_model_idx]["avg_train_loss"],2)
    print(f"  Seq best model = {best_model_idx} loss = {best_model_train_loss}")
    call(["cp",f"{out_path_seq}/seq_model_epoch_{best_model_idx}_train_loss_{best_model_train_loss}.pt",f"{out_path_seq}/best_seq_model_epoch_{best_model_idx}_train_loss_{best_model_train_loss}.pt"])


def build_and_train_bimodal(data_path, val_data_path, base_seq_model_path, out_path):
    """
    Main function for building & training GAT-net/Bimodal-net.
    Save model at each epoch & identify best performing model based on training loss.
    """

    # Create an output directory for saving models + per-epoch logs.
    out_path_bimodal = f"{out_path}/bimodal"
    call(['mkdir', out_path_bimodal])
    
    hist_dict={"epoch":[],"train_loss_batch_avg":[]}

    # GPU
    device = utils.getDevice()
    
    # Model
    base_model = seq_and_bimodal_networks.seq_network(config.model_params_dict, config.bigwig_tracks_path_list)
    
    print("#"*20)
    print(base_seq_model_path)
    print("#"*20)
    base_model.load_state_dict(torch.load(base_seq_model_path))

    for param in base_model.parameters():
        param.requires_grad = False
        
    bimodal_model=seq_and_bimodal_networks.bimodal_network_GAT(config.model_params_dict, base_model)
    
    print(bimodal_model)
    
    bimodal_model = nn.DataParallel(bimodal_model)
    
    # transfer model to GPU
    bimodal_model.to(device)
    
    bound_sample_all_dataset_loader, unbound_genome_dataset_loader_iter = getTrainDatasetGenerator_bimodal(data_path)
    
    loss_fn = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(bimodal_model.parameters(), lr=0.01, weight_decay=1e-6)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(config.EPOCHS):
        print(f"\nEPOCH {epoch}/{config.EPOCHS-1}")

        # Make sure gradient tracking is on, and do a pass over the data               
        bimodal_model.train(True)
        train_loss_batch_avg=train_one_epoch_bimodal(epoch, bound_sample_all_dataset_loader, unbound_genome_dataset_loader_iter, bimodal_model, loss_fn, optimizer,device)

        # We don't need gradients on to do reporting
        bimodal_model.train(False)

        print(f'  LOSS train = {train_loss_batch_avg}')
        
        torch.save(bimodal_model.module.state_dict(), f"{out_path_bimodal}/bimodal_model_epoch_{epoch}_train_loss_batch_avg_{round(train_loss_batch_avg,2)}.pt")  
        
        hist_dict["epoch"].append(epoch)
        hist_dict["train_loss_batch_avg"].append(train_loss_batch_avg)
        
    hist_df=pd.DataFrame.from_dict(hist_dict)
    hist_df.to_csv(f"{out_path_bimodal}/train_hist_bimodal.csv",index=False)
    print("  Finding best model:")
    best_model_idx = np.argmin(hist_df["train_loss_batch_avg"])
    best_model_train_loss=round(hist_df.iloc[best_model_idx]["train_loss_batch_avg"],2)
    print(f"  Bimodal best model = {best_model_idx} best_model_train_loss = {best_model_train_loss}")
    call(["cp",f"{out_path_bimodal}/bimodal_model_epoch_{best_model_idx}_train_loss_batch_avg_{best_model_train_loss}.pt",f"{out_path_bimodal}/best_bimodal_model_epoch_{best_model_idx}_train_loss_batch_avg_{best_model_train_loss}.pt"])


def train_bichrom(train_data_paths_seq, train_data_paths_bimodal, val_data_path, train_out_path):
    """
    This function that calls individual seq-net and GAT-net/Bimodal-net training functions.
    It also call function that predicts test-set performance of GAT-net/Bimodal-net.
    """
    
    # create the output directory:
    call(['mkdir', "-p", train_out_path])
    
    call(["cp", config.config_file_path, train_out_path])
     
    print(inspect.getsource(config))
    print()
    print(f"--> train_data_paths_seq = {train_data_paths_seq}")
    print(f"--> train_data_paths_bimodal = {train_data_paths_bimodal}")
    
    #"""
    # Train the sequence-only network (M-SEQ)
    print("\n--> Training seq")
    build_and_train_seq(train_data_paths_seq, val_data_path, train_out_path)

    # Find best seq model
    best_seq_model_path=None
    print("Finding best seq model....")
    for file in os.listdir(f"{config.train_out_path}/seqnet"):
        if "best" in file.lower():
            best_seq_model_path=f"{config.train_out_path}/seqnet/{file}"
            print(f"Best seq model found = {best_seq_model_path}")
            break
    if best_seq_model_path is None:
        print("Cannot find best seq model")
    #"""
    
    #best_seq_model_path = f"{train_out_path}/seqnet/best_seq_model_epoch_1_val_loss_0.29.pt"
    
    #"""
    # Train the bimodal network (M-SC)
    print("\n--> Training bimodal")
    build_and_train_bimodal(train_data_paths_bimodal, val_data_path, best_seq_model_path, train_out_path)
    
    # Find best bimodal model
    best_bimodal_model_path=None
    print("Finding best bimodal model....")
    for file in os.listdir(f"{config.train_out_path}/bimodal"):
        if "best" in file.lower():
            best_bimodal_model_path=f"{config.train_out_path}/bimodal/{file}"
            print(f"Best bimodal model found = {best_bimodal_model_path}")
            break
    if best_bimodal_model_path is None:
        print("Cannot find best bimodal model")
    #""" 
    
    #best_bimodal_model_path=f"{train_out_path}/bimodal/best_bimodal_model_epoch_1_train_loss_batch_avg_0.59.pt"
    
    # Save IDs
    
    call(["mkdir", "-p", f"{config.train_out_path}/ids"])
    with open(f"{config.train_out_path}/ids/bound_sample_all_id_list_seq.txt", "w") as fh:
        np.savetxt(fh, bound_sample_all_id_list_seq)
    with open(f"{config.train_out_path}/ids/unbound_all_id_list_seq.txt", "w") as fh:
        np.savetxt(fh, unbound_all_id_list_seq)
    with open(f"{config.train_out_path}/ids/bound_sample_all_id_list.txt", "w") as fh:
        np.savetxt(fh, bound_sample_all_id_list)
    with open(f"{config.train_out_path}/ids/unbound_genome_id_list.txt", "w") as fh:
        np.savetxt(fh, unbound_genome_id_list)         

    #"""
    predict_on_external_data.evaluate_models(config.internal_test_data_path,
                                             config.fa, 
                                             config.info, 
                                             config.cool_file_path,
                                             best_seq_model_path,
                                             best_bimodal_model_path,
                                             config.config_file_path,
                                             config.batchsize,
                                             config.data_loader_num_workers,
                                             f"{config.test_out_path}/internal_test_set_performance",
                                             onehot_seq_dict,
                                             config.context_window_len
                                            )

    
    predict_on_external_data.evaluate_models(config.external_test_data_path,
                                             config.fa, 
                                             config.info, 
                                             config.cool_file_path,
                                             best_seq_model_path,
                                             best_bimodal_model_path,
                                             config.config_file_path,
                                             config.batchsize,
                                             config.data_loader_num_workers,
                                             f"{config.test_out_path}/external_test_set_performance",
                                             onehot_seq_dict,
                                             config.context_window_len
                                            )
    #"""
    
# MAIN
print(f"Total Epochs = {config.EPOCHS}")
train_bichrom(config.train_data_paths_seq, config.train_data_paths_bimodal, config.val_data_path, config.train_out_path)
