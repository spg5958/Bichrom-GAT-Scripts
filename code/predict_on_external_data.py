from __future__ import division
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
from functools import partial
from torch.utils.data import Dataset, DataLoader
import pyfasta
import pyBigWig
import torch
from scipy.stats import beta
from sklearn.metrics import roc_curve
import seaborn as sns
sns.set_style('whitegrid')
from subprocess import call
import os
import seq_and_bimodal_networks_ResNet as seq_and_bimodal_networks
import importlib.util 
import inspect
import cooler
import config

_config=None
_onehot_seq_dict = None


class dataset_class_bimodal(Dataset):
    """
    Custom dataset class for reading GAT-net/Bimodal-net training data
    """

    def __init__(self, bed_file, info, cool_file_path, context_window_len):
        bed_file_name_prefix = bed_file.split("/")[-1].split(".")[0]
        print(bed_file_name_prefix)
        
        sizes = pd.read_csv(info, names=['chrom', 'chrsize'], sep='\t')
        
        self.data_df = pd.read_table(bed_file, header=None, names=["chrom", "start", "end", "type", "label", "id"])
        self.data_df['end'] = self.data_df['end'] + context_window_len//2
        self.data_df['start'] = self.data_df['start'] - context_window_len//2
        chrom_sizes_dict = (dict(zip(sizes.chrom, sizes.chrsize)))
        self.data_df['chr_limits_upper'] = self.data_df['chrom'].map(chrom_sizes_dict)
        self.data_df = self.data_df[self.data_df['end'] <= self.data_df['chr_limits_upper']]
        self.data_df = self.data_df[self.data_df['start'] >= 0]
        self.data_df = self.data_df[['chrom', 'start', 'end', "label", "id"]]
        print(self.data_df.shape)
        print(self.data_df["label"].value_counts())
        
        self.cooler_obj=cooler.Cooler(cool_file_path)
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        item=self.data_df.iloc[idx,:]
        data_point=utils.get_datapoint_features_bimodal(item, self.cooler_obj, _onehot_seq_dict, config.stride, config.window_len)
        print("DONE")
        return data_point

    
def getTestDatasetGenerator_bimodal(bed_file_path, fa_path, info, cool_file_path, batchsize, data_loader_num_workers, context_window_len):
    """
    Training batch generator for GAT-net/Bimodal-net
    """
    
    _test_dataset_bimodal=dataset_class_bimodal(bed_file_path, info, cool_file_path, context_window_len)
    print(f"  Data size = {len(_test_dataset_bimodal)}")
    test_dataloader = DataLoader(_test_dataset_bimodal, batch_size=batchsize, num_workers=data_loader_num_workers, shuffle=True)
    return test_dataloader


def get_probabilities(bed_file_path, fa_path, info, cool_file_path, model, batchsize, data_loader_num_workers, outfile, mode, context_window_len):
    """
    Predict binding probabilities for test-set using trained GAT-net/Bimodal-net
    """

    test_dataset=getTestDatasetGenerator_bimodal(bed_file_path, fa_path, info, cool_file_path, batchsize, data_loader_num_workers, context_window_len)

    model.train(False)
    probas_list=[]
    true_labels_list=[]
    for i,batch in enumerate(test_dataset):
        print(f"Batch = {i}")
        seq=batch["seq"].type(torch.float)
        seq=torch.permute(seq,(0,2,1))
        adj=batch["adj"].type(torch.float)
        labels=batch["label"].type(torch.float)
        labels=torch.reshape(labels,(len(labels),1))
        if mode=="seq":
            chromatin_outputs=model(seq).detach().numpy()
        elif mode=="bimodal":
            p=model(seq,adj).cpu().detach().numpy()
        probas_list.append(p)
        true_labels_list.append(labels)

    probas=np.concatenate(probas_list)
    true_labels = np.concatenate(true_labels_list)
    
    # erase the contents in outfile
    file = open(outfile, "w")
    file.close()
    # saving to file: 
    with open(outfile, "a") as fh:
        np.savetxt(fh, probas)

    return true_labels, probas


def get_metrics(test_labels, test_probas, out_file, model_name):
    """
    Get various performance metrics such as AUC ROC, AUC PRC, Confusion matrix from
    predicted probabilities and true class labels.
    Save output in a out_file.
    """

    # Calculate auROC
    auc_roc = sklearn.metrics.roc_auc_score(test_labels, test_probas)
    # Calculate auPRC
    auc_prc = sklearn.metrics.average_precision_score(test_labels, test_probas)
    pred_labels = np.where(test_probas.flatten() > 0.5, 1, 0)
    cm = sklearn.metrics.confusion_matrix(test_labels, pred_labels)
    # Write auROC and auPRC to records file.
    out_file.write("Model:{0}\n".format(model_name))
    out_file.write("AUC ROC:{0}\n".format(auc_roc))
    out_file.write("AUC PRC:{0}\n".format(auc_prc))
    out_file.write("\n")
    out_file.write("CONFUSION MATRIX:\n{0}\n".format(cm))
    out_file.write(f"Total points = {len(test_labels)}\n")
    out_file.write(f"Value counts True = \n{pd.Series(test_labels.flatten()).astype(int).value_counts().to_frame()}\n")
    out_file.write(f"Value counts Pred = \n{pd.Series(pred_labels).value_counts().to_frame()}")
    out_file.write("="*20)
    out_file.write("\n\n")
    return auc_roc, auc_prc

                
def evaluate_models(bed_file_path, fa_path, info, cool_file_path, model_seq_path, model_sc_path, config_file_path, batchsize, data_loader_num_workers, out_path, onehot_seq_dict, context_window_len):
    """
    Main function that can be used to evaluate performance of GAT-net/Bimodal-net on a test-set
    """

    global _config
    global _onehot_seq_dict
    _onehot_seq_dict = onehot_seq_dict
#     device = utils.getDevice()
    
    print("\n--> Predict test-set performance of best models")
    print(f"  {bed_file_path}")
    
    spec = importlib.util.spec_from_file_location("module.name", config_file_path)
    _config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_config)
    print(inspect.getsource(_config))
    print(_config.model_params_dict)
    
    print(f"model_seq_path = {model_seq_path}")
    print(f"model_sc_path = {model_sc_path}")
    # Find best seq model
    
    print("#"*20)
    print(model_seq_path)
    print("#"*20)
    
    model_seq = seq_and_bimodal_networks.seq_network(_config.model_params_dict, _config.bigwig_tracks_path_list)
    model_seq.load_state_dict(torch.load(model_seq_path))
    model_seq.eval()
    
    bimodal_model=seq_and_bimodal_networks.bimodal_network_GAT(_config.model_params_dict, model_seq)
    
    
    bimodal_model.load_state_dict(torch.load(model_sc_path))
    bimodal_model.eval()
        
#     model_seq.to(device)
#     bimodal_model.to(device)
    
    call(["mkdir", "-p", out_path])
    
    probas_file_out_path_seq=f"{out_path}/test_set_probs_seq.txt"
    probas_file_out_path_bimodal=f"{out_path}/test_set_probs_bimodal.txt"
    metrix_file = open(f"{out_path}/test_set_metrics.txt", "w")

    true_labels_bimodal, probas_sc = get_probabilities(bed_file_path, 
                                                       fa_path, 
                                                       info,
                                                       cool_file_path,
                                                       bimodal_model,
                                                       batchsize,
                                                       data_loader_num_workers,
                                                       probas_file_out_path_bimodal,
                                                       mode="bimodal",
                                                       context_window_len=context_window_len
                                                      )

    # Get the auROC and the auPRC for both M-SEQ and M-SC models:
    auc_roc_bimodal, auc_prc_bimodal = get_metrics(true_labels_bimodal, probas_sc, metrix_file, 'BIMODAL')
    auc_roc_bimodal, auc_prc_bimodal = get_metrics(true_labels_bimodal, np.zeros_like(true_labels_bimodal), metrix_file, 'BIMODAL_ZEROS')
    auc_roc_bimodal, auc_prc_bimodal = get_metrics(true_labels_bimodal, np.ones_like(true_labels_bimodal), metrix_file, 'BIMODAL_ONES')
    metrix_file.close()

    
    metric_dict={"model":["bimodal"],
                 "auc_roc":[auc_roc_bimodal],
                 "auc_prc":[auc_prc_bimodal]
                }
    
    metric_df=pd.DataFrame.from_dict(metric_dict)
    metric_df.to_csv(f"{out_path}/test_set_metrics.csv",index=False)
