import argparse
import yaml
import subprocess
import numpy as np
import pandas as pd
from pybedtools import BedTool
import utils
import os
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
import config
import pickle
from tqdm import tqdm
import pyfasta
import glob
import pyBigWig
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def construct_training_set_seq(genome_sizes_file, genome_fasta_file, peaks_file, blacklist_file, to_keep, to_filter, window_length, out_path, nbins, stride=5000):

    # # prepare files for defining coordiantes
    # curr_genome_bdt = utils.get_genome_sizes(genome_sizes_file, to_keep=to_keep, to_filter=to_filter)

    chip_seq_coordinates = utils.load_chipseq_data(peaks_file, genome_sizes_file=genome_sizes_file, to_keep=to_keep, to_filter=to_filter)

    print(f"chip_seq_coordinates = {chip_seq_coordinates.shape}")
    
    blacklist_bdt = BedTool(blacklist_file)

    bound_chip_peaks_bdt = BedTool.from_dataframe(chip_seq_coordinates).intersect(blacklist_bdt, v=True)

    print("Chopping genome")
    chopped_genome_df = utils.chop_genome(genome_sizes_file, to_keep, excl=blacklist_bdt, stride=config.chop_genome_stride, l=config.window_len)
    chopped_genome_bdt = BedTool.from_dataframe(chopped_genome_df)
    print(f"Chopping genome = {chopped_genome_df.count()}")
    
    bound_bdt_obj = chopped_genome_bdt.intersect(bound_chip_peaks_bdt, u=True)
    bound_bdt_obj = bound_bdt_obj.intersect(blacklist_bdt, v=True)
    
    bound_df = bound_bdt_obj.to_dataframe().assign(type="pos", label=1)
    print(f"bound_df = {bound_df.shape}")
    
    if config.num_oversample_chip_seq_peaks > 0:
        bound_oversample_df_list=[bound_df]
        for i in range(config.num_oversample_chip_seq_peaks):
            print(f"Oversampling {i}")
            _bound_oversample_df=utils.make_random_shift(bound_df, L=config.window_len)
            bound_oversample_df_list.append(_bound_oversample_df)
        bound_df=pd.concat(bound_oversample_df_list, axis=0)
        print(f"bound_df (after oversample) = {bound_df.shape}")
    else:
        print("Skipping oversampling....")
        
        
    print("getting unbound genome")
    unbound_bdt_obj = chopped_genome_bdt.intersect(bound_chip_peaks_bdt, v=True)
    unbound_df = unbound_bdt_obj.to_dataframe().assign(type="neg_genome", label=0)
    print(f"unbound_all_df = {unbound_df.shape}")
    

    training_df_seq=pd.concat([bound_df,unbound_df], axis=0)

    training_df_seq["id"]=list(range(training_df_seq.shape[0]))
    print(training_df_seq)
    print(f"training_df shape = {training_df_seq.shape}")
    
    print("#"*20)
    print(f"training_df_seq shape before sample = {training_df_seq.shape}")
    training_df_seq=training_df_seq.sample(frac=config.frac, replace=False)
    print(f"training_df_seq shape after sample = {training_df_seq.shape}")
    print("#"*20)
    print(training_df_seq)
    
    
    training_df_seq.to_csv(f"{out_path}/training_df_seq.bed", header=False, index=False, sep="\t")
    
    return f"{out_path}/training_df_seq.bed"
    
    
    
    #return f"{out_path}/bound_sample_acc_df.bed", f"{out_path}/bound_sample_inacc_df.bed", f"{out_path}/bound_sample_all_df.bed", f"{out_path}/unbound_acc_df.bed", f"{out_path}/unbound_inacc_df.bed", f"{out_path}/unbound_all_df.bed"
    
    

def construct_training_set_bimodal(genome_sizes_file, genome_fasta_file, peaks_file, blacklist_file, to_keep, to_filter, window_length, out_path, nbins, stride=5000):

    # # prepare files for defining coordiantes
    # curr_genome_bdt = utils.get_genome_sizes(genome_sizes_file, to_keep=to_keep, to_filter=to_filter)

    chip_seq_coordinates = utils.load_chipseq_data(peaks_file, genome_sizes_file=genome_sizes_file, to_keep=to_keep, to_filter=to_filter)

    print(f"chip_seq_coordinates = {chip_seq_coordinates.shape}")
    
    blacklist_bdt = BedTool(blacklist_file)

    bound_chip_peaks_bdt = BedTool.from_dataframe(chip_seq_coordinates).intersect(blacklist_bdt, v=True)

    print("Chopping genome")
    chopped_genome_df = utils.chop_genome(genome_sizes_file, to_keep, excl=blacklist_bdt, stride=config.stride, l=config.window_len)
    chopped_genome_bdt = BedTool.from_dataframe(chopped_genome_df)
    print(f"Chopping genome = {chopped_genome_df.shape}")
    
    bound_bdt_obj = chopped_genome_bdt.intersect(bound_chip_peaks_bdt, u=True)

    bound_bdt_obj = bound_bdt_obj.intersect(blacklist_bdt, v=True)
    
    bound_df = bound_bdt_obj.to_dataframe().assign(type="pos_all", label=1)
    print(f"bound_df = {bound_df.shape}")

    if config.num_oversample_chip_seq_peaks > 0:
        bound_oversample_df_list=[bound_df]
        for i in range(config.num_oversample_chip_seq_peaks):
            print(f"Oversampling {i}")
            _bound_oversample_df=utils.make_random_shift(bound_df, L=config.window_len)
            bound_oversample_df_list.append(_bound_oversample_df)
        bound_df=pd.concat(bound_oversample_df_list, axis=0)
        print(f"bound_df (after oversample) = {bound_df.shape}")
    else:
        print("Skipping oversampling....")
        
        
    # NEG. SAMPLES
    
    print("getting unbound genome")
    unbound_bdt_obj = chopped_genome_bdt.intersect(bound_chip_peaks_bdt, v=True)
    
    unbound_bdt_obj = unbound_bdt_obj.intersect(blacklist_bdt, v=True)
    
    unbound_df = unbound_bdt_obj.to_dataframe().assign(type="neg_genome", label=0)
    print(f"unbound_all_df = {unbound_df.shape}")

    print(unbound_df)
    print(unbound_df.shape)
    
    total_pos_sample_size=bound_df.shape[0]
    unbound_genome_sample_size=total_pos_sample_size*config.EPOCHS*2

    if unbound_df.shape[0]>=unbound_genome_sample_size:
        print("="*20)
        print("unbound_genome_bdt_df")
        print(unbound_df.shape[0],unbound_genome_sample_size)
        print("Sampling with No replacement....")
        print("="*20)
        unbound_df = unbound_df.sample(n=unbound_genome_sample_size, replace=False)
    else:
        print("="*20)
        print("unbound_genome_bdt_df")
        print(unbound_all_df.shape[0],unbound_genome_sample_size)
        print("Sampling with replacement....")
        print("="*20)
        unbound_df = unbound_df.sample(n=unbound_genome_sample_size, replace=True)
        
        num_oversample = floor((unbound_genome_sample_size-unbound_all_df.shape[0])/unbound_all_df.shape[0])
        print(f"Unbound oversample = {num_oversample}")
        unbound_oversample_df_list=[unbound_all_df]
        for i in range(num_oversample):
            print(f"Oversampling unbound regions {i}")
            _unbound_oversample_df=utils.make_random_shift(unbound_all_df, L=config.window_len+config.context_window_len)
            unbound_oversample_df_list.append(_unbound_oversample_df)
        unbound_df=pd.concat(unbound_oversample_df_list, axis=0)
        
        unbound_bdt_obj = BedTool.from_dataframe(unbound_df).intersect(bound_chip_peaks_bdt, v=True)
        unbound_df = unbound_bdt_obj.to_dataframe().assign(type="neg_genome", label=0)
        
        if unbound_df.shape[0]<unbound_genome_sample_size:
            unbound_df = unbound_df.sample(n=unbound_genome_sample_size, replace=True)
        
        print(f"unbound_df (after oversample) = {bound_oversample_df.shape}")   
        
    
    bound_df["id"] = list(range(bound_df.shape[0]))
    
    print("#"*20)
    print(f"bound_df shape before sample = {bound_df.shape}")
    bound_df = bound_df.sample(frac=config.frac, replace=False)
    print(f"bound_df shape after sample = {bound_df.shape}")
    print("#"*20)
    
    
    unbound_df["id"]=list(range(unbound_df.shape[0]))

    print("#"*20)
    print(f"unbound_df shape before sample = {unbound_df.shape}")
    unbound_df = unbound_df.sample(frac=config.frac, replace=False)
    print(f"unbound_df shape after sample = {unbound_df.shape}")
    print("#"*20)
    
    bound_df.to_csv(f"{out_path}/training_df_bimodal_bound.bed", header=False, index=False, sep="\t")
    unbound_df.to_csv(f"{out_path}/training_df_bimodal_unbound.bed", header=False, index=False, sep="\t")

    # training_df["label"]=np.nan

    print(f"bound_oversample_df shape = {bound_df.shape}")
    print(f"unbound_df shape = {unbound_df.shape}")

    return f"{out_path}/training_df_bimodal_bound.bed", f"{out_path}/training_df_bimodal_unbound.bed"


def construct_internal_test_set():
    
    out_path=config.data_path
    
    colnames=["chr","start","end","type","label","id"]
    training_df_seq=pd.read_csv(f"{out_path}/training_df_seq.bed", header=None, sep="\t", names=colnames)
    bound_df_train_bimodal= pd.read_csv(f"{out_path}/training_df_bimodal_bound.bed", header=None, sep="\t", names=colnames)
    unbound_df_train_bimodal = pd.read_csv(f"{out_path}/training_df_bimodal_unbound.bed", header=None, sep="\t", names=colnames)
    
    print(f"training_df_seq shape = {training_df_seq.shape}")
    print(f"bound_df_train_bimodal shape = {bound_df_train_bimodal.shape}")
    print(f"unbound_df_train_bimodal shape = {unbound_df_train_bimodal.shape}")
    
    bound_df_train_bimodal, bound_df_test = train_test_split(bound_df_train_bimodal, test_size=0.1)
    unbound_df_train_bimodal, unbound_df_test = train_test_split(unbound_df_train_bimodal, test_size=0.1)
    
    training_bdt_seq = BedTool.from_dataframe(training_df_seq)
    bound_bdt_test = BedTool.from_dataframe(bound_df_test)
    unbound_bdt_test = BedTool.from_dataframe(unbound_df_test)
    
    training_bdt_seq = training_bdt_seq.intersect(bound_bdt_test, v=True)
    training_bdt_seq = training_bdt_seq.intersect(unbound_bdt_test, v=True)
    
    training_df_seq = training_bdt_seq.to_dataframe()
    
    test_df = pd.concat([bound_df_test, unbound_df_test])
    test_df["id"]=list(range(test_df.shape[0]))
    
    print()
    print(f"training_df_seq shape = {training_df_seq.shape}")
    print(f"bound_df_train_bimodal shape = {bound_df_train_bimodal.shape}")
    print(f"unbound_df_train_bimodal shape = {unbound_df_train_bimodal.shape}")
    print(f"test_df shape = {test_df.shape}")
    
    print(test_df)
    
    training_df_seq.to_csv(f"{out_path}/training_df_seq.bed", header=False, index=False, sep="\t")
    bound_df_train_bimodal.to_csv(f"{out_path}/training_df_bimodal_bound.bed", header=False, index=False, sep="\t")
    unbound_df_train_bimodal.to_csv(f"{out_path}/training_df_bimodal_unbound.bed", header=False, index=False, sep="\t")
    test_df.to_csv(f"{out_path}/test_df_internal.bed", header=False, index=False, sep="\t")
    
    return f"{out_path}/test_df_internal.bed"

def construct_external_test_set(genome_sizes_file, peaks_file, blacklist_file, to_keep, window_length, stride=5000):

    out_path=config.data_path
    
    # prepare file for defining coordinates
    blacklist_bdt = BedTool(blacklist_file)
  
    # get the coordinates for test samples
    bound_chip_peaks = utils.load_chipseq_data(peaks_file, genome_sizes_file=genome_sizes_file, to_keep=to_keep)

    print(bound_chip_peaks)
    
    bound_chip_peaks_bdt = BedTool.from_dataframe(bound_chip_peaks).intersect(blacklist_bdt, v=True)
    
    chopped_genome_df = utils.chop_genome(genome_sizes_file, to_keep, excl=blacklist_bdt, stride=1000, l=window_length)
                  
    chopped_genome_bdt_obj = BedTool.from_dataframe(chopped_genome_df)
    
    bound_chip_peaks = chopped_genome_bdt_obj.intersect(bound_chip_peaks_bdt, u=True)
    
    bound_chip_peaks = bound_chip_peaks.to_dataframe().assign(type="pos_peak",label=1)
    
    print(bound_chip_peaks)
  
    unbound_genome_chop = chopped_genome_bdt_obj.intersect(bound_chip_peaks_bdt, v=True).to_dataframe().assign(type="neg_chop",label=0)
    
    print(f"bound_chip_peaks = {bound_chip_peaks.shape}")
    print(f"unbound_genome_chop = {unbound_genome_chop.shape}")
    
    bound_chip_peaks = bound_chip_peaks.sample(frac=config.frac)
    unbound_genome_chop = unbound_genome_chop.sample(frac=config.frac)
#     unbound_genome_chop = unbound_genome_chop.sample(n=bound_chip_peaks.shape[0]*10)
    test_coords = pd.concat([bound_chip_peaks, unbound_genome_chop])
    test_coords["id"]=list(range(test_coords.shape[0]))
    
    print(test_coords[test_coords["type"]=="pos_peak"].head())
    print(test_coords[test_coords["type"]=="neg_chop"].head())
    print(test_coords["type"].value_counts())
    
    test_coords.to_csv(f"{out_path}/test_df_external.bed", header=False, index=False, sep="\t")

    return f"{out_path}/test_df_external.bed"


def get_data_stats(bed_file_path_list=None, out_path=None):
    
    print("Getting data stats")
    
    out_file_path = f"{out_path}/stats.txt"
    
    with open(out_file_path, "w") as f:
        for bed_file_path in bed_file_path_list:
            bed_file_name = bed_file_path.split("/")[-1]
            print(bed_file_name)
            df = pd.read_csv(bed_file_path, header=None, names=["chrom", "start", "end", "type", "label", "id"], sep="\t")
            f.write("="*25)
            f.write("\n")
            f.write(bed_file_name)
            f.write("\n")
            f.write("\n")
            f.write(df["label"].value_counts().to_string())
            f.write("\n")
            f.write(df["label"].value_counts(normalize=True).to_string())
            f.write("\n")

        
def create_seq_onehot_pickle():
    
    print(config.seq_onehot_dict_out_path)
    if not os.path.exists(config.seq_onehot_dict_out_path):
        print("Creating")
        subprocess.call(['mkdir', "-p", config.seq_onehot_dict_out_path])
    else:
        print(f"onehot_seq_dict dir exists")
    
    _seq_onehot_dict_out_path = f"{config.seq_onehot_dict_out_path}/onehot_seq_dict_res_{config.stride}.pickle"
    
    if not os.path.isfile(_seq_onehot_dict_out_path):
        genome_sizes_df = pd.read_csv(config.info, sep='\t', header=None, names=['chrom', 'length'])
        print(genome_sizes_df)
        print(config.stride)
        genome_pyfasta = pyfasta.Fasta(config.fa)

        chr_dict={}
        for _, row in genome_sizes_df.iterrows():
            _chr=row["chrom"]
            _chr_length=row["length"]
            print(_chr, _chr_length)
            l=[]
            for i in tqdm(range(0, _chr_length, config.stride)):
                seq = genome_pyfasta[_chr][i:i+config.stride]
                seq_onehot = utils.dna2onehot(seq)
                seq_onehot = seq_onehot.astype("int8")
                l.append(seq_onehot)
            chr_dict[_chr]=l
   
        with open(_seq_onehot_dict_out_path, 'wb') as handle:
            pickle.dump(chr_dict, handle)
    else:
        print(f"onehot_seq_dict_res_{config.stride}.pickle exists")


def create_hdf5_chromatin_tracks():
    global arr_values
    print("#"*20)
    print(config.hdf5_chromatin_tracks_out_path)
    if not os.path.exists(config.hdf5_chromatin_tracks_out_path):
        print("Creating hdf5 dir")
        subprocess.call(['mkdir', "-p", config.hdf5_chromatin_tracks_out_path])
    else:
        print(f"hdf5_chromatin_tracks_out_path dir exists")
    
    _hdf5_chromatin_tracks_out_path = f"{config.hdf5_chromatin_tracks_out_path}/hdf5_chromatin_tracks_{config.stride}.h5"
    print(_hdf5_chromatin_tracks_out_path)
    
    if not os.path.isfile(_hdf5_chromatin_tracks_out_path):
#         bw_file_path_list = glob.glob(f'{config.chromatin_tracks_path}/*.bw')
        bw_file_path_list=config.chromatin_tracks_path_list
        bw_obj_list = [pyBigWig.open(bw) for bw in bw_file_path_list]

        genome_sizes_df = pd.read_csv(config.info, sep='\t', header=None, names=['chrom', 'length'])
        print(genome_sizes_df)
        print(config.stride)
        
        hdf5_file = h5py.File(_hdf5_chromatin_tracks_out_path, 'w')
        for bw_file_path,bw in zip(bw_file_path_list, bw_obj_list):
            chromatin_track_name = bw_file_path.split("/")[-1].split(".")[0]
            print(chromatin_track_name)
            hdf5_file.create_group(chromatin_track_name)
            for _, row in genome_sizes_df.iterrows(): 
                _chr=row["chrom"]
                _chr_length=row["length"]
                print(_chr, _chr_length)
                
                hdf5_file[chromatin_track_name].create_group(_chr)

                print(f'Mean = {bw.stats(_chr, type="mean")[0]}  Min = {bw.stats(_chr, type="min")[0]}  Max = {bw.stats(_chr, type="max")[0]}')
                
                # arr_values = []
                # for i in tqdm(range(0, _chr_length, config.stride)):
                    # if i+config.stride < _chr_length:
                        # value = bw.stats(_chr, i, i+config.stride, type="mean")[0]
                        # values = bw.values(_chr, i, i+config.stride)
                        # values = np.array(values)
                        # values = np.clip(values, None, 50000)
                        # values_log = np.log(values+1)
                        # mean_values_log = np.mean(values_log)
                        # arr_values.append(mean_values_log)
                
                values = bw.values(_chr, 0, _chr_length)
                end = (_chr_length//config.stride)*config.stride
                arr_values = np.array(values)[:end]
                arr_values = np.clip(arr_values, None, 10000)
                arr_values = np.log2(arr_values+1)
                arr_values = arr_values.reshape(-1,config.stride).mean(axis=1)
                print(end, arr_values.shape)
                data=np.array(arr_values).astype("float16")
                print(f'Mean = {np.mean(data)}  Min = {np.min(data)}  Max = {np.max(data)}')
                hdf5_file[f"{chromatin_track_name}/{_chr}"].create_dataset("data", data=data, chunks=True)
        hdf5_file.close()
    else:
        print("hdf5 exists")


def main():
    
    print(f"Frac = {config.frac}")
    print(f"window_len = {config.window_len}")
    print()
    
    if len(set.intersection(set(config.val_chroms), set(['chrM', 'chrUn']))) or len(set.intersection(set(config.test_chroms), set(['chrM', 'chrUn']))):
        raise ValueError("Validation and Test Sets must not use chrM, chrUn")

    if len(set.intersection(set(config.val_chroms), set(config.test_chroms))):
        raise ValueError("Validation and Test Sets must not have any intersection")

    print('Creating output directory')
    print(config.data_path)
    subprocess.call(['mkdir', "-p", config.data_path])
    subprocess.call(["cp", config.config_file_path, config.data_path])

    print(config.chromatin_tracks_path_list)
    
    """
    training_chrom_list = pd.read_csv(config.info, sep="\t", names=["chrom", "len"])["chrom"].tolist()
    print(training_chrom_list)
    print(len(training_chrom_list))
    for chrom in config.val_chroms+config.test_chroms:
        print(f"Removing {chrom} from training_chrom_list")
        training_chrom_list.remove(chrom)
    """
    # training_chrom_list = ["chr1"]
    
    print(config.training_chrom_list)
    print(len(config.training_chrom_list))

    
    # Construct Data
    print('-->Constructing train data ...')
    
    create_seq_onehot_pickle()

    create_hdf5_chromatin_tracks()

    print('-->Constructing train data seq...')
    construct_training_set_seq(genome_sizes_file=config.info,
                               genome_fasta_file=config.fa,
                               peaks_file=config.peaks,
                               blacklist_file=config.blacklist, 
                               window_length=config.window_len,
                               to_filter=config.val_chroms + config.test_chroms + ['chrM', 'chrUn'],
                               to_keep=config.training_chrom_list,
                               out_path=config.data_path,
                               nbins=config.nbins, 
                               stride=config.stride
                              )
    
    print('-->Constructing train data bimodal...')
    construct_training_set_bimodal(genome_sizes_file=config.info,
                                   genome_fasta_file=config.fa,
                                   peaks_file=config.peaks,
                                   blacklist_file=config.blacklist, 
                                   window_length=config.window_len,
                                   to_filter=config.val_chroms + config.test_chroms + ['chrM', 'chrUn'],
                                   to_keep=config.training_chrom_list,
                                   out_path=config.data_path,
                                   nbins=config.nbins, 
                                   stride=config.stride
                                  )
    

    # print('-->Constructing validation data ...')
    # bed_file_val = construct_test_set(genome_sizes_file=config.info,
    #                                   peaks_file=config.peaks,
    #                                   genome_fasta_file=config.fa,
    #                                   blacklist_file=config.blacklist,
    #                                   window_length=config.window_len,
    #                                   to_keep=config.val_chroms,
    #                                   out_prefix=config.data_path + '/data_val',
    #                                   nbins=config.nbins,
    #                                   stride=config.stride

    print('-->Constructing internal test data ...')
    bed_file_internal_test_set = construct_internal_test_set()
    
    print('-->Constructing external test data ...')
    construct_external_test_set(genome_sizes_file=config.info,
                                peaks_file=config.peaks,
                                blacklist_file=config.blacklist, 
                                to_keep=config.test_chroms,
                                window_length=config.window_len,
                                stride=config.stride,
                               )

    get_data_stats(bed_file_path_list = [f"{config.data_path}/training_df_seq.bed",
                                         f"{config.data_path}/training_df_bimodal_bound.bed",
                                         f"{config.data_path}/training_df_bimodal_unbound.bed",
                                         f"{config.data_path}/test_df_internal.bed",
                                         f"{config.data_path}/test_df_external.bed"
                                        ],
                   out_path=config.data_path
                  )

    
if __name__ == "__main__":
    main()
