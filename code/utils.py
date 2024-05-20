""" Helper module with methods for one-hot sequence encoding and training """

import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import os
import argparse
import pyfasta
import pyBigWig
import logging
import multiprocessing
import functools
import math
from multiprocessing import Pool
from pybedtools import Interval, BedTool
from tqdm import tqdm
#import tensorflow as tf


def dna2onehot(dnaSeq):
    DNA2index = {
        "A": 0,
        "T": 1,
        "G": 2,
        "C": 3
    }

    seqLen = len(dnaSeq)

    # initialize the matrix to seqlen x 4
    seqMatrixs = np.zeros((seqLen,4), dtype=int)
    # change the value to matrix
    dnaSeq = dnaSeq.upper()
    for j in range(0,seqLen):
        try:
            seqMatrixs[j, DNA2index[dnaSeq[j]]] = 1
        except KeyError as e:
            continue
    return seqMatrixs
   

def _get_datapoint_features(item, genome_pyfasta, bigwigs, clr, nbins, window_len, window_len_bimodal, window_len_hic, resolution):
    # get seq info
    seq = genome_pyfasta[item.chrom][int(item.start):int(item.end)]
    seq_onehot = dna2onehot(seq)

    # get chrom info
#     extra = (window_len_bimodal - (item.end - item.start))//2
    extra = 0
    ms = []
    for idx, bigwig in enumerate(bigwigs):
        m = np.nan_to_num(bigwig.values(item.chrom, item.start-extra, item.end+extra))
        #print(m.shape)           
        #m = m.reshape((nbins, -1))#.mean(axis=1, dtype=float)
        ms.append(m)
    #chrom_input = np.concatenate(ms)
    chrom_input = np.array(ms)

    # GAT input
#     extra = (window_len_hic - (item.end - item.start))//2
#     region_size = item.end+extra - (item.start-extra)
#     end_bin = region_size//resolution
    #print(f"  region_size = {region_size}")
    extra = 0
    region = (item.chrom, item.start-extra, item.end+extra)
    adj_list = clr.matrix(balance=False).fetch(region)
    adj = np.array(adj_list)
    adj = np.clip(adj, 0, 60_000)
#     adj = adj[0:end_bin, 0:end_bin]
    #print(f"  adj mat shape = {adj.shape}")
    
    return {"seq": seq_onehot, "chrom_input": chrom_input, "adj":adj, "label": item.label, "id": item.id}


"""
def get_datapoint_features(item, genome_pyfasta, bigwigs, clr_tf_record_iter, nbins, window_len, window_len_bimodal, window_len_hic, resolution, onehot_seq_dict, stride):
    
    # get seq info
    _chr=item.chrom
    bin1=int(np.floor(item.start/stride))
    bin2=int(np.ceil(item.end/stride))
    
    l=[]
    for bin_id in tqdm(range(bin1,bin2)):
        a=onehot_seq_dict[_chr][bin_id]
        l.append(a)
    seq_onehot=np.vstack(l)
    
    # GAT input
    example=next(clr_tf_record_iter)
    print("IDs")
    print(item.id, tf.io.parse_tensor(example["id"][0], out_type=tf.int32).numpy())
    assert item.id==tf.io.parse_tensor(example["id"][0], out_type=tf.int32).numpy(), "IDs do not match"
    adj=tf.io.parse_tensor(example["adj"][0], out_type=tf.float16)
    adj=adj.numpy()
    adj = np.clip(adj, 0, 60_000)
    print(adj.shape)
    print(adj)
    return {"seq": seq_onehot, "chrom_input": 1, "adj":adj, "label": item.label, "id": item.id}
"""


def get_datapoint_features_seq(item, onehot_seq_dict, stride, chromatin_tracks_path_list, h5, nbins):
    
    out_dict = {}
    # get seq info
    _chr=item.chrom
    bin1=int(np.floor(item.start/stride))
    bin2=int(np.ceil(item.end/stride))
        
#     print(f"start = {item.start}")
#     print(f"end = {item.end}")
#     print(bin2-bin1)
    
    l=[]
    for bin_id in range(bin1,bin2):
        a=onehot_seq_dict[_chr][bin_id]
        l.append(a)
    seq_onehot=np.vstack(l)
    
    out_dict = {"seq": seq_onehot, "label": item.label, "id": item.id}
    
    for chromatin_track_path in chromatin_tracks_path_list:
        chromatin_track_name = chromatin_track_path.split("/")[-1].split(".")[0]
        m = np.array(h5[f"{chromatin_track_name}/{item.chrom}/data"][bin1:bin2]).reshape((nbins, -1))
        m = m.mean(axis=1)
        out_dict[chromatin_track_name] = m
    return out_dict


def get_datapoint_features_test_seq(item, onehot_seq_dict, stride, chromatin_tracks_path_list, h5, nbins):
    
    out_dict = {}
    # get seq info
    _chr=item.chrom
    bin1=int(np.floor(item.start/stride))
    bin2=int(np.ceil(item.end/stride))
        
    l=[]
    for bin_id in range(bin1,bin2):
        a=onehot_seq_dict[_chr][bin_id]
        l.append(a)
    seq_onehot=np.vstack(l)
    
    out_dict = {"seq": seq_onehot, "start": item.start, "end": item.end, "label": item.label, "id": item.id}
    
    for chromatin_track_path in chromatin_tracks_path_list:
        chromatin_track_name = chromatin_track_path.split("/")[-1].split(".")[0]
        m = np.array(h5[f"{chromatin_track_name}/{item.chrom}/data"][bin1:bin2]).reshape((nbins, -1))
        m = m.mean(axis=1)
        out_dict[chromatin_track_name] = m
    return out_dict

    
def get_datapoint_features_bimodal(item, cooler_obj, onehot_seq_dict, stride, window_len):
    
    # get seq info
    _chr=item.chrom
    bin1=int(np.floor(item.start/stride))
    bin2=int(np.ceil(item.end/stride))
        
    l=[]
    for bin_id in range(bin1,bin2):
        a=onehot_seq_dict[_chr][bin_id]
        l.append(a)
    seq_onehot=np.vstack(l)
    
    # GAT input
    
    region = (item.chrom, item.start, item.end)
    adj_list = cooler_obj.matrix(balance=False).fetch(region)
    adj = np.array(adj_list)
            
    # pre-process
    adj = np.clip(adj, 0, 1000)
    adj = adj * (np.ones(adj.shape) - np.eye(adj.shape[0]))
    adj[adj>0] = 1
            
    end_bin = (item.end - item.start)//window_len
    adj = adj[0:end_bin, 0:end_bin]
#     print(f"end_bin = {end_bin}")
#     print(f"-->  adj mat shape = {adj.shape}")
    return {"seq": seq_onehot, "adj":adj, "label": item.label, "id": item.id}
            

def getDevice():
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"DEVICE = {device}")
    return device


"""
Utilities for iterating constructing data sets and iterating over DNA sequence data.
"""

def filter_chromosomes(input_df, to_filter=None, to_keep=None):
    """
    This function takes as input a pandas DataFrame
    Parameters:
        input_df (dataFrame): A pandas dataFrame, the first column is expected to
        be a chromosome. Example: chr1.
        to_filter (list): Default None (bool = False), will iterate over list
        objects and filter the listed chromosomes.
        ( Default: None, i.e. this condition will not be triggered unless a list
        is supplied)
        to_keep (list): Default None, will iterate over list objects and only
        retain the listed chromosomes.
    Returns:
          output_df (dataFrame): The filtered pandas dataFrame
    """
    if to_filter:
        output_df = input_df.copy()
        for chromosome in to_filter:
            # note: using the str.contains method to remove all
            # contigs; for example: chrUn_JH584304
            bool_filter = ~(output_df['chrom'].str.contains(chromosome))
            output_df = output_df[bool_filter]
    elif to_keep:
        # keep only the to_keep chromosomes:
        # note: this is slightly different from to_filter, because
        # at a time, if only one chromosome is retained, it can be used
        # sequentially.
        filtered_chromosomes = []
        for chromosome in to_keep:
            filtered_record = input_df[(input_df['chrom'] == chromosome)]
            filtered_chromosomes.append(filtered_record)
        # merge the retained chromosomes
        output_df = pd.concat(filtered_chromosomes)
    else:
        output_df = input_df
    return output_df


def get_genome_sizes(genome_sizes_file, to_filter=None, to_keep=None):
    """
    Loads the genome sizes file which should look like this:
    chr1    45900011
    chr2    10001401
    ...
    chrX    9981013
    This function parses this file, and saves the resulting intervals file
    as a BedTools object.
    "Random" contigs, chrUns and chrMs are filtered out.
    Parameters:
        genome_sizes_file (str): (Is in an input to the class,
        can be downloaded from UCSC genome browser)
        to_filter (list): Default None (bool = False), will iterate over list
        objects and filter the listed chromosomes.
        ( Default: None, i.e. this condition will not be triggered unless a list
        is supplied)
        to_keep (list): Default None, will iterate over list objects and only
        retain the listed chromosomes.
    Returns:
        A BedTools (from pybedtools) object containing all the chromosomes,
        start (0) and stop (chromosome size) positions
    """
    genome_sizes = pd.read_csv(genome_sizes_file, sep='\t', header=None, names=['chrom', 'length'])

    genome_sizes_filt = filter_chromosomes(genome_sizes, to_filter=to_filter, to_keep=to_keep)

    genome_bed_data = []
    # Note: Modifying this to deal with unexpected (incorrect) edge case \
    # BedTools shuffle behavior.
    # While shuffling data, BedTools shuffle is placing certain windows at the \
    # edge of a chromosome
    # Why it's doing that is unclear; will open an issue on GitHub.
    # It's probably placing the "start" co-ordinate within limits of the genome,
    # with the end coordinate not fitting.
    # This leads to the fasta file returning an incomplete sequence \
    # (< 500 base pairs)
    # This breaks the generator feeding into Model.fit.
    # Therefore, in the genome sizes file, buffering 550 from the edges
    # to allow for BedTools shuffle to place window without running of the
    # chromosome.
    for chrom, sizes in genome_sizes_filt.values:
        genome_bed_data.append(Interval(chrom, 0 + 550, sizes - 550))
    genome_bed_data = BedTool(genome_bed_data)
    return genome_bed_data


'''
def load_chipseq_data(chip_peaks_file, genome_sizes_file, to_filter=None, to_keep=None, window_len=None, window_len_bimodal=None, window_len_hic=None):
    """
    Loads the ChIP-seq peaks data.
    The chip peaks file is an events bed file:
    chr1:451350
    chr2:91024
    ...
    chrX:870000
    This file can be constructed using a any peak-caller. We use multiGPS.
    Also constructs a 1 bp long bedfile for each coordinate and a
    BedTools object which can be later used to generate
    negative sets.
    """
    chip_seq_data_dict={"chrom":[],"start":[]}
    with open(chip_peaks_file,"r") as file:
        for line in file.readlines():
            if not line.startswith("#"):
                chrom = line.split(":")[0]
                start = line.split(":")[1].split()[0]
                chip_seq_data_dict["chrom"].append(chrom)
                chip_seq_data_dict["start"].append(int(start))
    chip_seq_data = pd.DataFrame.from_dict(chip_seq_data_dict)
    print(chip_seq_data)
#     chip_seq_data = pd.read_csv(chip_peaks_file, delimiter=':', header=None, names=['chrom', 'start'])
    chip_seq_data['end'] = chip_seq_data['start'] + 1

    chip_seq_data = filter_chromosomes(chip_seq_data, to_filter=to_filter, to_keep=to_keep)

    sizes = pd.read_csv(genome_sizes_file, names=['chrom', 'chrsize'], sep='\t')

    # filtering out any regions that are close enough to the edges to
    # result in out-of-range windows when applying data augmentation.
    chrom_sizes_dict = (dict(zip(sizes.chrom, sizes.chrsize)))
    extra = (window_len_hic - window_len)//2 
    chip_seq_data['window_max'] = chip_seq_data['end'] + extra
    chip_seq_data['window_min'] = chip_seq_data['start'] - extra

    chip_seq_data['chr_limits_upper'] = chip_seq_data['chrom'].map(chrom_sizes_dict)
    chip_seq_data = chip_seq_data[chip_seq_data['window_max'] <= chip_seq_data['chr_limits_upper']]
    chip_seq_data = chip_seq_data[chip_seq_data['window_min'] >= 0]
    chip_seq_data = chip_seq_data[['chrom', 'start', 'end']]

    return chip_seq_data
'''


def load_chipseq_data(chip_peaks_file, genome_sizes_file, to_filter=None, to_keep=None):
    """
    Loads the ChIP-seq peaks data.
    The chip peaks file is an events bed file:
    chr1:451350
    chr2:91024
    ...
    chrX:870000
    This file can be constructed using a any peak-caller. We use multiGPS.
    Also constructs a 1 bp long bedfile for each coordinate and a
    BedTools object which can be later used to generate
    negative sets.
    """
    chip_seq_data_dict={"chrom":[],"start":[]}
    with open(chip_peaks_file,"r") as file:
        for line in file.readlines():
            if not line.startswith("#"):
                chrom = line.split(":")[0]
                start = line.split(":")[1].split()[0]
                chip_seq_data_dict["chrom"].append(chrom)
                chip_seq_data_dict["start"].append(int(start))
    chip_seq_data = pd.DataFrame.from_dict(chip_seq_data_dict)
    print(chip_seq_data)
    chip_seq_data['end'] = chip_seq_data['start'] + 1

    chip_seq_data = filter_chromosomes(chip_seq_data, to_filter=to_filter,
                                       to_keep=to_keep)

    sizes = pd.read_csv(genome_sizes_file, names=['chrom', 'chrsize'],
                        sep='\t')

    # filtering out any regions that are close enough to the edges to
    # result in out-of-range windows when applying data augmentation.
    chrom_sizes_dict = (dict(zip(sizes.chrom, sizes.chrsize)))
    chip_seq_data['window_max'] = chip_seq_data['end']
    chip_seq_data['window_min'] = chip_seq_data['start']

    chip_seq_data['chr_limits_upper'] = chip_seq_data['chrom'].map(chrom_sizes_dict)
    chip_seq_data = chip_seq_data[chip_seq_data['window_max'] <=
                                  chip_seq_data['chr_limits_upper']]
    chip_seq_data = chip_seq_data[chip_seq_data['window_min'] >= 0]
    chip_seq_data = chip_seq_data[['chrom', 'start', 'end']]

    return chip_seq_data


# def exclusion_regions(blacklist_file, chip_seq_data):
#     """
#     This function takes as input a bound bed file (from multiGPS).
#     The assumption is that the bed file reports the peak center
#     For example: chr2   45  46
#     It converts these peak centers into 501 base pair windows, and adds them to
#     the exclusion list which will be used when constructing negative sets.
#     It also adds the mm10 blacklisted windows to the exclusion list.
#     Parameters:
#         blacklist_file (str): Path to the blacklist file.
#         chip_seq_data (dataFrame): The pandas chip-seq data loaded by load_chipseq_data
#     Returns:
#         exclusion_windows (BedTool): A bedtools object containing all exclusion windows.
#         bound_exclusion_windows (BedTool): A bedtool object containing only
#         those exclusion windows where there exists a binding site.
#     """
#     temp_chip_file = chip_seq_data.copy()  # Doesn't modify OG array.
#     temp_chip_file['start'] = temp_chip_file['start'] - 250
#     temp_chip_file['end'] = temp_chip_file['end'] + 250

#     if blacklist_file is None:
#         print('No blacklist file specified ...')
#         exclusion_windows = BedTool.from_dataframe(temp_chip_file[['chrom', 'start','end']])
#     else:
#         bound_exclusion_windows = BedTool.from_dataframe(temp_chip_file[['chrom', 'start','end']])
#         blacklist_exclusion_windows = BedTool(blacklist_file)
#         exclusion_windows = BedTool.cat(*[blacklist_exclusion_windows, bound_exclusion_windows])
#     return exclusion_windows


def make_random_shift(coords, L, buffer=25):
    """
    This function takes as input a set of bed coordinates dataframe 
    It finds the mid-point for each record or Interval in the bed file,
    shifts the mid-point, and generates a windows of length L.

    If training window length is L, then we must ensure that the
    peak center is still within the training window.
    Therefore: -L/2 < shift < L/2
    To add in a buffer: -L/2 + 25 <= shift <= L/2 + 25
    # Note: The 50 here is a tunable hyper-parameter.
    Parameters:
        coords(pandas dataFrame): This is an input bedfile (first 3 column names: "chr", "start", "end")
    Returns:
        shifted_coords(pandas dataFrame): The output bedfile with shifted coords
    """
    low = int(-L/2 + buffer)
    high = int(L/2 - buffer)

    mid_point_list = np.arange(np.ceil(low/50)*50, np.floor(high/50 + 1)*50, 50)
    
    # result_df = (coords.assign(midpoint=lambda x: (x["start"]+x["end"])/2)
    #         .astype({"midpoint": int})
    #         .assign(midpoint=lambda x: x["midpoint"] + np.random.randint(low=low, high=high, size=len(coords)))
    #         .apply(lambda s: pd.Series([s["chrom"], int(s["midpoint"]-L/2), int(s["midpoint"]+L/2)],
    #                                     index=["chrom", "start", "end"]), axis=1))
    
    result_df = coords.copy()
    
    print(result_df)
    print(f"result_df={result_df.shape}")
    result_df["shift"] = np.random.choice(mid_point_list, len(result_df))
    result_df["start"] += result_df["shift"]
    result_df["end"] += result_df["shift"]
    result_df["start"]=result_df["start"].astype(int)
    result_df["end"]=result_df["end"].astype(int)
    result_df=result_df.drop("shift", axis=1)
    print(result_df)
    
    return result_df

def make_flank(coords, L, d):
    """
    Make flanking regions by:
    1. Shift midpoint by d
    2. Expand midpoint to upstream/downstream by L/2
    """
    return (coords.assign(midpoint=lambda x: (x["start"]+x["end"])/2)
                .astype({"midpoint": int})
                .assign(midpoint=lambda x: x["midpoint"] + d)
                .apply(lambda s: pd.Series([s["chrom"], int(s["midpoint"]-L/2), int(s["midpoint"]+L/2)],
                                            index=["chrom", "start", "end"]), axis=1))


def random_coords(gs, incl, excl, l=500, n=1000):
    """
    Randomly sample n intervals of length l from the genome,
    shuffle to make all intervals inside the desired regions 
    and outside exclusion regions
    """
    return (BedTool()
            .random(l=l, n=n, g=gs)
            .shuffle(g=gs, incl=incl.fn, excl=excl.fn)
            .to_dataframe()[["chrom", "start", "end"]])


'''
def chop_genome(gs, chroms, excl, stride=500, l=500, window_len_bimodal=None, window_len_hic=None, genome_sizes_file=None):
    """
    Given a genome size file and chromosome list,
    chop these chromosomes into intervals of length l,
    with include/exclude regions specified
    """
    print(gs)
    def intervals_loop(chrom, start, stride, l, size):
        intervals = []
        print(chrom)
        for pos in tqdm(range(start, size, stride)):
            if (pos + l) < size:
                intervals.append((chrom, pos, pos+l))
        return pd.DataFrame(intervals, columns=["chrom", "start", "end"])
     
    genome_sizes = (pd.read_csv(gs, sep="\t", names=["chrom", "len"])
                        .set_index("chrom")
                        .loc[chroms])
    genome_chops = pd.concat([intervals_loop(i.Index, 0, stride, l, i.len) 
                                for i in tqdm(genome_sizes.itertuples())])
    
    print("#"*40)
    print("Removing regioins that are at edge")
    print("#"*40)
    sizes = pd.read_csv(genome_sizes_file, names=['chrom', 'chrsize'], sep='\t')

    # filtering out any regions that are close enough to the edges to
    # result in out-of-range windows when applying data augmentation.
    chrom_sizes_dict = (dict(zip(sizes.chrom, sizes.chrsize)))
    extra = (window_len_hic - l)//2 
    genome_chops['window_max'] = genome_chops['end'] + extra
    genome_chops['window_min'] = genome_chops['start'] - extra
    
    genome_chops['chr_limits_upper'] = genome_chops['chrom'].map(chrom_sizes_dict)
    genome_chops = genome_chops[genome_chops['window_max'] <= genome_chops['chr_limits_upper']]
    genome_chops = genome_chops[genome_chops['window_min'] >= 0]
    
    genome_chops_bdt = BedTool.from_dataframe(genome_chops)
    
    return (genome_chops_bdt.intersect(excl, v=True)
                            .to_dataframe()[["chrom", "start", "end"]])
'''

    
def chop_genome(genome_sizes_file, chroms, excl=None, stride=None, l=None):
    """
    Given a genome size file and chromosome list,
    chop these chromosomes into intervals of length l,
    with include/exclude regions specified
    """
    def intervals_loop(chrom, start, stride, l, size):
        print(chrom)
        intervals = []
        while True:
            if (start + l) < size:
                intervals.append((chrom, start, start+l))
            else:
                break
            start += stride
        return pd.DataFrame(intervals, columns=["chrom", "start", "end"])
     
    genome_sizes = (pd.read_csv(genome_sizes_file, sep="\t", names=["chrom", "len"])
                    .set_index("chrom")
                    .loc[chroms])
    
    print(genome_sizes)
    
    genome_chops_df = pd.concat([intervals_loop(i.Index, 0, stride, l, i.len) 
                                for i in genome_sizes.itertuples()])
    

    if excl is not None:
        genome_chops_bdt = BedTool.from_dataframe(genome_chops_df)
        return genome_chops_bdt.intersect(excl, v=True).to_dataframe()#[["chrom", "start", "end"]]
    else:
        return genome_chops_df
#     return (intersect_middle(genome_chops_bdt, excl, v=True, window_length=l, stride=stride).to_dataframe()[["chrom", "start", "end"]])
    
    
def clean_bed(coords):
    """
    Clean the bed file:
    1. Remove intervals with start < 0
    """
    return coords.loc[coords["start"]>=0]


# def get_data(coords, genome_fasta, chromatin_tracks, nbins, reverse=False, numProcessors=1):
#     """
#     Given coordinates dataframe, extract the sequence and chromatin signal
#     """
#     y = coords["label"]

#     # get pointer
#     genome_pyfasta = pyfasta.Fasta(genome_fasta)

#     # split coordinates and assign chunks to workers
#     chunks = np.array_split(coords, numProcessors)
#     get_coverage_worker_freeze = functools.partial(get_coverage_worker, nbins=nbins, 
#                                                     bigwig_files=chromatin_tracks, reverse=reverse)
#     pool = Pool(numProcessors)
#     res = pool.map_async(get_coverage_worker_freeze, chunks)

#     # let's take care of sequence
#     X_seq = get_sequence_worker(coords, genome_pyfasta, reverse=reverse)

#     # gather the results
#     chromatin_out_lists = res.get()
#     chromatin_out_lists = np.concatenate(chromatin_out_lists, axis=1)

#     return X_seq, chromatin_out_lists, y


def dna2onehot(dnaSeq):
    DNA2index = {
        "A": 0,
        "T": 1,
        "G": 2,
        "C": 3
    }

    seqLen = len(dnaSeq)

    # initialize the matrix to seqlen x 4
    seqMatrixs = np.zeros((seqLen,4), dtype=int)
    # change the value to matrix
    dnaSeq = dnaSeq.upper()
    for j in range(0,seqLen):
        try:
            seqMatrixs[j, DNA2index[dnaSeq[j]]] = 1
        except KeyError as e:
            continue
    return seqMatrixs


def rev_comp(inp_str):
    rc_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'c': 'g',
               'g': 'c', 't': 'a', 'a': 't', 'n': 'n', 'N': 'N'}
    outp_str = list()
    for nucl in inp_str:
        outp_str.append(rc_dict[nucl])
    return ''.join(outp_str)[::-1]    


def reshape_chrom_input(x, nbins):
    
    x = x.reshape((x.shape[0], x.shape[1], nbins, -1)).mean(axis=-1, dtype=float)
    return x


def intersect_middle(bed_obj1, bed_obj2, u=False, v=False, window_length=None, stride=None):
    middle_region_dict={}
    bed_df1 = bed_obj1.to_dataframe() 
    print(window_length, stride)
    middle_start = bed_df1["start"] + (window_length - stride)//2
    middle_end = middle_start + stride
    middle_region_dict["chrom"] = bed_df1["chrom"]
    middle_region_dict["start_middle"] = middle_start
    middle_region_dict["end_middle"] = middle_end
    middle_region_dict["start_full"] = bed_df1["start"]
    middle_region_dict["end_full"] = bed_df1["end"]
    middle_region_df = pd.DataFrame.from_dict(middle_region_dict)
    
    middle_region_bdt = BedTool.from_dataframe(middle_region_df)
    out_bdt_obj = "s"
    if u:
        out_bdt_obj = middle_region_bdt.intersect(bed_obj2, u=True)
    elif v:
        out_bdt_obj = middle_region_bdt.intersect(bed_obj2, v=True)
    
    out_bed_df = out_bdt_obj.to_dataframe(names = ["chrom", "start_middle", "end_middle", "start", "end"])
    print("Middle bed")
    print(out_bed_df.head())
#     out_bed_df["start"] = out_bed_df["start"] - (window_length - stride)//2
#     out_bed_df["end"] = out_bed_df["end"] + (window_length - stride)//2
    out_bed_df = out_bed_df[["chrom", "start", "end"]]
    print(out_bed_df.head())
    out_bed_obj = BedTool.from_dataframe(out_bed_df)
    return out_bed_obj


# def get_sequence_worker(coords, fasta, reverse=False):
#     """
#     Get the sequence in provided regions
#     """
#     seqs = []
#     for item in coords.itertuples():
#         seq = fasta[item.chrom][int(item.start):int(item.end)]
#         if reverse:
#             seq = rev_comp(seq)
#         seqs.append(seq)
#     return seqs


# def get_coverage_worker(coords, bigwig_files, nbins, reverse=False):
#     """
#     Get the signal coverage in provided regions, summarize mean in each bin
#     """
#     bigwigs = [pyBigWig.open(bw) for bw in bigwig_files]

#     ms = [[] for x in bigwigs]
#     for idx, bigwig in enumerate(bigwigs):
#         for item in coords.itertuples():
#             try:
#                 m = (np.nan_to_num(bigwig.values(item.chrom, item.start, item.end))
#                                     .reshape((nbins, -1))
#                                     .mean(axis=1))
#             except RuntimeError as e:
#                 logging.warning(e)
#                 logging.warning(f"Skip region: {item}")
#                 continue
#             if reverse:
#                 m = m[::-1]
#             ms[idx].append(m)
#     return np.array(ms)
