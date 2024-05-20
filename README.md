# Bichrom with graph attention network (GAT)

### Description 
This is an extension to our earlier work on Bichrom (https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02218-6). We have added Graph Attention Network (GAT( to predict the binding of TFs using pre-existing contact matrix data. The GAT is implemented from this papar (https://genome.cshlp.org/content/32/5/930). <br>

There two networks in this approach:
1) **Sequence Netowrk (seq-net):** This networks is trained first on the sequence data to predict ChIP-seq track. It also serves as the feature generator for the nodes in the contact matrix.
2) **GAT Network:** This network is trained on the contact matrix and features extracted from trained seq-net to predict the binding probability.

### Data Construction Strategy

### Training Strategy

## Input Data

- MultiGPS .events file
- ChIP-seq .bigwig file
- Genome sizes file (.info)
- Genome .fasta file
- Blacklist regions .bed file

## How run bichrom

### 1) Modify parameters in config.py as necessary

### 2) Construct Training and Test Datasets
`./construct_data.sh`

### 3) Train Bichrom GAT
`./train_bichrom.sh`

### 4) Predict ChIP-seq Track (optional)
`./predict_chip_seq_track_from_seqnet.sh`

## Output

### Step-2 Output (`./construct_data.sh`)
- common_data: Directory containing onehot_seq dictionary & chip-seq hdf5 file 
- training_df_seq.bed
- training_df_bimodal_bound.bed
- training_df_bimodal_unbound.bed
- test_df_internal.bed
- test_df_external.bed
- stats.txt

### Step-3 Output (`./train_bichrom.sh`)
- seqnet: Directory containing saved seqnet models at each epoch
- bimodal: Directory containing saved GAT models at each epoch
- internal_test_set_performance
- external_test_set_performance
- ids: training samples IDs (for debug purposes) 

