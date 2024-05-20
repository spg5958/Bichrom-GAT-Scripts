# Bichrom with graph attention network (GAT)
This is an extension to our earlier work on Bichrom. We have added Graph Attention Network to predict the binding of TFs using pre-existing contact matrix data.

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
- Internal

