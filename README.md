# Bichrom with graph attention network (GAT)

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
training_df_seq.bed
training_df_bimodal_bound.bed
training_df_bimodal_unbound.bed
test_df_internal.bed
test_df_external.bed
stats.txt


