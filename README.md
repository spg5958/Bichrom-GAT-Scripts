# Bichrom
This is an extension to our earlier work on Bichrom (https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02218-6). We have added Graph Attention Network (GAT) to predict the TF-DNA binding using pre-existing contact matrix data. The GAT is implemented from this papar (https://genome.cshlp.org/content/32/5/930). <br>

## Installation and Requirements 

[IN DEVELOPMENT]

Clone and navigate to the Bichrom-GAT-Scripts/code. <br>
```
cd  Bichrom-GAT-Scripts/code
```
## Brief Description of Each Script

- **config.py:** This is the configuration file which contains various parameters for data construction & training
- **construct_data.py:** This script generates training and test set
- **construct_data.sh:** This script submits job.
- **predict_chip_seq_track_from_seqnet.py:** 

## Usage

### Step 1 - Collect the following data in any input directory
You can find some input files in `example_input` directory.

- **MultiGPS .events file**
- **ChIP-seq .bigwig file**
- **Genome sizes file (.info)**
- **Genome .fasta file**
- **Blacklist regions .bed file**

### Step 2 - Modify parameters in config.py file as necessary
Some important parameters that you may need to modify are given below. Please refer to the `code/config.py` for additional parameters
```
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
in_path = "../example_input"

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
```

### Step 3 - Construct Input Data
In order to reduce the training time, we have incorporated few strategies during data construction.
1) Pre-compute one-hot encoded matrices and store them into a dictionary
2) Convert TF .bigwig file into HDF5 file

To improve the generalization of the GAT model, we construct ubound (-ve) training that is much large than bound (+ve) training set. We try generate enough -ve samples such that every batch will have unique -ve samples (Although this is not alway guaranteed).
```
# Activate conda environment 
source activate bichrom

Run: 
./construct_data.sh
```

### Step 3 - Output 
construct_data.py will produce following files which includes train, test bed files and other files in the specified output directory.

- **common_data:** Directory containing onehot_seq dictionary & chip-seq hdf5 file 
- **training_df_seq.bed:** Bed file containing the regions for prediction
- **training_df_bimodal_bound.bed:**
- **training_df_bimodal_unbound.bed:**
- **test_df_internal.bed:**
- **test_df_external.bed:**
- **val_df_external.bed:**
- **stats.txt:**
- **config.py:** Copy of configuration file
 
### Step 4 - Train and Evaluate Bichrom-GAT
There two networks in this approach:
1) **Sequence-only Netowrk (seq-net):** This networks is trained first on the sequence data to predict ChIP-seq track. It also serves as the feature generator for the nodes in the contact matrix. The input to seq-next is region of length `config.window_length` (prediction window) which get expanded to both size by half of `config.context_window_length` before training. Then, seq-net is trained on the sequence of size `config.window_length + config.context_window_length`. It predicts the ChIP-seq bigwig values averaged over bins of size 100. E.g. If input size of 400bp and context window size is 10,000bp then input is expanded to both sides by 5000pb and the out size is 
2) **GAT Network (GAT-net/Bimodal-net):** This network is trained on the contact matrix and features extracted from trained seq-net to predict the binding probability.
```
Run:
./train_bichrom.sh
```

### Step 4 - Description of Bichrom-GAT's Output
Bichrom output directory. 
  - **seqnet:** 
    - records the training loss of seq-net at each epoch.
    - stores models (PyTorch object) checkpointed after each epoch.
    - also, stores best performing model (i.e., mininum train loss).
    - creates train_hist_seq.csv which contains average training loss at every epoch
  - **bimodal:** 
    - records the training loss of GAT-net/Bimodal-net at each epoch.
    - stores models (PyTorch object) checkpointed after each epoch.
    -  also, stores best performing model (i.e., mininum train loss).
    - creates train_hist_seq.csv which contains average training loss at every epoch
  - **test_set_performance:**
    - **internal_test_set_performance:** stores internal test-set performance of the best GAT-net/Bimodal-net
      - test_set_metrics.txt: stores AUC ROC, AUC PRC, Confusion matrix, and number of +ve & -ve predictions at 0.5 cut-off
      - test_set_metrics.csv: stores epoch, AUC ROC, and AUC PRC
      - test_set_probs_bimodal.txt: stores probabilities predicted by best GAT-net/Bimodal-net
    - **external_test_set_performance:** stores external test-set performance of the best GAT-net/Bimodal-net
      - test_set_metrics.txt: stores AUC ROC, AUC PRC, Confusion matrix, and number of +ve & -ve predictions at 0.5 cut-off
      - test_set_metrics.csv: stores epoch, AUC ROC, and AUC PRC
      - test_set_probs_bimodal.txt: stores probabilities predicted by best GAT-net/Bimodal-net
 - **ids:** training samples IDs (for debugging purposes) 

  
### Step 5 - Moded Inference (Predict ChIP-seq Track) [optional]
Modify the following parameters in `predict_chip_seq_track_from_seqnet.py` file.
```
# chromosome for which to predict bigwig track
_chr="chr10"

# trained seq-net model path
seq_model_path=f"{config.train_out_path}/seqnet/best_seq_model_epoch_1_train_loss_0.32.pt"

# output directory path
out_path=f"{config.exp_path}/predict_chip_seq_track_from_seqnet"

# window length of each sample. This shold be equal to total window length that seq-net was train on (prediction window length + context window length)
chop_genome_window_size=config.window_len+config.context_window_len
```
Then run the following command to predict on ChIP-seq tracks <br>
```
Run:
./predict_chip_seq_track_from_seqnet.sh
```
