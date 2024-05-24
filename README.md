# Bichrom
This is an extension to our earlier work on Bichrom (https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02218-6). We have added Graph Attention Network (GAT) to predict the TF-DNA binding using pre-existing contact matrix data. The GAT is implemented from this papar (https://genome.cshlp.org/content/32/5/930). <br>

## Installation and Requirements 

[IN DEVELOPMENT]

Clone and navigate to the Bichrom-GAT-Scripts/code. <br>
`cd  Bichrom-GAT-Scripts/code`

## Usage

### Step 1 - Collect following data in input directory
You can find some input files in `example_input` directory.

- MultiGPS .events file
- ChIP-seq .bigwig file
- Genome sizes file (.info)
- Genome .fasta file
- Blacklist regions .bed file

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


```
# Activate conda environment 
source activate bichrom

usage: 
./construct_data.sh
```

### Step 3 - Output 
construct_data.py will produce following files which includes train, test bed files and other files in the specified output directory.

- common_data: Directory containing onehot_seq dictionary & chip-seq hdf5 file 
- training_df_seq.bed
- training_df_bimodal_bound.bed
- training_df_bimodal_unbound.bed
- test_df_internal.bed
- test_df_external.bed
- stats.txt
- config.py: Copy of configuration file
 
### Step 2 - Train Bichrom

```
cd trainNN  
To view help:   
python run_bichrom.py -h
usage: run_bichrom.py [-h] -training_schema_yaml TRAINING_SCHEMA_YAML -len LEN
                      -outdir OUTDIR -nbins NBINS

Train and Evaluate Bichrom

optional arguments:
  -h, --help            show this help message and exit
  -training_schema_yaml TRAINING_SCHEMA_YAML
                        YAML file with paths to train, test and val data
  -len LEN              Size of genomic windows
  -outdir OUTDIR        Output directory
  -nbins NBINS          Number of bins

```
  
**Required arguments**: 

**training_schema_yaml**:  
This configuration files contains paths to the formatted train, test and validation data. This file will be automatically generated using construct_data.py (**see above - construct_data.py will output bichrom.yaml**).

In order to construct the training data, we implement several sampling strategies including over-sampling the negative training regions from accessible chromatin and from genomic regions flanking the TF binding sites (detailed in the paper). However, if you would like to construct training data using your own strategy, please input a custom configuration file here. More details for custom configuration files can be found at the bottom of the README.  

**len**:  
The size of genomic windows used for training, validation and testing. (Recommended: 500).    
**nbins**:  
The number of bins to use for binning the chromatin data.   
**outdir**:   
Bichrom's output directory.  


### Step 2 - Description of Bichrom's Output
Bichrom output directory. 
  * seqnet: 
    * records the validation loss and auPRC for each epoch the sequence-only network (Bichrom-SEQ).
    * stores models (tf.Keras Models) checkpointed after each epoch. 
    * stores ouput probabilities over the test data for a sequence-only network. 
  * bichrom: 
    * records the validation loss and auPRC for each epoch the Bichrom. 
    * stores models (tf.Keras Models) checkpointed after each epoch. 
    * stores the Bichrom ouput probabilities over testing data. 
  * metrics.txt: stores the test auROC and the auPRC for both a sequence-only network and for Bichrom. 
  * best_model.hdf5: A Bichrom tensorflow.Keras Model (with the highest validation set auPRC)
  * precision-recall curves for the sequence-only network and Bichrom.
  
~~### Optional: Custom Training Sets and YAML files~~

**TODO**: Due to currently Bichrom saving dataset in Tensorflow TFRecord format, a new way of providing custom training set and yaml files will be released.

### 2-D Bichrom embeddings
For 2-D latenet embeddings, please refer to the README in the ```Bichrom/latent_embeddings directory```

### Moded Inference (Predict)
Use `trainNN/predict_bed.py` to predict on user-provided regions

```
cd trainNN  
To view help:   
python predict_bed.py -h
usage: predict_bed.py [-h] -mseq MSEQ -msc MSC -fa FA -chromtracks CHROMTRACKS [CHROMTRACKS ...] 
                           -nbins NBINS -prefix PREFIX -bed BED

Use Bichrom model for prediction given bed file

optional arguments:
  -h, --help            show this help message and exit
  -mseq MSEQ            Sequence Model
  -msc MSC              Bichrom Model
  -fa FA                The fasta file for the genome of interest
  -chromtracks CHROMTRACKS [CHROMTRACKS ...]
                        A list of BigWig files for all input chromatin experiments, please follow the same order of training data
  -nbins NBINS          Number of bins for chromatin tracks
  -prefix PREFIX        Output prefix
  -bed BED              bed file describing region used for prediction

```
  
**Required arguments**: 

**mseq**/**msc**

Sequence-only/Bichrom model saved in HDF5 format

**fa**

Fasta file of the genome

**chromtracks**

Bigwig files used in `construct_data` step, **NOTE: Please provide the bigwig files in the exact same order as provided to construct_data.py**

**nbins**

Number of bins for binning, **NOTE: Please use the exactly same number as used in construct_data.py**

**prefix**

Prefix of the output predictions

**bed**

Bed file containing the regions for prediction
