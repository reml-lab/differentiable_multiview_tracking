# Differentiable Multi-View Tracking
Repository for our paper: End-to-End Differentiable Multi-View Tracking: Architecture and Fine-Tuning Experiments accepted at FUSION 2025.

If you are interested in this project don't hesitate to email Colin Samplawski at colin.samplawski@sri.com for more information.

### Usage
- The environment for this project is managed using the [apptainer container framework](https://apptainer.org/docs/user/latest/).
- The container file can be downloaded from the following [Google Drive link](https://drive.google.com/file/d/1gBvVCxps5prjaX_uIMpG-Y3FZJ8XR2uU/view?usp=sharing).
- The evaluation scripts require an `SRCROOT` environment variable that contains the path to the cloned repo. This can be set with `export` or using a `.bashrc` file.

### Data Cache
A cache of the DETR query embeddings, videos as loose PNG files, and object ground truth is available at the following [Google Drive link](https://drive.google.com/file/d/1kfaSOtuHW29-9cI_ZOJ3C_qSBKMtdxWx/view?usp=sharing).
Download it and unzip to `$SRCROOT/cache`.

### Checkpoints
Likewise trained model checkpoints are available at the following [Google Drive link](https://drive.google.com/file/d/1tAHY9aqqIpHoqR1LFHfYC5W6f1oSeGBY/view?usp=sharing). Unzip them to `$SRCROOT/checkpoints`

### Evaluation
To run an evaluation use the following bash command: `bash exps/single_obj/eval.sh` or `bash exps/multi_obj/eval.sh`
