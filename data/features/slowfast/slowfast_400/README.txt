SlowFast CNN, trained on kinetics-400 dataset.
Documentation about the model can be found at: https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
[1] Christoph Feichtenhofer et al, “SlowFast Networks for Video Recognition” https://arxiv.org/pdf/1812.03982.pdf
Related Jupyter notebook: find at code/07_slowfast/01_slowfast_ft.ipynb

This CNN receives batches of 32 frames, and outputs a 400-dim vector of probabilities of the frame-batch depicting one of the 400 actions included in the kinetics-400 dataset.
This folder contains the final 400-dim output of the network for frame batches extracted from the fish movie.
The first 31 frames are discarded.
