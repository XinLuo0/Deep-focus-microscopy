# Deep focus microscopy

Efficient extended-depth-of-field high-resolution fluorescence imaging through optical hardware and deep learning optimizations

# Overview

We propose deep focus microscopy, an efficient framework optimized both in hardware and algorithm to address the tradeoff between resolution and DOF. Deep focus microscopy could achieve large-DOF and high-resolution imaging by integrating a deep focus network (DFnet) into light field microscopy (LFM) setups. Based on our constructed dataset, deep focus microscopy features significantly enhanced spatial resolution of ~260 nm, extended DOF of over 30 μm, broad generalization across diverse sample structures, and 4 orders of magnitude fewer computational costs than WFM or other LFM technologies. We demonstrate the excellent performance of deep focus microscopy in zebrafish embryos and mouse livers *in vivo*, including capturing a series of time-lapse 2D high-resolution images of cell division and migrasome formation without background contamination.

# Environment

We recommend the version of platform based on the configuration of NVIDIA RTX 3090 GPU:

- Python 3.7
- Pytorch 2.0.1
- CUDA  12.0

# Train

- Download the dataset and unzip them to ```./Datasets/```. Place the label HR images in ```./Datasets/train_HR``` and the raw LF images in ```./Datasets/train_LF```.

- To start training the model, run:

  ```python
  python train.py
  ```

- Checkpoint models will be saved to `./runs/`.

# Test

- We provide a demo test data `./Data/LF/LF.tif`. More extensive data pairs can be found in our proposed dataset.

- To test the model, run:

  ```python
  python test.py
  ```

- The high-resolution reconstructed results (`.tif` files) will be saved in `./Data/HR`.