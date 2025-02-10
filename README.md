# SIGMAformer: A Spatiotemporal Gaussian Mixture Correlation Transformer for Global Weather Forecasting

## Proposed SIGMAformer framework

#### The proposed framework replaces the conventional attention mechanism with the DSTC module, which enhances spatiotemporal weather forecasting by leveraging GMM-based pattern extraction to compute and aggregate weighted temporal and spatial correlations.


![SIGMAformer Framework](images/framework.png)


## Usage 

1. Global weather datasets can be obtained from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing).

2. Install Pytorch and other necessary dependencies.
```
pip install -r requirements.txt
```
3. Train and evaluate model. We provide the experiment scripts under the folder ./scripts/. You can reproduce the experiment results as the following examples:

```
bash ./scripts/Global_Temp/SIGMAformer.sh
bash ./scripts/Global_Wind/SIGMAformer.sh
```
