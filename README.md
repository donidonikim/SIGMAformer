# SIGMAformer: A Spatiotemporal Gaussian Mixture Correlation Transformer for Global Weather Forecasting

## Proposed SIGMAformer framework

![SIGMAformer Framework](images/framework.png)

#### (a) Global distribution of weather observation stations used in the spatiotemporal weather forecasting. (b) Transformer-based forecasting framework, where the conventional attention mechanism is replaced by the proposed DSTC module. (c) The DSTC module includes two key components: temporal correlation modeling and spatial correlation modeling. These components compute weighted spatiotemporal correlations that are aggregated for the final output. (d) Detailed process of spatiotemporal correlation calculation using a GMM-based pattern extraction technique. This includes sampling, noise injection, updating the GMM, and interpolating cluster results to align with the original temporal resolution.
