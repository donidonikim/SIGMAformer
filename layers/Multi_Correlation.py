import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np

class GMMPatternExtractor(nn.Module):
    def __init__(
        self,num_patterns=3,device='cuda',sample_rate=2,threshold=0.0001,noise_factor=0.001,n_components=3,update_interval=100,  
        gmm_pretrained_path=None): 
        super(GMMPatternExtractor, self).__init__()
        self.gmm_model = None
        if gmm_pretrained_path:
            self.gmm_model = joblib.load(gmm_pretrained_path)
        self.num_patterns = num_patterns
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.noise_factor = noise_factor
        self.n_components = n_components
        self.pattern_cache = None
        self.update_interval = update_interval
        self.update_counter = 0

    def _fit_gmm(self, sampled_queries):
        gmm = GaussianMixture(
            n_components=self.n_components, covariance_type='full', n_init=1, max_iter=100
        )
        sampled_queries += np.random.normal(0, self.noise_factor, sampled_queries.shape)
        gmm.fit(sampled_queries)
        return gmm

    def fit_gmm_patterns(self, queries):
        B, L, H, E = queries.shape
        sampled_queries = queries[:, ::self.sample_rate, :, :].reshape(-1, E).cpu().detach().numpy()

        if sampled_queries.shape[0] < self.n_components * 10:
            return 
        if self.update_counter % self.update_interval == 0:
            self.gmm_model = self._fit_gmm(sampled_queries)
        self.update_counter += 1

    def apply_gmm_patterns(self, queries):
        if self.gmm_model is None:
            return torch.ones(*queries.shape[:3], 1, device=self.device)

        B, L, H, E = queries.shape
        sampled_queries = queries[:, ::self.sample_rate, :, :].reshape(-1, E).cpu().detach().numpy()

        cluster_labels = self.gmm_model.predict(sampled_queries)
        cluster_tensor = torch.tensor(cluster_labels, device=self.device, dtype=torch.float32)
        cluster_tensor = cluster_tensor.view(B, -1, H)

        interpolated_pattern = F.interpolate(
            cluster_tensor.unsqueeze(-1).permute(0, 2, 1, 3),
            size=L,
            mode='nearest'
        ).permute(0, 2, 1, 3)

        return interpolated_pattern

class TempCorr(nn.Module):
    def __init__(self, output_attention=False, dropout=0.1, device='cuda'):
        super(TempCorr, self).__init__()
        self.pattern_extractor = GMMPatternExtractor(device=device)
        self.output_attention = output_attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        self.pattern_extractor.fit_gmm_patterns(queries)
        patterns = self.pattern_extractor.apply_gmm_patterns(queries)

        if patterns.shape != values.shape:
            patterns = patterns.expand_as(values)

        combined_values = values + self.dropout(values * torch.sigmoid(patterns))
        return (combined_values, patterns) if self.output_attention else (combined_values, None)


class TempCorrLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None, device='cuda'):
        super(TempCorrLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads).to(device)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads).to(device)
        self.value_projection = nn.Linear(d_model, d_values * n_heads).to(device)
        self.out_projection = nn.Linear(d_values * n_heads, d_model).to(device)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class SpatCorr(nn.Module):
    def __init__(self, mask_flag=True, output_attention=False, device='cuda'):
        super(SpatCorr, self).__init__()
        self.output_attention = output_attention
        self.pattern_extractor = GMMPatternExtractor(device=device)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
    
        gmm_patterns = self._compute_gmm_patterns(queries, values)
    
        global_correlation = torch.matmul(
            queries.view(B, L, H, -1), keys.transpose(2, 3).contiguous()
        ) / (E ** 0.5 + 1e-8)  # (B, L, H, L)
    
        global_correlation = global_correlation.mean(dim=-1, keepdim=True) 
        global_correlation = F.interpolate(
            global_correlation.permute(0, 2, 1, 3), 
            size=values.shape[1],
            mode="nearest"
        ).permute(0, 2, 1, 3)  # (B, H, L, 1)
    
        if gmm_patterns.shape[1] != values.shape[1]:  
            gmm_patterns = F.interpolate(
                gmm_patterns.permute(0, 2, 1, 3), 
                size=values.shape[1], 
                mode="nearest"
            ).permute(0, 2, 1, 3)   
    
        local_correlation = values * torch.sigmoid(global_correlation) * gmm_patterns
    
        # Hierarchical Fusion
        combined_values = local_correlation + torch.sigmoid(global_correlation)
    
        return (combined_values.contiguous(), gmm_patterns) if self.output_attention else (combined_values.contiguous(), None)

    def _compute_gmm_patterns(self, queries, values):
        if self.pattern_extractor is None:
            return torch.ones_like(values, device=values.device)
        
        self.pattern_extractor.fit_gmm_patterns(queries)
        gmm_patterns = self.pattern_extractor.apply_gmm_patterns(queries)

        if gmm_patterns.shape[1] != values.shape[1]:
            gmm_patterns = F.interpolate(
                gmm_patterns.permute(0, 2, 1, 3), 
                size=values.shape[1], 
                mode="nearest"
            ).permute(0, 2, 1, 3)

        return gmm_patterns

class SpatCorrLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None, device='cuda'):
        super(SpatCorrLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads).to(device)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads).to(device)
        self.value_projection = nn.Linear(d_model, d_values * n_heads).to(device)
        self.out_projection = nn.Linear(d_values * n_heads, d_model).to(device)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class SpatioTempCorr(nn.Module):
    def __init__(self, auto_correlation, cross_correlation, node_num, in_feats, out_feats, num_heads, dropout=0.1):
        super(SpatioTempCorr, self).__init__()
        self.node_num = node_num
        self.auto_correlation = auto_correlation
        self.cross_correlation = cross_correlation
        self.dropout = nn.Dropout(dropout)
        
        self.weight_auto = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weight_cross = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        self.out_projection = nn.Linear(out_feats, out_feats)

    def forward(self, x, cross, mask=None):
        B, L, C = x.shape
        
        auto_corr_values, _ = self.auto_correlation(x, cross, cross, attn_mask=mask)
        cross_corr_values, _ = self.cross_correlation(x, x, cross, attn_mask=mask)
        
        combined_values = (
            self.weight_auto * auto_corr_values +
            self.weight_cross * cross_corr_values
        )

        combined_output = self.out_projection(combined_values.reshape(B * L, -1)).reshape(B, L, -1)
        return self.dropout(combined_output)
