"""PyTorch module for SIMVI for modeling cellular interaction variation."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


import torch.optim as optim

#from simvi.module.utils import GATv2Conv_weight
#from simvi.module.utils import StochasticGates
from torch_geometric.nn.conv import GATv2Conv

torch.backends.cudnn.benchmark = True


class SimVIModule(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_output: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        kl_weight: float = 1,
        kl_gatweight: float = 1,
        lam_mi: float = 50,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"
        self.dispersion = "gene"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.use_observed_lib_size = use_observed_lib_size
        self.var_eps = var_eps
        self.lam_mi = lam_mi
        self.kl_weight = kl_weight
        self.kl_gatweight = kl_gatweight
        #self.weight_ = nn.Parameter(torch.ones(n_output)*0.5,requires_grad=True)

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        cat_list = [n_batch]

        self.base_encoder = Encoder(
            n_input,
            n_output,
            n_cat_list=cat_list,
            n_layers=2,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
            var_eps = var_eps
        )
        self.gat_mean = GATv2Conv(in_channels=n_hidden,out_channels=n_output,add_self_loops=False)
        self.gat_var = GATv2Conv(in_channels=n_hidden,out_channels=n_output,add_self_loops=False)

        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )

        self.decoder = DecoderSCVI(
            2 * n_output,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

    @staticmethod
    def _get_inference_input_from_concat_tensors(
        tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = dict(x=torch.Tensor(x), batch_index=torch.Tensor(batch_index))
        return input_dict

    @staticmethod
    def _get_generative_input_from_concat_tensors(
        concat_tensors: Dict[str, Dict[str, torch.Tensor]], index: str
    ) -> Dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = dict(batch_index=batch_index)
        return input_dict

    @staticmethod
    def _get_generative_input_from_inference_outputs(
        inference_outputs: Dict[str, Dict[str, torch.Tensor]], data_source: str
    ) -> Dict[str, torch.Tensor]:
        z = inference_outputs[data_source]["z"]
        s = inference_outputs[data_source]["s"]
        library = inference_outputs[data_source]["library"]
        return dict(z=z, s=s, library=library)


    @staticmethod
    def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
        return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))
    
    @auto_move_data
    def _compute_local_library_params(
        self, batch_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars
        
    @auto_move_data
    def _generic_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        edge_index: torch.Tensor,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        x_ = torch.log(1 + x_)

        q_m, q_v, z = self.base_encoder(x_, batch_index)
        q = self.base_encoder.encoder(x_, batch_index)
        qgat_m = self.gat_mean(q,edge_index)
        qgat_v = self.gat_var(q,edge_index)
        qgat_v = torch.exp(qgat_v) + self.var_eps
        dist = Normal(qgat_m, qgat_v.sqrt())
        z_gat = dist.rsample()
        
        z_all = torch.cat((z_gat,z),1)
        qall_m = torch.cat((qgat_m,q_m),1)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(x_, batch_index)
            library = library_encoded

        if n_samples > 1:
            q_m = self._reshape_tensor_for_samples(q_m, n_samples)
            q_v = self._reshape_tensor_for_samples(q_v, n_samples)
            z = self._reshape_tensor_for_samples(z, n_samples)
            qgat_m = self._reshape_tensor_for_samples(qgat_m, n_samples)
            qgat_v = self._reshape_tensor_for_samples(qgat_v, n_samples)
            qall_m = self._reshape_tensor_for_samples(qall_m, n_samples)
            z_gat = self._reshape_tensor_for_samples(z_gat, n_samples)
            z_all = self._reshape_tensor_for_samples(z_all, n_samples)

            if self.use_observed_lib_size:
                library = self._reshape_tensor_for_samples(library, n_samples)
            else:
                ql_m = self._reshape_tensor_for_samples(ql_m, n_samples)
                ql_v = self._reshape_tensor_for_samples(ql_v, n_samples)
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(
            z=z,
            q_m=q_m,
            q_v=q_v,
            qgat_m=qgat_m,
            qgat_v=qgat_v,
            z_gat=z_gat,
            qall_m = qall_m,
            z_all=z_all,
            library=library,
            ql_m=ql_m,
            ql_v=ql_v,
            batch_index = batch_index
        )
        return outputs

    @auto_move_data
    def inference(
        self,
        data: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        inference_input = self._get_inference_input_from_concat_tensors(data)
        outputs = self._generic_inference(**inference_input, edge_index = edge_index, n_samples=n_samples)
        return outputs


    @auto_move_data
    def _generic_generative(
        self,
        z_all: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            z_all,
            library,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    @auto_move_data
    def generative(
        self,
        latent: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        latent_z_shape = latent["z_all"].shape
        batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        latent_batch_size = latent["z"].shape[batch_size_dim]
        generative_input = {}

        for key in ["z_all", "library","batch_index"]:
            generative_input[key] = latent[key]

        outputs = self._generic_generative(**generative_input)
        return outputs

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute likelihood loss for negative binomial distribution. 

        Args:
        ----
            x: Input data.
            px_rate: Mean of distribution.
            px_r: Inverse dispersion.
            px_dropout: Logits scale of zero inflation probability.

        Returns
        -------
            Negative log likelihood (reconstruction loss) for each data point. If number
            of latent samples == 1, the tensor has shape `(batch_size, )`. If number
            of latent samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )
        return recon_loss


    @staticmethod
    def latent_kl_divergence(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and prior Gaussian.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
            prior_mean: Mean of the prior Gaussian.
            prior_var: Variance of the prior Gaussian.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
        self,
        batch_index: torch.Tensor,
        variational_library_mean: torch.Tensor,
        variational_library_var: torch.Tensor,
        library: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between library size variational posterior and prior.

        Both the variational posterior and prior are Log-Normal.
        Args:
        ----
            batch_index: Batch indices for batch-specific library size mean and
                variance.
            variational_library_mean: Mean of variational Log-Normal.
            variational_library_var: Variance of variational Log-Normal.
            library: Sampled library size.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    def _generic_loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x = torch.Tensor(tensors[REGISTRY_KEYS.X_KEY])
        batch_index = torch.Tensor(tensors[REGISTRY_KEYS.BATCH_KEY])



        q_m = inference_outputs["q_m"]
        q_v = inference_outputs["q_v"]
        qgat_m = inference_outputs["qgat_m"]
        qgat_v = inference_outputs["qgat_v"]
        library = inference_outputs["library"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        prior_z_m = torch.zeros_like(q_m)
        prior_z_v = torch.ones_like(q_v)
        prior_zgat_m = torch.zeros_like(qgat_m)
        prior_zgat_v = torch.ones_like(qgat_v)

        recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout)
        mi_loss = self.mi_loss(inference_outputs)
        kl_z = self.latent_kl_divergence(q_m, q_v, prior_z_m, prior_z_v)
        kl_zgat = self.latent_kl_divergence(qgat_m, qgat_v, prior_zgat_m, prior_zgat_v)
        kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        return dict(
            recon_loss=recon_loss,
            kl_z=kl_z,
            kl_zgat=kl_zgat,
            kl_library=kl_library,
            mi_loss = mi_loss,
        )

    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
    ) -> LossRecorder:
        """
        Compute (generator) loss terms for SIMVI. Here we use full-batch training due to the
        nature of graph-structured data. Also we implement the beta-VAE approach 
        (larger KL weight than 1) to enforce the distanglement of latent factors

        Args:
        ----
            concat_tensors: Tuple of data mini-batch. The first element contains
                background data mini-batch. The second element contains target data
                mini-batch.
            inference_outputs: Dictionary of inference step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            generative_outputs: Dictionary of generative step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            kl_weight: Importance weight for KL divergence of background and salient
                latent variables, relative to KL divergence of library size.

        Returns
        -------
            An scvi.module.base.LossRecorder instance that records the following:
            loss: One-dimensional tensor for overall loss used for optimization.
            reconstruction_loss: Reconstruction loss with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_local: KL divergence term with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_global: One-dimensional tensor for global KL divergence term.
        """

        losses = self._generic_loss(
            tensors,
            inference_outputs,
            generative_outputs,
        )
        reconst_loss = losses["recon_loss"]
        mi_loss = losses["mi_loss"]
        kl_divergence_z = losses["kl_z"]
        kl_divergence_zgat = losses["kl_zgat"]
        kl_divergence_l = losses["kl_library"]

        kl_local_for_warmup = kl_divergence_z + kl_divergence_zgat
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = self.kl_weight * kl_divergence_z + self.kl_gatweight * kl_divergence_zgat + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local) + self.lam_mi * mi_loss

        kl_local = dict(
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_zgat=kl_divergence_zgat
        )

        # LossRecorder internally sums the `reconst_loss`, `kl_local`, and `kl_global`
        # terms before logging, so we do the same for our `wasserstein_loss` term.
        return LossRecorder(
            loss,
            reconst_loss,
            self.lam_mi * mi_loss,
            kl_local
        )
        
    def mi_loss(
        self,
        inference_outputs: Dict[str, torch.Tensor],
    ):
        psi_x = inference_outputs["z"].detach()
        psi_y = inference_outputs["z_gat"]
        
        C_yy = self._cov(psi_y, psi_y)
        C_yx = self._cov(psi_y, psi_x)
        C_xy = self._cov(psi_x, psi_y)
        C_xx = self._cov(psi_x, psi_x)

        C_xx_inv = torch.inverse(C_xx+torch.eye(C_xx.shape[0], device=psi_x.device)*1e-3)

        l2 = -torch.logdet(C_yy-torch.linalg.multi_dot([C_yx,C_xx_inv,C_xy])) + torch.logdet(C_yy)
        return l2
    
    
    @staticmethod
    def _cov(psi_x, psi_y):
        """
        :return: covariance matrix
        """
        N = psi_x.shape[0]
        return (psi_y.T @ psi_x).T / (N - 1)
        
