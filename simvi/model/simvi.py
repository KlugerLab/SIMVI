"""Model class for SIMVI for single cell expression data."""

import logging
import warnings
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import math
import torch
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.spatial import Delaunay
import pytorch_lightning as pl
import torch.optim as optim
from scvi import settings
from tqdm import tqdm
import torch.nn.functional as F

from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.dataloaders._anntorchdataset import AnnTorchDataset
from scvi.model._utils import (
    _get_batch_code_from_category,
    _init_library_size,
    scrna_raw_counts_properties,
)
from scvi.model.base import BaseModelClass
from scvi.model.base._utils import _de_core
from scvi.utils import setup_anndata_dsp
from scvi.train import TrainingPlan, TrainRunner
from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders._data_splitting import validate_data_split


from simvi.module.simvi import SimVIModule

logger = logging.getLogger(__name__)
Number = Union[int, float]


class SimVIModel(BaseModelClass):
    """
    Model class for SIMVI.
    Args:
    ----
        adata: AnnData object that has been registered via
            `SimVIModel.setup_anndata`.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_latent: Dimensionality of the latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        lam_mi: mutual information regularization strength.
        var_eps: minimum variance for variance encoder.
        kl_weight: kl divergence coefficient for the MLP encoder.
        kl_gatweight: kl divergence coefficient for the GAT encoder.
    """

    def __init__(
        self,
        adata: AnnData,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0,
        use_observed_lib_size: bool = True,
        lam_mi: float = 1,
        var_eps: float = 1e-4,
        kl_weight: float = 1,
        kl_gatweight: float = 1,
    ) -> None:
        super(SimVIModel, self).__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.module = SimVIModule(
            n_input=self.summary_stats["n_vars"],
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_output=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            lam_mi = lam_mi,
            var_eps = var_eps,
            kl_weight = kl_weight,
            kl_gatweight = kl_gatweight,
        )
        self._model_summary_string = "SimVI"
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Set up AnnData instance for SIMVI model.

        Args:
        ----
            adata: AnnData object containing raw counts. Rows represent cells, columns
                represent features.
            layer: If not None, uses this as the key in adata.layers for raw count data.
            batch_key: Key in `adata.obs` for batch information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_batch"]`. If None, assign the same batch to all the
                data.
            labels_key: Key in `adata.obs` for label information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_labels"]`. If None, assign the same label to all the
                data.
            size_factor_key: Key in `adata.obs` for size factor information. Instead of
                using library size as a size factor, the provided size factor column
                will be used as offset in the mean of the likelihood. Assumed to be on
                linear scale.
            categorical_covariate_keys: Keys in `adata.obs` corresponding to categorical
                data. Used in some models.
            continuous_covariate_keys: Keys in `adata.obs` corresponding to continuous
                data. Used in some models.

        Returns
        -------
            If `copy` is True, return the modified `adata` set up for SIMVI
            model, otherwise `adata` is modified in place.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def extract_edge_index(
        adata: AnnData,
        batch_key: Optional[str] = None,
        spatial_key: Optional[str] = 'spatial',
        method: str = 'knn',
        n_neighbors: int = 30,
        ):
        """
        Define edge_index for SIMVI model training.

        Args:
        ----
            adata: AnnData object.
            batch_key: Key in `adata.obs` for batch information. If batch_key is none,
            assume the adata is from the same batch. Otherwise, we create edge_index
            based on each batch and concatenate them.
            spatial_key: Key in `adata.obsm` for spatial location.
            method: method for establishing the graph proximity relationship between
            cells. Two available methods are: knn and Delouney. Knn is used as default
            due to its flexible neighbor number selection.
            n_neighbors: The number of n_neighbors of knn graph. Not used if the graph
            is based on Delouney triangularization.

        Returns
        -------
            edge_index: torch.Tensor.
        """
        if batch_key is not None:
            j = 0
            for i in adata.obs[batch_key].unique():
                adata_tmp = adata[adata.obs[batch_key]==i].copy()
                if method == 'knn':
                    A = kneighbors_graph(adata_tmp.obsm[spatial_key],n_neighbors = n_neighbors)
                    edge_index_tmp, edge_weight = from_scipy_sparse_matrix(A)
                    label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                    edge_index_tmp = label[edge_index_tmp]
                    if j == 0:
                        edge_index = edge_index_tmp
                        j = 1
                    else:
                        edge_index = torch.cat((edge_index,edge_index_tmp),1)

                else:
                    tri = Delaunay(adata_tmp.obsm[spatial_key])
                    triangles = tri.simplices
                    edges = set()
                    for triangle in triangles:
                        for i in range(3):
                            edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                            edges.add(edge)
                    edge_index_tmp = torch.tensor(list(edges)).t().contiguous()
                    label = torch.arange(adata.shape[0])[adata.obs_names.isin(adata_tmp.obs_names)]
                    edge_index_tmp = label[edge_index_tmp]
                    if j == 0:
                        edge_index = edge_index_tmp
                        j = 1
                    else:
                        edge_index = torch.cat((edge_index,edge_index_tmp),1)
        else:
            if method == 'knn':
                A = kneighbors_graph(adata.obsm[spatial_key],n_neighbors = n_neighbors)
                edge_index, edge_weight = from_scipy_sparse_matrix(A)
            else:
                tri = Delaunay(adata.obsm[spatial_key])
                triangles = tri.simplices
                edges = set()
                for triangle in triangles:
                    for i in range(3):
                        edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                        edges.add(edge)
                edge_index = torch.tensor(list(edges)).t().contiguous()

        return edge_index


    @torch.no_grad()
    def get_latent_representation(
        self,
        edge_index,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "all",
    ) -> np.ndarray:
        """
        Return the latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to full batch training.
        representation_kind: "intrinsic", "interaction" or "all" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["intrinsic", "interaction","all"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        data = AnnTorchDataset(self.adata_manager)
        outputs = self.module.inference(data[np.arange(data.get_data('X').shape[0])],edge_index)
        latent = []
        if representation_kind == "intrinsic":
            latent_m = outputs["q_m"]
            latent_sample = outputs["z"]
        elif representation_kind == "interaction":
            latent_m = outputs["qgat_m"]
            latent_sample = outputs["z_gat"]

        elif representation_kind == "all":
            latent_m = outputs["qall_m"]
            latent_sample = outputs["z_all"]

        if give_mean:
            latent_sample = latent_m

        latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()
    
    
    @torch.no_grad()
    def get_decoded_expression(
        self,
        edge_index,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "all",
    ) -> np.ndarray:
        """
        Return decoded expression for each cell.
        Depracated
        """
        available_representation_kinds = ["intrinsic", "interaction","all"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        data = AnnTorchDataset(self.adata_manager)
        outputs = self.module.inference(data[np.arange(data.get_data('X').shape[0])],edge_index)
        
        decoded = []
        if representation_kind == "intrinsic":
            decoded_m = self.module._generic_generative(outputs["q_m"],outputs["library"],outputs["batch_index"])["px_scale"]
            decoded_sample = self.module._generic_generative(outputs["z"],outputs["library"],outputs["batch_index"])["px_scale"]
            
        elif representation_kind == "interaction":
            decoded_m = self.module._generic_generative(outputs["qgat_m"],outputs["library"],outputs["batch_index"])["px_scale"]
            decoded_sample = self.module._generic_generative(outputs["z_gat"],outputs["library"],outputs["batch_index"])["px_scale"]

        elif representation_kind == "all":
            decoded_m = self.module._generic_generative(outputs["qall_m"],outputs["library"],outputs["batch_index"])["px_scale"]
            decoded_sample = self.module._generic_generative(outputs["z_all"],outputs["library"],outputs["batch_index"])["px_scale"]
        
        if give_mean:
            decoded_sample = decoded_m
            
        decoded += [decoded_sample.detach().cpu()]
        return torch.cat(decoded).numpy()
    
    @torch.no_grad()
    def get_spatial_effect(
        self,
        edge_index,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        mode: str = "original",
        n_neighbors = 20,
    ) -> np.ndarray:
        """
        Return the spatial effect for each cell. 

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to full batch training.
        mode: "original" or "reconstructed" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_genes)`.
        """
        available_modes = ["original", "reconstructed"]
        assert mode in available_modes, (
            f"mode = {mode} is not one of"
            f" {available_modes}"
        )
        
        data = AnnTorchDataset(self.adata_manager)
        outputs = self.module.inference(data[np.arange(data.get_data('X').shape[0])],edge_index)
        if mode == 'original':
            embedding = outputs["q_m"].numpy()
            A = kneighbors_graph(embedding,n_neighbors = n_neighbors)
            adata_tmp = sc.AnnData(data[np.arange(data.get_data('X').shape[0])]["X"])
            sc.pp.normalize_total(adata_tmp)
            sc.pp.log1p(adata_tmp)
            decoded = adata_tmp.X - (A.toarray() @ adata_tmp.X / n_neighbors)
            return decoded
        else:
            if give_mean:
                decoded_m = self.module._generic_generative(outputs["qall_m"],outputs["library"],outputs["batch_index"])["px_scale"]
                decoded_null = self.module._generic_generative(torch.cat((outputs["q_m"],outputs["qgat_m"]*0+outputs["qgat_m"].mean(axis=0)[None,:]),1),outputs["library"],outputs["batch_index"])["px_scale"]
            else:
                decoded_m = self.module._generic_generative(outputs["z_all"],outputs["library"],outputs["batch_index"])["px_scale"]
                decoded_null = self.module._generic_generative(torch.cat((outputs["z"],outputs["z_gat"]*0+outputs["z_gat"].mean(axis=0)[None,:]),1),outputs["library"],outputs["batch_index"])["px_scale"]
            decoded_sample = decoded_m - decoded_null
            decoded = []
            decoded += [decoded_sample.detach().cpu()]
            return torch.cat(decoded).numpy()
    
    @torch.no_grad()
    def get_attention(
        self,
        edge_index
    ) -> np.ndarray:
        data = AnnTorchDataset(self.adata_manager)
        inference_input = self.module._get_inference_input_from_concat_tensors(data[np.arange(data.get_data('X').shape[0])])
        q = self.module.base_encoder.encoder(inference_input["x"], inference_input["batch_index"])
        q_l = self.module.gat_mean.lin_l(q)
        q_r = self.module.gat_mean.lin_r(q)
        x = q_l.view((q_l.shape[0],1,q_l.shape[1])) + q_r.view((1,q_r.shape[0],q_r.shape[1]))
        x = F.leaky_relu(x, self.module.gat_mean.negative_slope)
        alpha = (x * self.module.gat_mean.att).sum(dim=-1)
        alpha = alpha * to_dense_adj(edge_index).squeeze()
        sm = torch.nn.Softmax(dim=0)
        for i in range(alpha.shape[1]):
            tmp = (alpha[:,i] != 0)
            alpha[tmp,i] = sm(alpha[tmp,i])
        return alpha

    
    @torch.no_grad()
    def counterfactual_expression(
        self,
        indices,
        adata: Optional[AnnData] = None,
        give_mean: bool = True,
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Answers the what-if question: What the cell expression would be if it is in another position?
        Depracated
        """
        data = AnnTorchDataset(self.adata_manager)
        result_list = []
        for i in range(len(indices)):
            data_full = data[np.arange(data.get_data('X').shape[0])].copy()
            for key in data_full.keys():
                data_full[key][indices[i][1]] = data_full[key][indices[i][0]].copy()
            for k in range(n_samples):
                latent_dict = self.module.inference(data_full,edge_index)
                latent_dict_masked = {}
                for key, value in latent_dict.items():
                    if value is None:
                        latent_dict_masked[key] = None
                    else:
                        latent_dict_masked[key] = value[indices[i][1]]
                decoder_dict = self.module.generative(latent_dict_masked).numpy()
                if k == 0:
                    ex = decoder_dict["px_r"]
                else:
                    ex = torch.cat((ex,decoder_dict["px_r"]),0)
            result_list.append(ex.detach().cpu().numpy())
        return result_list


    def train(
        self,
        edge_index: torch.Tensor,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        lr = 1e-3,
        weight_decay = 1e-4,
    ) -> None:
        """
        Train the SIMVI model. In our setting, we consider full-batch training, therefore
        we rewrite the training function. 

        Args:
        ----
        
            max_epochs: Number of passes through the dataset. If `None`, default to
                `np.min([round((20000 / n_cells) * 400), 400])`.
            use_gpu: Use default GPU if available (if `None` or `True`), or index of
                GPU to use (if `int`), or name of GPU (if `str`, e.g., `"cuda:0"`),
                or use CPU (if `False`).
            train_size: Size of training set in the range [0.0, 1.0].
            validation_size: Size of the validation set. If `None`, default to
                `1 - train_size`. If `train_size + validation_size < 1`, the remaining
                cells belong to the test set.
            lr: Learning rate.
            weight_decay: L2 regularization strength.

        Returns
        -------
            None. The model is trained.
        """
        if max_epochs is None:
            n_cells = self.adata_manager.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        if validation_size is None:
            validation_size = 1 - train_size

        n_train, n_val = validate_data_split(self.adata_manager.adata.n_obs, train_size, validation_size)
        random_state = np.random.RandomState(seed=settings.seed)
        permutation = random_state.permutation(self.adata_manager.adata.n_obs)
        train_mask = permutation[:n_val]
        val_mask = permutation[n_val : (n_val + n_train)]
        test_mask = permutation[(n_val + n_train) :]

        data = AnnTorchDataset(self.adata_manager)

        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss = []
        val_loss = []
        pbar = tqdm(range(1, max_epochs + 1))
        for epoch in pbar:
            train_loss.append(_train(self.module, data, edge_index, train_mask, optimizer).detach())
            val_loss.append(_eval(self.module, data, edge_index, val_mask).detach())
            pbar.set_description('Epoch '+str(epoch)+'/'+str(max_epochs))
            pbar.set_postfix(train_loss=train_loss[epoch-1].numpy(), val_loss=val_loss[epoch-1].numpy())
            #print('Epoch ',epoch)
        return train_loss, val_loss

def _train(model, data, edge_index, mask, optimizer):
    model.train()
    optimizer.zero_grad()
    latent_dict = model.inference(data[np.arange(data.get_data('X').shape[0])],edge_index)
    #print(latent_dict)
    latent_dict_masked = {}
    for key, value in latent_dict.items():
        if value is None:
            latent_dict_masked[key] = None
        else:
            latent_dict_masked[key] = value[mask]
    decoder_dict = model.generative(latent_dict_masked)
    lossrecorder = model.loss(data[mask], latent_dict_masked, decoder_dict)
    loss = lossrecorder.loss
    loss.backward()
    optimizer.step()
    return loss

def _eval(model, data, edge_index, mask):
    model.eval()
    latent_dict = model.inference(data[np.arange(data.get_data('X').shape[0])],edge_index)
    #print(latent_dict)
    latent_dict_masked = {}
    for key, value in latent_dict.items():
        if value is None:
            latent_dict_masked[key] = None
        else:
            latent_dict_masked[key] = value[mask]
    decoder_dict = model.generative(latent_dict_masked)
    lossrecorder = model.loss(data[mask], latent_dict_masked, decoder_dict)
    return lossrecorder.loss
    
def _prob(loc,scale,value):
    ### diagonal covariance, therefore the density can be decomposed
    var = (scale * scale)
    log_scale = torch.log(scale)
    log_prob = -((value[None,:] - loc) * (value[None,:] - loc)) / (2 * var) - log_scale - torch.log(torch.tensor(math.sqrt(2 * math.pi)))
    return torch.exp(log_prob.sum(1))
        
