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
from torch.nn.utils import clip_grad_value_
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import issparse
from scipy.stats import f
from statsmodels.stats.multitest import fdrcorrection


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
        n_batch: Number of batches. 
        n_hidden: Number of nodes per hidden layer.
        n_intrinsic: Dimensionality of the intrinsic variation.
        n_spatial: Dimensionality of the spatial variation.
        n_layers: Number of decoder layers. Note that in our implementation, encoder is fixed to have two layers.
        lam_mi: Coefficient of the independence regularization term. When using the mmd option, a coefficient of 1000 is recommended. When using the mi option, the value of 5 is recommended.
        reg_to_use: 'mmd' (Maximal Mean Discrepancy) or 'mi' (Closed-form mutual information).
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        permutation rate: the rate of permutation to use in the training. (The permutation step itself is optional)
        var_eps: minimal variance for the variational posteriors.
        kl_weight: The kl divergence coefficient for intrinsic variation.
        kl_gatweight: The kl divergence coefficient for spatial variation.
        attention_heads: the number of attention heads.
    """

    def __init__(
        self,
        adata: AnnData,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_intrinsic: int = 20,
        n_spatial: int = 20,
        n_layers: int = 1,
        dropout_rate: float = 0,
        use_observed_lib_size: bool = True,
        lam_mi: float = 1000,
        reg_to_use: str = 'mmd',
        dis_to_use: str = 'zinb',
        permutation_rate: float = 0.25,
        var_eps: float = 1e-4,
        kl_weight: float = 1,
        kl_gatweight: float = 0.01,
        attention_heads: int = 1,
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
            n_output=n_intrinsic,
            n_spatial=n_spatial,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            lam_mi = lam_mi,
            reg_to_use = reg_to_use,
            dis_to_use = dis_to_use,
            permutation_rate = permutation_rate,
            var_eps = var_eps,
            kl_weight = kl_weight,
            kl_gatweight = kl_gatweight,
            heads = attention_heads,
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
        Set up AnnData instance for SIMVI model. A standard function to call in scvi-tools pipeline.

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
            size_factor_key: Key in `adata.obs` for size factor information. 
            categorical_covariate_keys: Keys in `adata.obs` corresponding to categorical
                data. Not used in SIMVI.
            continuous_covariate_keys: Keys in `adata.obs` corresponding to continuous
                data. Not used in SIMVI.

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
                edge_index = torch.tensor(list(edges)).t().contiguous().type(torch.LongTensor)

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
        available_representation_kinds = ["intrinsic", "interaction","output","all"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data('X').shape[0])]
        for key, value in data.items():
            data[key] = torch.Tensor(value).to(next(self.module.base_encoder.parameters()).device)
        outputs = self.module.inference(data,edge_index,eval_mode=True)
        latent = []
        if representation_kind == "intrinsic":
            latent_m = outputs["q_m"]
            latent_sample = outputs["z"]
        elif representation_kind == "interaction":
            latent_m = outputs["qgat_m"]
            latent_sample = outputs["z_gat"]
        elif representation_kind == "output":
            latent_m = self.module.gat_mean.lin_r(outputs["q_m"][:,-self.module.n_spatial:])
            latent_sample = latent_m
        elif representation_kind == "all":
            latent_m = outputs["qall_m"]
            latent_sample = outputs["z_all"]

        if give_mean:
            latent_sample = latent_m

        latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()
    
    
    
    @torch.no_grad()
    def get_attention(
        self,
        edge_index,
    ) -> np.ndarray:
        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data('X').shape[0])]
        for key, value in data.items():
            data[key] = torch.Tensor(value).to(next(self.module.base_encoder.parameters()).device)
        inference_input = self.module._get_inference_input_from_concat_tensors(data,eval_mode=True)
        q_m, q_v, z = self.module.base_encoder(inference_input["x"], inference_input["batch_index"])
        q_l = self.module.gat_mean.lin_l(q_m[:,-self.module.n_spatial:])
        q_r = self.module.gat_mean.lin_r(q_m[:,-self.module.n_spatial:])
        x = q_l[edge_index[1]] + q_r[edge_index[0]]
        #x = q_l.view((q_l.shape[0],1,q_l.shape[1])) + q_r.view((1,q_r.shape[0],q_r.shape[1]))
        x = F.leaky_relu(x, self.module.gat_mean.negative_slope)
        alpha = (x * self.module.gat_mean.att).sum(dim=-1).squeeze().detach().cpu()
        size = torch.Size([q_l.shape[0], q_l.shape[0]])
        #coo = coo_matrix((alpha, (edge_index.cpu().numpy()[0], indices[1])), shape=shape)

        sparse_matrix = torch.sparse_coo_tensor(edge_index.cpu(), alpha, size)
        sparse_matrix = torch.sparse.softmax(sparse_matrix,dim=0)
        
        indices = sparse_matrix.indices().numpy()
        values = sparse_matrix.values().numpy()
        shape = sparse_matrix.size()

        coo = coo_matrix((values, (indices[0], indices[1])), shape=shape)
        csr = coo.tocsr()

        return csr.T

    @torch.no_grad()
    def get_archetypes(
        self,
        embedding,
        noc=5,
        delta=0.1,
        conv_crit=0.00001,
        maxiter=200,
        verbose=False,
    ) -> np.ndarray:

        from py_pcha import PCHA
        XC, S, C, SSE, varexpl = PCHA(embedding.T, noc=noc, delta=delta,conv_crit=conv_crit, maxiter=maxiter, verbose=verbose)
        
        return XC.T, S, varexpl    
    
    @torch.no_grad()    
    def get_se(
        self,
        edge_index: Optional[torch.Tensor] = None,
        adata: Optional[AnnData] = None,
        z_label: Optional[str] = 'simvi_z',
        s_label: Optional[str] = 'simvi_s',
        transformation = 'log1p',
        batch_label = None,
        num_arch = 5,
        delta = 0.1,
        maxiter = 200,
        Kfold = 5,
        eps = 0,
        thres = 0.95,
        positivity_filter = False,
        cell_type_label = None,
        obsm_label = None,
        mode = 'individual',
    ) -> np.ndarray:
        """
        Return the spatial effect for each cell in spatial omics data. Requires training the SIMVI model in priori.

        Args:
        ----
        edge_index: The object created by function "extract_edge_index".
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        z_label: the name of the intrinsic variation in adata.obsm. If adata is `None`, then it is calculated in this function.
        s_label: the name of the spatial variation in adata.obsm. If adata is `None`, then it is calculated in this function.
        transformation: If `log1p`, perform log1p on a copy of the data. Else, operate on the given adata.X.
        batch_label: If given, then add it as a covariate in the double machine learning model.
        num_arch, delta, maxiter, Kfold, eps: parameters of archetypal transformation.
        thres: Thres2 in positivity index calculation.
        positivity_filter: If True, only return the spatial effect of cells satisfying positivity condition, and return the indices of these celles.
        cell_type_label: If given, then add it as a covariate in the double machine learning model.
        obsm_label: If given, then add it as a covariate in the double machine learning model.
        Returns
        -------
            If positivity is `False`, return spatial effect, R2s, p-values, and archetypes. Else, additionally return indices satisfying the positivity condition. 
        """
        ## If adata is not provided, infer the latent space first. Else, use the existing variations
        
        if adata is None:
            data = AnnTorchDataset(self.adata_manager)
            data = data[np.arange(data.get_data('X').shape[0])]
            for key, value in data.items():
                data[key] = torch.Tensor(value).to(next(self.module.base_encoder.parameters()).device)
            outputs = self.module.inference(data,edge_index,eval_mode=True)
        
            batch_index = outputs["batch_index"]
            latent_z = outputs["q_m"].detach().cpu().numpy()
            latent_s = outputs["qgat_m"].detach().cpu().numpy()
        
            adata_tmp = self.adata_manager.adata.copy()
            
        else:
            adata_tmp = adata.copy()
            
            latent_z = adata_tmp.obsm[z_label]
            latent_s = adata_tmp.obsm[s_label]
            
        ## estimate generalized propensity score
        if transformation == 'log1p':
            sc.pp.normalize_total(adata_tmp)
            sc.pp.log1p(adata_tmp)
            
        if batch_label is not None:
            df = pd.get_dummies(adata_tmp.obs[batch_label]).values
            latent_z = np.hstack((latent_z,df))
            
        if cell_type_label is not None:
            df2 = pd.get_dummies(adata_tmp.obs[cell_type_label]).values
            latent_z = np.hstack((latent_z,df2))
            #df2_ = df2 / df2.sum(axis=0) 
            
        if obsm_label is not None:
            df3 = np.asarray(adata_tmp.obsm[obsm_label])
            latent_z = np.hstack((latent_z,df3))

        if issparse(adata_tmp.X):
            adata_tmp.X = adata_tmp.X.toarray()
            
        arc, S, varexpl = self.get_archetypes(latent_s,noc=num_arch,delta=delta,maxiter=maxiter)
        
        ## S is the continuous treatment variable
        S = np.asarray(S.T)
        
        if positivity_filter:
            sc.pp.neighbors(adata_tmp,use_rep=z_label)
            sc.tl.leiden(adata_tmp,resolution=0.6)
            df = pd.DataFrame(S.copy())
            df = (df>0.5).astype(int)
            df['cluster'] = adata_tmp.obs['leiden'].values.copy()
            df = df.loc[df.sum(axis=1)>0]
            df_ = pd.DataFrame(df.groupby('cluster').mean().max(axis=0))
            
            positive_indices = ((pd.DataFrame(S.copy(),index=adata_tmp.obs_names).loc[:,df_.values>=thres]>0.5).sum(axis=1) == 0)
            adata_tmp = adata_tmp[positive_indices]
            latent_z = latent_z[positive_indices]
            latent_s = latent_s[positive_indices]
            S = S[positive_indices]
            S = S[:,df_.values.flatten()<thres]
        
        np.random.seed(42)
        indices = np.random.permutation(latent_z.shape[0])
        split_data = np.array_split(indices, Kfold)        
        
        
        
        if mode == 'individual':     
            se_list = []
            r2_zlist = []
            r2_slist = []
            r2_zpvlist = []
            r2_spvlist = []
            for i in range(S.shape[1]):
                se = np.zeros((latent_z.shape[0],adata_tmp.shape[1]))
                r2_z = np.zeros(adata_tmp.shape[1])
                r2_s = np.zeros(adata_tmp.shape[1])
                for ind in split_data:
                    Si = S[ind,i]
                    lr = LinearRegression()
                    lr.fit(latent_z[ind],Si)
                    lr2 = LinearRegression()
                    lr2.fit(latent_z[ind],adata_tmp.X[ind])
                    Si_ = Si - lr.predict(latent_z[ind])
                    X_ = adata_tmp.X[ind] - lr2.predict(latent_z[ind])
                    lr3 = LinearRegression()
                    lr3.fit(latent_z[ind],X_ / (Si_[:,None]+eps),sample_weight = Si_ ** 2)
                    se[ind] = lr3.predict(latent_z[ind]) * Si_[:,None]
                    r2_z = r2_z + r2_score(adata_tmp.X[ind],lr2.predict(latent_z[ind]), multioutput='raw_values')
                    r2_s = r2_s + r2_score(X_, lr3.predict(latent_z[ind]) * Si_[:,None], multioutput='raw_values')
                
            #if cell_type_label is not None:
                
            #    ct_means = pd.DataFrame(se)
            #    ct_means[cell_type_label] = adata_tmp.obs[cell_type_label].values.copy()
            #    ct_means = ct_means.groupby(cell_type_label).median()
            #    #pd.DataFrame(df2_.T @ se, index = pd.get_dummies(adata_tmp.obs[cell_type_label]).columns.copy())
            #    se = se - ct_means.loc[adata_tmp.obs[cell_type_label].values].values
                
                
                se_list.append(se)
                r2_zlist.append(r2_z/Kfold)
                r2_slist.append(r2_s/Kfold)
                r2_zpvlist.append(return_f_pv(adata_tmp.X[ind],r2_z/Kfold))
                r2_spvlist.append(return_f_pv(latent_z[ind],r2_s/Kfold))
            if positivity_filter:
                return positive_indices, se_list, r2_zlist, r2_slist, r2_zpvlist, r2_spvlist, S
            else:
                return se_list, r2_zlist, r2_slist, r2_zpvlist, r2_spvlist, S
        else:
            se = np.zeros((latent_z.shape[0],adata_tmp.shape[1]))
            r2_z = np.zeros(adata_tmp.shape[1])
            r2_s = np.zeros(adata_tmp.shape[1])
            for ind in split_data:
                lr = LinearRegression()
                lr.fit(latent_z[ind],S[ind])
                lr2 = LinearRegression()
                lr2.fit(latent_z[ind],adata_tmp.X[ind])
                S_ = S[ind] - lr.predict(latent_z[ind])
                X_ = adata_tmp.X[ind] - lr2.predict(latent_z[ind])
                lr3 = LinearRegression()
                design_matrix = (latent_z[ind][:,:,None] * S_[:,None,:]).reshape(latent_z[ind].shape[0],latent_z.shape[1] * S_.shape[1])
                design_matrix = np.hstack((design_matrix,S_))
                lr3.fit(design_matrix,X_)
                se[ind] = lr3.predict(design_matrix)
                r2_z = r2_z + r2_score(adata_tmp.X[ind],lr2.predict(latent_z[ind]), multioutput='raw_values')
                r2_s = r2_s + r2_score(X_, lr3.predict(design_matrix), multioutput='raw_values')
            if positivity_filter:
                return positive_indices, se, r2_z/Kfold, r2_s/Kfold, return_f_pv(adata_tmp.X[ind],r2_z/Kfold), return_f_pv(adata_tmp.X[ind],r2_s/Kfold), S
            else:
                return se, r2_z/Kfold, r2_s/Kfold, return_f_pv(adata_tmp.X[ind],r2_z/Kfold), return_f_pv(adata_tmp.X[ind],r2_s/Kfold), S
                


    def train(
        self,
        edge_index: torch.Tensor,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        batch_size: Optional[int] = None,
        anneal_epochs: int = 50,
        mae_epochs: int = 80,
        validation_size: Optional[float] = None,
        lr = 1e-3,
        weight_decay = 1e-4,
        device = None,
    ) -> None:
        """
        Train the SIMVI model. In our setting, we consider full-batch training, therefore
        we rewrite the training function. 

        Args:
        ----
            edge_index: tensor returned by model.extract_edge_index.
            max_epochs: Number of passes through the dataset. If `None`, default to
                `np.min([round((20000 / n_cells) * 400), 400])`.
            use_gpu: Use default GPU if available (if `None` or `True`), or index of
                GPU to use (if `int`), or name of GPU (if `str`, e.g., `"cuda:0"`),
                or use CPU (if `False`).
            train_size: Size of training set in the range [0.0, 1.0].
            batch_size: Mini-batch size to use during training.
            anneal_epochs: The number of epoches that use KL annealing.
            mae_epochs: The number of epoches that corrupts input data.
            validation_size: Size of the validation set. If `None`, default to
                `1 - train_size`. If `train_size + validation_size < 1`, the remaining
                cells belong to the test set.
            lr: learning rate. Default 1e-3.
            weight_decay: weight decay (serve as l2 regularization). Default 1e-4.
            device: The GPU to train the model on. If none, use torch.device("cuda") or cpu.

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
        train_mask = permutation[:n_train]
        val_mask = permutation[n_train : (n_val + n_train)]
        test_mask = permutation[(n_val + n_train) :]
        if device is None:
            if use_gpu & torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = 'cpu'
        self.module = self.module.to(device)
        edge_index = edge_index.to(device)

        data = AnnTorchDataset(self.adata_manager)
        data = data[np.arange(data.get_data('X').shape[0])]
        for key, value in data.items():
            data[key] = torch.Tensor(value).to(device)

        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss = []
        val_loss = []
        pbar = tqdm(range(1, max_epochs + 1))
            
        if batch_size is not None:
            batch_indices = [train_mask[i:i + batch_size] for i in range(0, train_mask.shape[0], batch_size)]
            train_loader = {}
            for i, batch_index in enumerate(batch_indices):
                data_masked = {}
                for key, value in data.items():
                    if value is None:
                        data_masked[key] = None
                    else:
                        data_masked[key] = value[batch_index]

                train_loader[i] = data_masked
        else:
            data_masked = {}
            for key, value in data.items():
                if value is None:
                    data_masked[key] = None
                else:
                    data_masked[key] = value[train_mask]
            train_loader = data_masked
            
        val_loader = {}
        for key, value in data.items():
            if value is None:
                val_loader[key] = None
            else:
                val_loader[key] = value[val_mask]

        for epoch in pbar:
            weight = min(1.0, epoch / anneal_epochs)
            if epoch < mae_epochs:
                eval_mode = False
            else:
                eval_mode = True
            train_loss.append(_train(self.module, data, edge_index, train_mask, train_loader, optimizer, batch_size, weight, eval_mode))
            val_loss.append(_eval(self.module, data, edge_index, val_mask,val_loader, weight))
            pbar.set_description('Epoch '+str(epoch)+'/'+str(max_epochs))
            pbar.set_postfix(train_loss=train_loss[epoch-1], val_loss=val_loss[epoch-1].numpy())

        return train_loss, val_loss

def _train(model, data, edge_index, mask, train_loader, optimizer, batch_size, weight, eval_mode):
    train_loss = []
    model.train()
    #print(latent_dict)
    if batch_size is None:
        optimizer.zero_grad()
        latent_dict = model.inference(data,edge_index,eval_mode=eval_mode)
        latent_dict_masked = {}
        for key, value in latent_dict.items():
            if value is None:
                latent_dict_masked[key] = None
            else:
                latent_dict_masked[key] = value[mask]

        decoder_dict = model.generative(latent_dict_masked)
        lossrecorder = model.loss(train_loader, latent_dict_masked, decoder_dict, weight)
        loss = lossrecorder.loss
        loss.backward()
        #clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()
        train_loss.append(loss.detach())
    else:
        batch_indices = [mask[i:i + batch_size] for i in range(0, mask.shape[0], batch_size)]
        
        for i, batch_index in enumerate(batch_indices):
            optimizer.zero_grad()
            latent_dict = model.inference(data,edge_index,eval_mode=eval_mode)
            latent_dict_masked = {}
            data_masked = {}
            for key, value in latent_dict.items():
                if value is None:
                    latent_dict_masked[key] = None
                else:
                    latent_dict_masked[key] = value[batch_index]

            decoder_dict = model.generative(latent_dict_masked)
            lossrecorder = model.loss(train_loader[i], latent_dict_masked, decoder_dict, weight)
            loss = lossrecorder.loss
            loss.backward()
            #clip_grad_value_(model.parameters(), clip_value=1)
            optimizer.step()
            train_loss.append(loss.detach().cpu())
    return np.array(train_loss).mean()

def _eval(model, data, edge_index, mask,val_loader, weight):
    model.eval()
    latent_dict = model.inference(data,edge_index,eval_mode=True)
    #print(latent_dict)
    latent_dict_masked = {}
    for key, value in latent_dict.items():
        if value is None:
            latent_dict_masked[key] = None
        else:
            latent_dict_masked[key] = value[mask]
            
    decoder_dict = model.generative(latent_dict_masked)
    lossrecorder = model.loss(val_loader, latent_dict_masked, decoder_dict, weight)
    return lossrecorder.loss.detach().cpu()
    
def _prob(loc,scale,value):
    ### diagonal covariance, therefore the density can be decomposed
    var = (scale * scale)
    log_scale = torch.log(scale)
    log_prob = -((value[None,:] - loc) * (value[None,:] - loc)) / (2 * var) - log_scale - torch.log(torch.tensor(math.sqrt(2 * math.pi)))
    return torch.exp(log_prob.sum(1))

def get_f_pv(X,y,lr):
    N = X.shape[0]
    K = X.shape[1] + 1
    Rsq = r2_score(y,lr.predict(X), multioutput='raw_values')
    fstat = (Rsq/(1-Rsq))*((N-K-1)/K)

    df_model = X.shape[1]
    df_residuals = X.shape[0] - X.shape[1] + 1

    p_values = f.sf(fstat, df_model, df_residuals)
    
    rej, adj_p = fdrcorrection(p_values)
    
    return adj_p

def return_f_pv(X,Rsq):
    N = X.shape[0]
    K = X.shape[1] + 1
    fstat = (Rsq/(1-Rsq))*((N-K-1)/K)

    df_model = X.shape[1]
    df_residuals = X.shape[0] - X.shape[1] + 1
    p_values = f.sf(fstat, df_model, df_residuals)
    
    rej, adj_p = fdrcorrection(p_values)
    return adj_p
        
