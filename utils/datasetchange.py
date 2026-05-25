# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:53:35 2023

@author: ab978
"""

from torch_geometric.data import InMemoryDataset, HeteroData, Data
import numpy as np
import os
import torch
import h5py, pdb
from natsort import natsorted
from tqdm import tqdm
import pandas as pd
import json
import sys

# reshape vs transpose https://discuss.pytorch.org/t/different-between-permute-transpose-view-which-should-i-use/32916

class CovarianceTemporalDataset(InMemoryDataset):
    '''This is the first dataset I implemented where the structure of the data seems not to be the one that the 
    GATConv layer is expected. '''
    def __init__(self, hdf5_file, root='processed_data/cached_datasets_temporal/', transform=None, pre_transform=None, seq_length=None):
        self.hdf5_file = hdf5_file
        self.root = root
        self.seq_length = seq_length
        super(CovarianceTemporalDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return [os.path.basename(self.hdf5_file)]

    @property
    def processed_file_names(self):
        return ['data_temp.pt']
    
    @property
    def processed_dir(self):
        return self.root
    
    def process(self):
        data_list = []
    
        # Load covariance matrices from the HDF5 file
        with h5py.File(self.hdf5_file, 'r') as f:
            keys = list(f.keys())
            for i in range(len(f) - self.seq_length):
                seq_data_list = []
                for j in range(self.seq_length):
                    cov_matrix = np.array(f[keys[i+j]])
                    next_cov_matrix  = np.array(f[keys[i+j+1]])
                    
                    # Create the adjacency matrix from the covariance matrix
                    adj_matrix = cov_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the adjacency matrix (excluding diagonal)
                    mask = np.triu(np.ones_like(adj_matrix), k=1) > 0
                    edge_weights = torch.tensor(adj_matrix[mask], dtype=torch.float)
    
                    # Create edge_index tensor
                    edge_index = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()
    
                    # Extract the variances (diagonal) as node features
                    node_features = np.diag(cov_matrix)
                    x = torch.tensor(node_features, dtype=torch.float).view(-1, 1)
    
                    # Process the next covariance matrix to create target values
                    adj_matrix_next = next_cov_matrix.copy()
                    np.fill_diagonal(adj_matrix_next, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the next adjacency matrix (excluding diagonal)
                    mask = np.triu(np.ones_like(adj_matrix_next), k=1) > 0
                    y_edge_weight = torch.tensor(adj_matrix_next[mask], dtype=torch.float).view(-1, 1)
    
                    # Extract the variances (diagonal) of the next covariance matrix as target node features
                    y_x = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float).view(-1, 1)
                    

                    # Create PyTorch Geometric Data object
                    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights,
                                y_edge_weight=y_edge_weight, y_x=y_x)
                    
                    seq_data_list.append(data)

                    

                # Combine the sequence of Data objects into a single object with the desired format
                # seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=0),
                #                 edge_index=torch.stack([d.edge_index for d in seq_data_list], dim=0),
                #                 edge_weight=torch.stack([d.edge_weight for d in seq_data_list], dim=0),
                #                 y_edge_weight=torch.stack([d.y_edge_weight for d in seq_data_list], dim=0),
                #                 y_x=torch.stack([d.y_x for d in seq_data_list], dim=0))

                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=0).transpose(0,1),
                                edge_index=torch.stack([d.edge_index for d in seq_data_list], dim=0).transpose(0,1),
                                edge_weight=torch.stack([d.edge_weight for d in seq_data_list], dim=0).transpose(0,1),
                                y_edge_weight=torch.stack([d.y_edge_weight for d in seq_data_list], dim=0).transpose(0,1),
                                y_x=torch.stack([d.y_x for d in seq_data_list], dim=0).transpose(0,1))
                
                

                data_list.append(seq_data)

        
        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(os.getcwd(),self.processed_paths[0]))
        
class CovarianceLSTMDataset():
    def __init__(self, hdf5_file1, hdf5_file2, root='processed_data/cached_datasets_lagged/', seq_length=None):
        self.hdf5_file1 = hdf5_file1
        self.hdf5_file2 = hdf5_file2
        self.root = root
        self.seq_length = seq_length

    def process(self):
        x_matrices = []
        y_x_vectors = []
        with h5py.File(self.hdf5_file1, 'r') as f1, h5py.File(self.hdf5_file2, 'r') as f2:
            keys = list(f1.keys())
            keys = natsorted(keys)
            for i in tqdm(iterable=range(len(f1) - self.seq_length),desc='Creating LSTM dataset...'):
                seq_data_list = []
                for j in range(self.seq_length):
                    cov_matrix = np.array(f1[keys[i+j]])
                    covol_matrix = np.array(f2[keys[i+j]])

                    # Extracting vols (diagonal of cov_matrix)
                    vols = np.diag(cov_matrix)
                    # Extracting covols (unique values outside the diagonals of cov_matrix)
                    covols = cov_matrix[np.triu_indices(cov_matrix.shape[0], k=1)]
                    # Extracting volvol (diagonal of covol_matrix)
                    volvol = np.diag(covol_matrix)
                    # Extracting covolvols (unique values outside the diagonals in covol_matrix)
                    covolvols = covol_matrix[np.triu_indices(covol_matrix.shape[0], k=1)]

                    # Concatenating all the features
                    features = np.concatenate([vols, covols, volvol, covolvols])
                    x = torch.tensor(features, dtype=torch.float)

                    # Assuming the target is the diagonal of the next covariance matrix
                    next_cov_matrix = np.array(f1[keys[i+j+1]])
                    y_x = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float)

                    seq_data_list.append((x, y_x))

                x_matrix = torch.stack([data[0] for data in seq_data_list], dim=0).numpy()
                y_x_vector = seq_data_list[-1][-1].numpy()
                x_matrices.append(x_matrix)
                y_x_vectors.append(y_x_vector)

        # Ensure the directory exists
        os.makedirs(self.root, exist_ok=True)
    
        # Saving the entire dataset as .npy files in the specified directory
        np.save(os.path.join(self.root, 'x_matrices.npy'), x_matrices[:])
        np.save(os.path.join(self.root, 'y_x_vectors.npy'), y_x_vectors[:])
       

class FinalHeteroDataset(InMemoryDataset):
    """
    Dataset class used to build heterogeneous graph sequences.

    Additional option:
    - include_energy (bool): whether to include energy nodes together with
      their related features and edges. If False, the dataset only contains
      stock nodes and stock-stock relations.
    """
    
    def __init__(self, hdf5_file_vol, hdf5_file_volvol, stock_har_rv_folder, 
                 energy_har_rv_folder, stock_energy_corr_folder, energy_energy_corr_folder, 
                 node_info_file, root, transform=None, pre_transform=None, 
                 seq_length=15, intraday_points=3, 
                 include_energy=True):  # Additional switch for energy nodes.
        
        # --- Basic settings ---
        self.hdf5_file_vol = hdf5_file_vol
        self.hdf5_file_volvol = hdf5_file_volvol
        self.stock_har_rv_folder = stock_har_rv_folder
        self.node_info_file = node_info_file
        self.intraday_points = intraday_points
        self.seq_length = seq_length
        self.include_energy = include_energy  # Save the switch state.

        # Configure energy-related paths conditionally based on include_energy.
        if self.include_energy:
            self.energy_har_rv_folder = energy_har_rv_folder
            self.stock_energy_corr_folder = stock_energy_corr_folder
            self.energy_energy_corr_folder = energy_energy_corr_folder
        else:
            print("--- [Config] Energy nodes are disabled (include_energy=False) ---")
            # Set these paths to None so the loading steps can be skipped later.
            self.energy_har_rv_folder = None
            self.stock_energy_corr_folder = None
            self.energy_energy_corr_folder = None
            
        # --- Load node metadata ---
        with open(node_info_file, 'r', encoding='utf-8') as f:
            self.node_info = json.load(f)
        
        self.stock_ids = self.node_info['stock_ids']
        
        # Load energy IDs conditionally.
        self.energy_ids = self.node_info['energy_ids'] if self.include_energy else []
        
        # node_order must remain complete because it defines the index order
        # inside the HDF5 matrices.
        self.node_order = self.node_info['node_order'] 
        
        self.num_stocks = len(self.stock_ids)
        self.num_energy = len(self.energy_ids)  # This becomes 0 when include_energy=False.
        self.num_nodes = len(self.node_order)  # Keep the total node count unchanged.

        # --- Map nodes to global indices ---
        self.node_to_idx = {name: i for i, name in enumerate(self.node_order)}
        
        # Global indices of stock nodes in the HDF5 matrices.
        self.stock_indices = torch.tensor([self.node_to_idx[sid] for sid in self.stock_ids], dtype=torch.long)
        
        # Set energy indices conditionally.
        if self.include_energy:
            # Global indices of energy nodes in the HDF5 matrices.
            self.energy_indices = torch.tensor([self.node_to_idx[eid] for eid in self.energy_ids], dtype=torch.long)
        else:
            self.energy_indices = torch.tensor([], dtype=torch.long)  # Use an empty tensor.

        # --- Initialize the parent class ---
        super().__init__(root, transform, pre_transform)
        
        # --- Load cache if it already exists ---
        if os.path.exists(self.processed_paths[0]):
            print(f"--- Loading data from cache directory '{self.processed_dir}'... ---")
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("--- Cached data loaded successfully. ---")

    @property
    def raw_file_names(self):
        # Assume that all required files already exist in the provided folders
        # and HDF5 paths.
        return []

    @property
    def processed_file_names(self):
        # Name of the cache file created by process().
        return ['data.pt']

    def _load_features_to_dict(self, folder_path, file_ids):
        """
        Helper function that loads feature files from a folder into a dict.
        If folder_path is None (for include_energy=False), loading is skipped.
        """
        # Return an empty dict directly when the path is None or missing.
        if folder_path is None or not os.path.exists(folder_path):
            return {}
            
        feature_dict = {}
        for asset_id in file_ids:
            try:
                # Find the file whose name starts with asset_id.
                fpath = next(os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith(asset_id))
                df = pd.read_csv(fpath, header=None, index_col=0, parse_dates=True)
                feature_dict[asset_id] = df
            except StopIteration:
                print(f"Warning: could not find a file for asset {asset_id} in folder {folder_path}.")
        return feature_dict

    def process(self):
        """
        Core data processing method.
        It checks self.include_energy to decide whether energy data should be processed.
        """
        print("\n==========================================================")
        print("====== Running process() to build the final cache... ======")
        if not self.include_energy:  # Print the active mode.
             print("====== [Mode] Stock-only processing (include_energy=False) ======")
        print("==========================================================")
        
        # --- 1. Load node features (HAR-RV) ---
        
        # Load energy features conditionally.
        all_har_features = {
            **self._load_features_to_dict(self.stock_har_rv_folder, self.stock_ids)
        }
        if self.include_energy:
            all_har_features.update(self._load_features_to_dict(self.energy_har_rv_folder, self.energy_ids))
        
        # Raise an error if no features were loaded.
        if not all_har_features:
            raise FileNotFoundError("No HAR-RV feature files were found in the specified paths.")
            
        # Use the first loaded feature file to determine valid trading dates.
        valid_dates = next(iter(all_har_features.values())).index
        
        # --- 2. Load edge features (correlations) ---
        corr_df_dict = {}
        
        # Include energy-related correlation folders only when needed.
        all_corr_folders = []
        if self.include_energy:
            # These correlations are only needed when energy nodes are included.
            all_corr_folders = [self.stock_energy_corr_folder, self.energy_energy_corr_folder]
        
        for folder in all_corr_folders:
            if not os.path.exists(folder): continue
            for pair_file in os.listdir(folder):
                if pair_file.endswith('.csv'):
                    pair_name = os.path.splitext(pair_file)[0]
                    path = os.path.join(folder, pair_file)
                    corr_values = pd.read_csv(path, header=None).values.flatten()
                    # Make sure correlation data are aligned with the date index.
                    if len(corr_values) == len(valid_dates):
                        corr_df_dict[pair_name] = pd.Series(corr_values, index=valid_dates)
        
        # Merge all correlation series into one DataFrame.
        all_corr_df = pd.DataFrame(corr_df_dict)

        # --- 3. Build the high-frequency timestamp mapping ---
        # all_hf_steps stores tuples of (global HDF5 index, corresponding date).
        all_hf_steps = []
        for date in valid_dates:
            day_start_index = valid_dates.get_loc(date) * self.intraday_points
            for point_idx in range(self.intraday_points):
                all_hf_steps.append((day_start_index + point_idx, date))
        
        data_list = []  # Store all graph samples (HeteroData).
        skipped_samples_count = 0
        
        # Define relations conditionally based on whether energy nodes are used.
        if self.include_energy:
            # Full mode: include all three relation types.
            relations = [('stock', 'to', 'stock'), ('stock', 'to', 'energy'), 
                         ('energy', 'to', 'stock')]
        else:
            # Stock-only mode: keep stock-stock relations only.
            relations = [('stock', 'to', 'stock')]
        
        # --- 4. Iterate over the timeline and build graph samples ---
        with h5py.File(self.hdf5_file_vol, 'r') as f_vol, h5py.File(self.hdf5_file_volvol, 'r') as f_volvol:
            
            # We need seq_length inputs and at least 1+intraday_points future
            # observations to construct the labels.
            iterator_range = len(all_hf_steps) - self.seq_length - self.intraday_points
            
            for i in tqdm(range(iterator_range), desc="Building heterogeneous graph sequences"):
                
                seq_data_list = []  # Store all timestep graphs within one sequence.
                is_valid_sequence = True
                
                # Length used for energy-node placeholders. It is determined
                # when the first stock node is processed.
                feature_len_per_step = -1 

                # --- 4a. Build features for T=1...seq_length ---
                for j in range(self.seq_length):
                    hf_global_idx, current_date = all_hf_steps[i + j]
                    
                    try:
                        # Load the high-frequency covariance matrices from HDF5.
                        cov_matrix = np.array(f_vol[str(hf_global_idx)])
                        volvol_matrix = np.array(f_volvol[str(hf_global_idx)])
                    except KeyError:
                        # Skip the whole sequence if this timestep is missing.
                        is_valid_sequence = False; break
                        
                    data = HeteroData()
                    
                    # --- Build node features for the current timestep t ---
                    node_features_t = []  # Temporary list, stored in node_order.
                    
                    for node_idx, node_id in enumerate(self.node_order):
                        
                        if node_id in self.stock_ids:
                            # --- A. Process stock nodes ---
                            f_har = all_har_features[node_id].loc[current_date].values
                            f_vi = np.array([cov_matrix[node_idx, node_idx]])  # Own volatility.
                            
                            # Stock-stock covariance, excluding the node itself.
                            stock_local_idx = self.stock_ids.index(node_id)
                            f_cij_ss = np.delete(cov_matrix[node_idx, self.stock_indices], stock_local_idx)
                            
                            features_to_concat = [f_har, f_vi, f_cij_ss]  # Base features.
                            
                            # Add stock-energy correlation features conditionally.
                            if self.include_energy:
                                corr_values_se = [ all_corr_df.get(f"{node_id}_{eid}", all_corr_df.get(f"{eid}_{node_id}", pd.Series([0])))[current_date] for eid in self.energy_ids ]
                                f_corr_se = np.array(corr_values_se)
                                features_to_concat.append(f_corr_se)
                            
                            final_features = np.concatenate(features_to_concat)
                            node_features_t.append(final_features)
                            
                            if feature_len_per_step == -1:  # Record feature length.
                                feature_len_per_step = len(final_features)

                        elif self.include_energy and node_id in self.energy_ids:
                            # --- B. Process energy nodes (include_energy=True only) ---
                            f_har = all_har_features[node_id].loc[current_date].values
                            f_vi = np.array([cov_matrix[node_idx, node_idx]])  # Own volatility.
                            f_cij_es = cov_matrix[node_idx, self.stock_indices]  # Energy-stock covariance.
                            
                            # Energy-energy correlations, excluding the node itself.
                            corr_values_ee = []
                            for other_eid in self.energy_ids:
                                if node_id == other_eid: continue
                                pair1, pair2 = f"{node_id}_{other_eid}", f"{other_eid}_{node_id}"
                                corr_series = all_corr_df.get(pair1, all_corr_df.get(pair2, pd.Series([0])))
                                corr_values_ee.append(corr_series[current_date])
                            f_corr_ee = np.array(corr_values_ee)
                            
                            node_features_t.append(np.concatenate([f_har, f_vi, f_cij_es, f_corr_ee]))
                        
                        elif (not self.include_energy) and (node_id in self.energy_ids):
                            # --- C. Add placeholders for disabled energy nodes ---
                            # HDF5 indices defined by node_order are fixed, so we
                            # still need to fill these positions.
                            # Placeholder length must match stock feature length.
                            
                            if feature_len_per_step == -1:
                                # This should not happen in theory, but keep it as
                                # a safeguard and recompute the stock feature length.
                                base_har_len = 3  # Assume 3 HAR features.
                                base_vi_len = 1
                                base_cij_ss_len = self.num_stocks - 1
                                # f_corr_se is excluded here because include_energy=False.
                                feature_len_per_step = base_har_len + base_vi_len + base_cij_ss_len
                            
                            # Add an all-zero placeholder.
                            node_features_t.append(np.zeros(feature_len_per_step))

                    # --- Combine node features ---
                    x_all = torch.tensor(np.array(node_features_t), dtype=torch.float)
                    
                    # Assign by index to the 'stock' node type.
                    data['stock'].x = x_all[self.stock_indices]
                    
                    # Assign energy features conditionally.
                    if self.include_energy:
                        data['energy'].x = x_all[self.energy_indices]
                    
                    # --- Build edges for the current timestep t ---
                    adj_matrix = volvol_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)  # Remove self-loops.
                    global_edge_index = torch.from_numpy(np.vstack(np.where(adj_matrix != 0))).long()
                    
                    global_edge_attr = None
                    if global_edge_index.numel() > 0:
                        variances = torch.tensor(np.diag(volvol_matrix), dtype=torch.float)
                        source_vars = variances[global_edge_index[0]]
                        target_vars = variances[global_edge_index[1]]
                        covars = torch.tensor(adj_matrix[global_edge_index[0], global_edge_index[1]], dtype=torch.float)
                        global_edge_attr = torch.stack([covars, source_vars, target_vars], dim=1)
                    
                    # relations is already conditional here. When include_energy=False,
                    # it only contains [('stock', 'to', 'stock')].
                    for src, rel, dst in relations:
                        if global_edge_index.numel() == 0: continue
                        
                        # If include_energy=False, self.energy_indices is empty.
                        global_src_indices = self.stock_indices if src == 'stock' else self.energy_indices
                        global_dst_indices = self.stock_indices if dst == 'stock' else self.energy_indices
                        
                        # The mask handles both cases automatically:
                        # 1. include_energy=True: normal processing.
                        # 2. include_energy=False:
                        #    - ('stock', 'to', 'stock'): works normally.
                        #    - ('stock', 'to', 'energy'): skipped because dst is empty.
                        #    - ('energy', 'to', 'stock'): skipped because src is empty.
                        mask = torch.isin(global_edge_index[0], global_src_indices) & torch.isin(global_edge_index[1], global_dst_indices)
                        
                        if mask.sum() == 0: continue  # No edges of this relation type.

                        # --- Map global indices to local heterogeneous-graph indices ---
                        src_map = {idx.item(): i for i, idx in enumerate(global_src_indices)}
                        dst_map = {idx.item(): i for i, idx in enumerate(global_dst_indices)}
                        
                        local_edge_index = global_edge_index[:, mask]
                        
                        # Use list comprehension explicitly because in-place tensor
                        # edits can behave inconsistently in some PyTorch versions.
                        local_edge_index_0 = torch.tensor([src_map[idx.item()] for idx in local_edge_index[0]])
                        local_edge_index_1 = torch.tensor([dst_map[idx.item()] for idx in local_edge_index[1]])
                        local_edge_index = torch.stack([local_edge_index_0, local_edge_index_1], dim=0)
                            
                        data[src, rel, dst].edge_index = local_edge_index
                        data[src, rel, dst].edge_attr = global_edge_attr[mask]
                    
                    # --- Build labels y (T+1 and T+1d) ---
                    try:
                        # Y_high: high-frequency volatility at T+1.
                        next_hf_global_idx, _ = all_hf_steps[i + j + 1]
                        next_cov_matrix = np.array(f_vol[str(next_hf_global_idx)])
                        y_high_all = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float)
                        # Labels are only defined for stocks.
                        data['stock'].y_high = y_high_all[self.stock_indices]
                        
                        # Y_low: low-frequency volatility (RV) at T+1d.
                        next_day_date = all_hf_steps[i + j + self.intraday_points][1]
                        y_low_values = [all_har_features[sid].iloc[:, 0].loc[next_day_date] for sid in self.stock_ids]
                        data['stock'].y_low = torch.tensor(y_low_values, dtype=torch.float)
                    except (KeyError, IndexError):
                        # Skip the full sequence when labels are missing.
                        is_valid_sequence = False; break
                    
                    seq_data_list.append(data)

                # --- 4b. Sequence construction finished ---
                if not is_valid_sequence:
                    continue  # Move to the next i.
                
                # --- 5. Aggregate sequence features ---
                # Merge seq_length graphs into one graph, stacking timesteps
                # along the feature dimension.
                final_data = HeteroData()
                
                # Aggregate stock features.
                final_data['stock'].x = torch.cat([d['stock'].x for d in seq_data_list], dim=1)
                
                # Aggregate energy features conditionally.
                if self.include_energy:
                    final_data['energy'].x = torch.cat([d['energy'].x for d in seq_data_list], dim=1)
                
                # --- Aggregate edges using the edge structure from the last timestep ---
                last_data = seq_data_list[-1]
                
                # edge_types is already conditional here.
                for rel in last_data.edge_types:
                    # Use edge indices from T=seq_length.
                    final_data[rel].edge_index = last_data[rel].edge_index
                    
                    # Aggregate edge features across timesteps.
                    edge_attr_seq = []
                    single_step_edge_dim = 3  # (covar, var_src, var_dst)
                    
                    for d in seq_data_list:
                        if rel in d and 'edge_attr' in d[rel]:
                            edge_attr_seq.append(d[rel].edge_attr)
                        else:
                            # Fill with zeros when a relation is absent at a timestep.
                            num_edges = last_data[rel].edge_index.size(1)
                            placeholder = torch.zeros(num_edges, single_step_edge_dim, dtype=torch.float)
                            edge_attr_seq.append(placeholder)
                    
                    final_data[rel].edge_attr = torch.cat(edge_attr_seq, dim=1)

                # Use the labels from the last timestep.
                final_data['stock'].y_high = last_data['stock'].y_high
                final_data['stock'].y_low = last_data['stock'].y_low
                
                # --- 6. Validate and keep the sample ---
                
                # Keep the sample only if it has at least one edge across all relation types.
                total_edges = sum(index.size(1) for index in final_data.edge_index_dict.values())
                
                if total_edges > 0:
                    data_list.append(final_data)
                else:
                    # Skip isolated graphs with no edges.
                    skipped_samples_count += 1

        # --- 7. Finish the loop and save the cache ---
        if skipped_samples_count > 0:
            print(f"\n[Data processing warning] Skipped {skipped_samples_count} samples because they had no edges.")

        if not data_list:
            raise ValueError("No valid graph samples could be created. Please check the HDF5 data and processing logic.")

        print("\nPacking and saving the final dataset...")
        # PyG collate packs data_list into one large graph object.
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        print(f"Dataset cache was created successfully and saved to {self.processed_paths[0]}.")
        print("==========================================================")

class CovarianceLaggedMultiOutputDataset(InMemoryDataset):
    def __init__(self, hdf5_file1, hdf5_file2, root='processed_data/cached_datasets_lagged_moutput/', transform=None, pre_transform=None, seq_length=None, future_steps=14):
        self.hdf5_file1 = hdf5_file1
        self.hdf5_file2 = hdf5_file2
        self.root = root
        self.seq_length = seq_length
        self.future_steps = future_steps
        super(CovarianceLaggedMultiOutputDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.hdf5_file1), os.path.basename(self.hdf5_file2)]


    @property
    def processed_file_names(self):
        return ['data_temp.pt']
    
    @property
    def processed_dir(self):
        return self.root
    
    def process(self):
        data_list = []
    
        # Load covariance matrices from the HDF5 file
        # Open both HDF5 files simultaneously
        with h5py.File(self.hdf5_file1, 'r') as f1, h5py.File(self.hdf5_file2, 'r') as f2:
            keys = list(f1.keys())
            # order keys
            keys = natsorted(keys)
            ordered = all(int(keys[i]) <= int(keys[i+1]) for i in range(len(keys)-1))
            assert ordered, 'Keys of files 1 have not been ordered'
            keys = list(f2.keys())
            # order keys
            keys = natsorted(keys)
            ordered = all(int(keys[i]) <= int(keys[i+1]) for i in range(len(keys)-1))
            assert ordered, 'Keys of files 2 have not been ordered'
    
            for i in range(len(f1) - self.seq_length - self.future_steps + 1):
                seq_data_list = []
                for j in range(self.seq_length):
                    # vol data
                    cov_matrix = np.array(f1[keys[i+j]])
                    # volvol data
                    covol_matrix = np.array(f2[keys[i+j]])
                    
                    # Create the adjacency matrix from the covariance matrix
                    adj_matrix = covol_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)  # Set the diagonal to zero
    
                    # # Create edge_index tensor
                    mask = np.triu(np.ones_like(adj_matrix), k=1) > 0
                    # edge_index = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()
    
                    # # Extract the variances (diagonal) as a separate tensor
                    # variances = torch.tensor(np.diag(covol_matrix), dtype=torch.float)
                    # source_indices = edge_index[0]
                    # target_indices = edge_index[1]
                    # source_variances = variances[source_indices]
                    # target_variances = variances[target_indices]
                    # covariances = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    # edge_attr = torch.stack([covariances, source_variances, target_variances], dim=1)
                    # Create edge_index tensor for upper triangle
                    edge_index_upper = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()
                    # Create edge_index tensor for lower triangle (transpose of upper)
                    edge_index_lower = edge_index_upper[[1, 0], :]
                    # Combine both to get full edge_index
                    edge_index = torch.cat([edge_index_upper, edge_index_lower], dim=1)

                    # edge_attr = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    # Extract the variances (diagonal) as a separate tensor
                    variances = torch.tensor(np.diag(covol_matrix), dtype=torch.float)
                    # Get the source and target indices from the edge_index
                    source_indices = edge_index[0]
                    target_indices = edge_index[1]
                    # Extract the variances for the source and target nodes
                    source_variances = variances[source_indices]
                    target_variances = variances[target_indices]
                    # Create the original edge attributes (covariances)
                    covariances = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    # Duplicate the covariances for the lower triangle
                    covariances = torch.cat([covariances, covariances])
                    # Concatenate the covariances with the source and target variances
                    edge_attr = torch.stack([covariances, source_variances, target_variances], dim=1)
                    
    
                    x = torch.tensor(cov_matrix, dtype=torch.float)
    
                    # Create a tensor to store the next 'self.future_steps' for each node
                    y_x_future_steps = []
                    for k in range(self.future_steps):
                        next_cov_matrix_k_steps = np.array(f1[keys[i+j+k+1]])
                        y_x_k_steps = torch.tensor(np.diag(next_cov_matrix_k_steps), dtype=torch.float)
                        y_x_future_steps.append(y_x_k_steps)
    
                    y_x_future_steps = torch.stack(y_x_future_steps, dim=1)  # Shape will be [num_nodes, self.future_steps]
    
                    # Create PyTorch Geometric Data object
                    data = Data(x=x, 
                                edge_index=edge_index,
                                edge_attr=edge_attr,
                                y_x=y_x_future_steps)
    
                    seq_data_list.append(data)
                    
    
                # Combine the sequence of Data objects into a single object with the desired format
                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=2).reshape(seq_data_list[0].x.shape[0],-1),
                                edge_index=seq_data_list[-1].edge_index,
                                edge_attr=torch.stack([d.edge_attr for d in seq_data_list], dim=2).reshape(seq_data_list[-1].edge_index.shape[1],-1),
                                y_x=seq_data_list[-1].y_x.reshape(-1))  # This will now be a [num_nodes, self.future_steps] tensor
    
                data_list.append(seq_data)


    
        data, slices = self.collate(data_list[:]) # to get a stationary dataset
        torch.save((data, slices), os.path.join(os.getcwd(),self.processed_paths[0]))


        
class CovarianceSparseDataset(InMemoryDataset):
    def __init__(self, hdf5_file, root='processed_data/cached_datasets_lagged/', transform=None, pre_transform=None, seq_length=None, threshold=None):
        self.hdf5_file = hdf5_file
        self.root = root
        self.seq_length = seq_length
        self.threshold = threshold
        super(CovarianceSparseDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return [os.path.basename(self.hdf5_file)]

    @property
    def processed_file_names(self):
        return ['data_temp.pt']
    
    @property
    def processed_dir(self):
        return self.root
    
    def process(self):
        data_list = []
    
        # Load covariance matrices from the HDF5 file
        with h5py.File(self.hdf5_file, 'r') as f:
            keys = list(f.keys())
            # order keys
            keys = natsorted(keys)
            ordered = all(int(keys[i]) <= int(keys[i+1]) for i in range(len(keys)-1))
            assert ordered, 'Keys have not been ordered'
            # TODO cutoff hardcoded
            for i in range(len(f)-int(len(f)*0.2), len(f) - self.seq_length):
            # for i in range(len(f) - self.seq_length):
                seq_data_list = []
                for j in range(self.seq_length):
                    cov_matrix = np.array(f[keys[i+j]])
                    next_cov_matrix  = np.array(f[keys[i+j+1]])
                    assert int(keys[i+j]) + 1 == int(keys[i+j+1]), 'The labeling process is not considering consecutive matrices'
                    
                    # Create the adjacency matrix from the covariance matrix
                    adj_matrix = cov_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the adjacency matrix (excluding diagonal)
                    upper_triangle = np.triu(np.ones_like(adj_matrix), k=1)
                    # mask = upper_triangle.astype(bool) & (adj_matrix > 0)
                    mask = upper_triangle.astype(bool) & ((adj_matrix > self.threshold) | (adj_matrix < -self.threshold))

                    # Create edge_index tensor
                    edge_index = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()  
                      
                    edge_attr = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    

                    # Extract the variances (diagonal) as node features
                    node_features = np.diag(cov_matrix)
                    x = torch.tensor(node_features, dtype=torch.float)
    
                    # Process the next covariance matrix to create target values
                    adj_matrix_next = next_cov_matrix.copy()
                    np.fill_diagonal(adj_matrix_next, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the next adjacency matrix (excluding diagonal)
                    # mask = np.triu(np.ones_like(adj_matrix_next), k=1) > 0
                    # y_edge = torch.tensor(adj_matrix_next[mask], dtype=torch.float)
    
                    # Extract the variances (diagonal) of the next covariance matrix as target node features
                    y_x = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float)
                    
                    
                    # Create PyTorch Geometric Data object
                    data = Data(x=x, edge_index=edge_index, 
                                edge_attr=edge_attr,
                                # y_edge=y_edge, 
                                y_x=y_x)
                    
                    seq_data_list.append(data)
                    
                    
                
                # Combine the sequence of Data objects into a single object with the desired format
                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=1),
                                edge_index=seq_data_list[-1].edge_index,
                                edge_attr=seq_data_list[-1].edge_attr,
                                # y_edge=seq_data_list[-1].y_edge,
                                y_x=seq_data_list[-1].y_x)
                
                data_list.append(seq_data)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
        
def check_reverse_edges_exist(edge_index):
    edge_index_t = edge_index.t().contiguous()
    
    # Convert to set for faster lookup
    edge_set = set(map(tuple, edge_index_t.tolist()))

    for edge in edge_set:
        if edge[::-1] not in edge_set:
            return False
            
    return True
        
        
        
# if not self.fully_connected:
#     upper_triangle = np.triu(np.ones_like(adj_matrix), k=1)
#     sparse_mask = upper_triangle.astype(bool) & (adj_matrix > 0)
#     sparse_edge_index = torch.tensor(np.argwhere(sparse_mask), dtype=torch.long).t().contiguous() 
    
#     fixed_value = -999
#     modified_edge_index = edge_index.clone()
#     for i in range(edge_index.size(1)):
#         edge = edge_index[:, i]
#         found = torch.any(torch.all(torch.eq(sparse_edge_index, edge.view(2, 1)), dim=0))
#         if not found:
#             modified_edge_index[:, i] = fixed_value
#     edge_index = modified_edge_index.clone()
#     # go back
#     # modified_edge_index[:, modified_edge_index[0] != -999]
    
#     sparse_edge_attr = torch.tensor(adj_matrix[sparse_mask], dtype=torch.float)
#     modified_edge_attr = edge_attr.clone()

#     for i in range(edge_attr.size(0)):
#         value = edge_attr[i]
#         if value.item() not in sparse_edge_attr.tolist():
#             modified_edge_attr[i] = fixed_value
#     # go back
#     # modified_edge_attr[modified_edge_attr!=-999]
