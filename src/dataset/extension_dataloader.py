import os
from typing import Any, Callable, Dict, Optional

import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms as trsfs
from src.dataset.utils import get_subset, load_file, encode_loc
import numpy as np
import pandas as pd
from PIL import Image
import xarray as xr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EbirdMMETimeseriesDataset(Dataset):
    def __init__(
        self,
        num_species,
        df_paths,
        years=['2022', '2023'], # note that the targets cover 2010-2022, and 2023-2025, they are labelled as 2022, 2023
        data_base_dir="SatBird_data_complete",
        sat_root="satellite_visual",
        env_root="env_timeseries",
        hotspots_root="historical_hotspots",
        targets_folder="targets",
        mode: Optional[str] = "train",
        transforms:Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        env_transform:Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        Args:
            df_paths: dataframe with paths to data for each hotspot
            sat_root: folder with satellite data (Landsat/Sentinel) organized by year
            env_root: folder with environmental features (.nc or .tif)
            num_species: total number of species (for one-hot labels)
            years: list of years per hotspot
        """
        self.df = df_paths
        self.mode = mode
        self.years = years
        self.data_base_dir = data_base_dir
        self.sat_root = sat_root
        self.env_root = env_root
        self.hotspots_root = hotspots_root
        self.targets_root = targets_folder
        self.num_species = num_species
        self.env_transform = env_transform

    def __len__(self):
        return len(self.df)
    
    def preprocess_satellite(self, img):
        """
        img is CHW float32 numpy array
        """
        # convert to tensor
        img = torch.from_numpy(img)

        # resize using interpolate, unsqueeze = (1, 3, H, W)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(448, 448), mode='bilinear')
        img = img.squeeze(0) # (3, 448, 448)

        return img

    def load_tif(self, path):
        """Load and normalize a .tif file."""
        with Image.open(path) as img:
            arr = np.array(img).astype(np.float32)
            if arr.ndim == 2:  # single channel
                arr = np.expand_dims(arr, axis=-1)
            return arr
        
    def load_nc(self, path):
        ds = xr.open_dataset(path)
        env = ds["env_data"].values  
        return env

    def load_env(self, path):
        """Load environmental .npy or .tif file."""
        if path.endswith(".npy"):
            arr = np.load(path).astype(np.float32)
        elif path.endswith(".tif"):
            arr = self.load_tif(path)
        elif path.endswith(".nc"):
            arr = self.load_nc(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        return arr

    def __getitem__(self, idx):
        hotspot_id = self.df.iloc[idx]["hotspot_id"]

        sat_seq, env_seq, hotspot_seq = [], [], []
        
        # all files are just saved in their individual year folders, lowk makes sense to me
        # could change to storing datacubes if we need more space (like we do with environmental data)
        for year in self.years:
            # satellite images
            sat_path = os.path.join(self.data_base_dir, self.sat_root, 'sat_rasters_' + year, hotspot_id + '.tif')
            img = load_file(sat_path)
            sats = self.preprocess_satellite(img)
            sat_seq.append(sats)

        # environmental data
        env_path = os.path.join(self.data_base_dir, self.env_root, hotspot_id + '.nc')  
        env_dat = self.load_env(env_path)
        # this is different from the SatBird way -- loaded already timeseries dependent
        env_seq = torch.tensor(env_dat, dtype=torch.float32)

        # hotspot data
        hotspot_path = os.path.join(self.data_base_dir, self.targets_root, 'targets_2022', hotspot_id + '.json') 
        hotspot_dat = load_file(hotspot_path)
        hotspot = torch.tensor(hotspot_dat["probs"], dtype=torch.float32)
        hotspot_seq.append(hotspot)
        
        # stack into time dimension, generalizable since hotspot_seq is currently just 1 year :(
        sat_seq = torch.stack(sat_seq, dim=0)   # [T, C, H, W]
        hotspot_seq = torch.stack(hotspot_seq, dim=0)     # [T, num_species]

        # grab target!!
        species = load_file(os.path.join(self.data_base_dir, self.targets_root, 'targets_2023', hotspot_id + '.json'))
        
        # we're doing probs sry gotta simplify :)
        target = torch.tensor(species["probs"], dtype=torch.float32)
        num_complete_checklists = species["num_complete_checklists"]

        return [
            hotspot_id,
            self.years,
            sat_seq,
            env_seq,
            hotspot_seq,
            target,
            num_complete_checklists,
        ]
