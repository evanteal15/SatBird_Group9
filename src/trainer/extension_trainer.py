# SatBird + GeoPlant's https://www.kaggle.com/code/picekl/sentinel-landsat-bioclim-baseline
import os
import torch
import tqdm
from typing import Any, Dict, Optional, cast
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig

import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import precision_recall_fscore_support, f1_score

from src.utils.config_utils import load_opts
from src.losses.losses import CustomCrossEntropyLoss, WeightedCustomCrossEntropyLoss, RMSLELoss, CustomFocalLoss
from src.losses.metrics import get_metrics
from src.dataset.extension_dataloader import EbirdMMETimeseriesDataset
from src.trainer.utils import get_target_size, get_nb_bands, get_scheduler, init_first_layer_weights
from src.transforms.transforms import get_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EbirdDataModule(pl.LightningDataModule):
    def __init__(self, opts) -> None:
        super().__init__()
        self.opts = opts

        self.seed = self.opts.program.seed
        self.batch_size = self.opts.data.loaders.batch_size
        self.num_workers = self.opts.data.loaders.num_workers
        self.data_base_dir = self.opts.data.files.base
        if self.opts.split == "cluster":
            df = pd.read_csv("SatBird_data_complete/training/satbird_clustered_summer.csv")
            self.df_train = df[df['split'] == 'train']
            self.df_val  = df[df['split'] == 'valid']
            self.df_test = df[df['split'] == 'test']
        else:
            self.df_train = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.train))
            self.df_val = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.val))
            self.df_test = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.test))
        self.num_species = self.opts.data.total_species
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        """create the train/test/val splits and prepare the transforms for the multires"""
        self.train_dataset = EbirdMMETimeseriesDataset(df_paths=self.df_train, data_base_dir=self.data_base_dir, mode="train",
                                                    num_species=self.num_species)

        self.test_dataset = EbirdMMETimeseriesDataset(df_paths=self.df_test, data_base_dir=self.data_base_dir, mode="test",
                                                   num_species=self.num_species)

        self.val_dataset = EbirdMMETimeseriesDataset(df_paths=self.df_val, data_base_dir=self.data_base_dir, mode="val",
                                                  num_species=self.num_species)


    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, )

    def val_dataloader(self) -> DataLoader[Any]:
        """Returns the validation dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, )

    def test_dataloader(self) -> DataLoader[Any]:
        """Returns the test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, )


class MultimodalEnsemble(pl.LightningModule):
    def __init__(self, target_size, opts):
        super().__init__()
        self.opts = opts

        self.target_size = target_size

        # ----------------------------
        # Landsat branch
        # Input: [B, 2, 3, 448, 448] 
        # we separate by year and process them separately to try and get timeseries dependent
        # ----------------------------
        self.resnet = True
        self.temporal = None
        if not self.resnet:
            self.temporal = "transformer"
            self.landsat_model = models.resnet18()
            self.landsat_model.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.landsat_model.maxpool = nn.Identity() # should remove these?
            self.landsat_model.fc = nn.Identity() # should remove these?
            self.landsat_model.to(device)
            
            # Temporal model
            if self.temporal == "transformer":
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.landsat_out, nhead=8, batch_first=True
                )
                self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.temporal_model.to(device)
            elif self.temporal == "gru":
                self.temporal_model = nn.GRU(self.landsat_out, self.landsat_out, batch_first=True)
            else:
                self.temporal_model = nn.Identity()
        else:
            self.landsat_model = models.resnet18()
            self.landsat_model.fc = nn.Linear(512, 512)
            self.landsat_model.to(device)

        self.landsat_out = 512

        # ----------------------------
        # Bioclim branch
        # Using 3 channels (tmax, tmin, prec)
        # ----------------------------
        self.bioclim_model = models.resnet18()

        self.bioclim_model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,     # smaller kernel for 15×12 inputs
            stride=1,          # DO NOT stride 2, or spatial dims collapse
            padding=1,
            bias=False,
        )

        # Remove maxpool (destroys resolution)
        self.bioclim_model.maxpool = nn.Identity()
        self.bioclim_model.fc = nn.Identity()
        self.bioclim_model.to(device)

        # Output size stays 512 for resnet18
        bioclim_out = 512

        # ----------------------------
        # Past checklist branch (MLP)
        # Input: [B, 1, 670] → squeeze → [B, 670]
        # ----------------------------
        self.checklist_mlp = nn.Sequential(
            nn.Linear(670, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.checklist_mlp.to(device)
        checklist_out = 512

        # ----------------------------
        # Final fusion network
        # ----------------------------
        total_in = self.landsat_out + bioclim_out + checklist_out

        self.fc = nn.Sequential(
            nn.Linear(total_in, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, self.target_size),
        )
        self.fc.to(device)

        # ===== Metrics setup =====
        metrics = get_metrics(opts)
        for (name, value, _) in metrics:
            setattr(self, "val_" + name, value)
            setattr(self, "train_" + name, value)
            setattr(self, "test_" + name, value)
        self.metrics = metrics

        self.config_task(opts)

    def forward(self, landsat, bioclim, past_checklist):
        # landsat: [B, 2, 3, 448, 448]
        B, T, C, H, W = landsat.shape        
        landsat_dat = landsat.view(B*T, C, H, W)
        feats = self.landsat_model(landsat_dat) # CNN for each timestep [B*T, 512]
        

        # apply temporal model
        if self.temporal:
            # reshape back to time series [B, T, 512]
            feats = feats.view(B, T, self.landsat_out)
            if isinstance(self.temporal_model, nn.GRU):
                feats, _ = self.temporal_model(feats)   # [B, T, 512]
            else:
                feats = self.temporal_model(feats)      # transformer
        landsat_feat = feats.view(B, T, self.landsat_out).squeeze(1)
        landsat_feat = landsat_feat.mean(dim=1)
        # bioclim: [B, 1, 27, 50, 50]
        bioclim_feat = self.bioclim_model(bioclim)

        # past_checklist: [B, 1, 670]
        past_checklist = past_checklist.squeeze(1)  # → [B, 670]
        past_checklist_feat = self.checklist_mlp(past_checklist)

        # fuse
        x = torch.cat([landsat_feat, bioclim_feat, past_checklist_feat], dim=1)

        return self.fc(x)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def config_task(self, opts, **kwargs: Any) -> None:
        if opts.losses.criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif opts.losses.criterion == "MAE":
            self.criterion = nn.L1Loss()
        elif opts.losses.criterion == "RMSLE":
            self.criterion = RMSLELoss()
        elif opts.losses.criterion == "Focal":
            self.criterion = CustomFocalLoss()
        else:
            # target is num checklists reporting species i / total number of checklists at a hotspot
            if opts.experiment.module.use_weighted_loss:
                self.criterion = WeightedCustomCrossEntropyLoss()
                print("Training with Weighted CE Loss")
            else:
                self.criterion = CustomCrossEntropyLoss()
                print("Training with Custom CE Loss")
    
    def validation_step(self, batch, batch_idx):
        hotspot_id, years, landsat, bioclim, past_checklists, targets, num_checklists = batch

        landsat = landsat.to(device)
        bioclim = bioclim.to(device)
        past_checklists = past_checklists.to(device)
        targets = targets.to(device)

        
        outputs = self(landsat, bioclim, past_checklists)
        preds = torch.sigmoid(outputs).type_as(targets)
        
        loss = self.criterion(preds, targets)
        for (name, _, scale) in self.metrics:
            nname = "test_" + name
            if name == "accuracy":
                value = getattr(self, name)(preds, targets.type(torch.uint8))
                print(nname, getattr(self, name))
            elif name == 'r2':
                value = torch.mean(getattr(self, nname)(targets, preds))
            else:
                value = getattr(self, nname)(targets, preds)
                print(nname, getattr(self, nname)(targets, preds))

            self.log(nname, value, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        hotspot_id, years, landsat, bioclim, past_checklists, targets, num_checklists = batch

        landsat = landsat.to(device)
        bioclim = bioclim.to(device)
        past_checklists = past_checklists.to(device)
        targets = targets.to(device)

        
        outputs = self(landsat, bioclim, past_checklists)
        preds = torch.sigmoid(outputs).type_as(targets)
        
        loss = self.criterion(preds, targets)
        for (name, _, scale) in self.metrics:
            nname = "test_" + name
            if name == "accuracy":
                value = getattr(self, name)(preds, targets.type(torch.uint8))
                print(nname, getattr(self, name))
            elif name == 'r2':
                value = torch.mean(getattr(self, nname)(targets, preds))
            else:
                value = getattr(self, nname)(targets, preds)
                print(nname, getattr(self, nname)(targets, preds))

            self.log(nname, value, on_epoch=True)
        self.log("test_loss", loss, on_epoch=True)

        if self.opts.save_preds_path != "":
            preds_path = os.path.join(self.opts.base_dir, self.opts.save_preds_path)
            if os.path.exists(preds_path):
                os.mkdir(preds_path)
            for i, elem in enumerate(preds):
                np.save(os.path.join(preds_path, batch["hotspot_id"][i] + ".npy"), elem.cpu().detach().numpy())
        print("saved elems")


hydra_config_path = Path(__file__).resolve().parent / "configs/extension_config.yaml"

@hydra.main(version_base=None, config_path="/gpfs/accounts/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/SatBird_Group9/configs/", config_name="extension_config")
def main(opts):
    print("Starting up trainer.")
    hydra_opts = dict(OmegaConf.to_container(opts))
    args = hydra_opts.pop("args", None)
    num_species = args["num_species"]

    # Check if cuda is available
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("DEVICE = CUDA")
    else:
        print("CUDA not available")
    
    model = MultimodalEnsemble(num_species, opts).to(device)

    # Hyperparameters
    learning_rate = 0.00025
    num_epochs = 10
    positive_weigh_factor = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=25)

    base_dir = args['base_dir']
    if not base_dir:
        base_dir = get_original_cwd()

    config_path = os.path.join(base_dir, "configs", args['config'])
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    config = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
    datamodule = EbirdDataModule(config)
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    val_loader = datamodule.val_dataloader()

    """ These dataloaders return
            "hotspot_id": hotspot_id,
            "years": torch.tensor(self.years),
            "sat_seq": sat_seq,
            "env_seq": env_seq,
            "hotspot_seq": hotspot_seq,
            "target": target,
            "num_complete_checklists": num_complete_checklists,
    """

    retrain = True
    if retrain:
        print(f"Training for {num_epochs} epochs started.")
        for epoch in range(num_epochs):
            model.train()
            
            for batch_idx, (hotspot_id, years, landsat, bioclim, past_checklists, targets, num_checklists) in enumerate(train_loader):
                
                # load in everything on our GPU :)
                bioclim = bioclim.to(device)
                landsat = landsat.to(device)
                past_checklists = past_checklists.to(device)
                targets = targets.to(device)

                # run everything thru the model
                logits = model(landsat, bioclim, past_checklists)  # [B, C] or [B, C, T]

                weighted_loss_operations = {
                    "sqrt": torch.sqrt,
                    "log": torch.log,
                    "nchklists": lambda x: x,
                }

                weight_type = "sqrt" # can change this for other types of weighting
            
                # batch["num_complete_checklists"] -> shape [B]
                new_weights = weighted_loss_operations[weight_type](num_checklists.to(device))

                # broadcast weights to match target shape
                new_weights = new_weights.view(-1, *([1] * (targets.ndim - 1)))  # [B,1] or [B,1,1]

                # elementwise BCE loss
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

                # apply hotspot weights
                weighted_loss = bce * new_weights
                loss = weighted_loss.mean()

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print update
                if batch_idx % 278 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

            # adjust training params according to scheduler
            scheduler.step()
            print("Scheduler:",scheduler.state_dict())

        # Save the trained model
        model.eval()
        torch.save(model.state_dict(), "multimodal-model.pth")
        print("Done")
    else:
        print("Loading model from path.")
        model.load_state_dict(torch.load("multimodal-model.pth", weights_only=True))
        model.eval()

    # let's evaluate!!

    # val step
    trainer = pl.Trainer(accelerator='gpu', devices=1)
    trainer.validate(model, dataloaders=val_loader)

    # train step
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
