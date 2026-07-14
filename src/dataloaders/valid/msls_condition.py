from typing import Optional, Callable, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from src.utils import config_manager


class MSLSConditionDataset(Dataset):
    """
    MSLS-val condition-specific subset (night / season) for evaluating robustness.
    
    Uses the same database as standard msls-val, but with filtered queries
    targeting specific condition changes (illumination or season).
    
    Generate the required .npy files using: scripts/generate_msls_condition_splits.py
    """

    CONDITION_FILES = {
        "night": ("msls_val_night_qImages.npy", "msls_val_night_gt_25m.npy"),
        "season": ("msls_val_season_qImages.npy", "msls_val_season_gt_25m.npy"),
    }

    def __init__(
        self,
        condition: str,
        dataset_path: Optional[str] = None,
        input_transform: Optional[Callable] = None,
    ):
        self.input_transform = input_transform

        if condition not in self.CONDITION_FILES:
            raise ValueError(f"Unknown condition '{condition}'. Choose from: {list(self.CONDITION_FILES.keys())}")

        if dataset_path is None:
            dataset_path = config_manager.get_dataset_path(dataset_name="msls-val", dataset_type="val")
        else:
            dataset_path = Path(dataset_path)

        q_file, gt_file = self.CONDITION_FILES[condition]

        self.dataset_name = f"msls-val-{condition}"
        self.dataset_path = dataset_path
        self.dbImages = np.load(dataset_path / "msls_val_dbImages.npy")
        self.qImages = np.load(dataset_path / q_file)
        self.ground_truth = np.load(dataset_path / gt_file, allow_pickle=True)

        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_path = self.image_paths[index]
        img = Image.open(self.dataset_path / img_path)
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)
