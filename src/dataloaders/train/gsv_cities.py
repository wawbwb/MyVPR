# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

"""
GSV-Cities dataset 
====================

This module implements a PyTorch Dataset class for GSV-Cities dataset from the paper:

"GSV-Cities: Toward Appropriate Supervised Visual Place Recognition" 
by Ali-bey et al., published in Neurocomputing, 2022.


Citation:
    @article{ali2022gsv,
        title={{GSV-Cities}: Toward appropriate supervised visual place recognition},
        author={Ali-bey, Amar and Chaib-draa, Brahim and Gigu{\`e}re, Philippe},
        journal={Neurocomputing},
        volume={513},
        pages={194--203},
        year={2022},
        publisher={Elsevier}
    }

URL: https://arxiv.org/abs/2210.10239
"""

import hashlib
import json

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.utils import config_manager



# First, check if the dataset is downloaded and the path put in the config/data/config.yaml file
# available_train_datasets = ConfigManager.get_dataset_paths_by_type("train")

# assert "gsv_cities" in available_train_datasets, "GSV-Cities dataset not found in the configuration file. Please check `config/data/config.yaml` file."

# Transforms are passed to the dataset, if not, we will use this standard transform
default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Now we can define the dataset class
class GSVCitiesDataset(Dataset):
    def __init__(self,
                 dataset_path=None,
                 cities="all", # or None
                 img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 hard_mining=False,
                 return_augmented=False,
                 aug_transform=None,
                 return_metadata=False,
                 return_teacher_view=False,
                 teacher_transform=None,
                 semantic_cache_dir=None,
                 ):
        """
        Args:
            cities (list): List of city names to use in the dataset. Default is "all" or None which uses all cities.
            base_path (Path): Base path for the dataset files.
            img_per_place (int): The number of images per place.
            random_sample_from_each_place (bool): Whether to sample images randomly from each place.
            transform (callable): Optional transform to apply on images.
            hard_mining (bool): Whether you are performing hard negative mining or not.
            return_augmented (bool): Whether to return a second, augmented image tensor.
            aug_transform (callable): Transform used for the augmented images.
            return_metadata (bool): Whether to append per-image metadata to the
                returned tuple. Metadata contains coordinates and optional
                condition diagnostics in the exact sampled-image order.
            return_teacher_view (bool): Add a deterministic, clean image tensor
                to metadata for frozen-teacher pair selection.
            teacher_transform (callable): Deterministic transform for the clean
                teacher view. Required when ``return_teacher_view`` is true.
            semantic_cache_dir (path-like): Directory containing the sparse
                CLIP semantic-affinity cache.  Enabling it appends the sampled
                cache rows to metadata in exactly the same order as the K
                returned images.
        """
        super().__init__()
        
        # check if the dataset path is provided, if not, use the one in the config/data/config.yaml file
        if dataset_path is None:
            print("No dataset path provided. Using `gsv-cities-light`. We will try to load the one in the config/data/config.yaml file.")
            dataset_path = config_manager.get_dataset_path(
                dataset_name="gsv-cities-light", 
                dataset_type="train")
        else:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path {dataset_path} does not exist. Please check the path.")
            
        self.base_path = Path(dataset_path)
        
        # let's check if the cities are valid
        if cities == "all" or cities is None:
            # get all cities from the Dataframes folder
            cities = [f.name[:-4] for f in self.base_path.glob("Dataframes/*.csv")]
        else:
            for city in cities:
                if not (self.base_path / 'Dataframes' / f'{city}.csv').exists():
                    raise FileNotFoundError(f"Dataframe for city {city} not found. Please check the city name.")

        self.cities = cities
        self.img_per_place = img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        self.hard_mining = hard_mining
        self.return_augmented = return_augmented
        self.aug_transform = aug_transform
        self.semantic_cache_dir = (
            Path(semantic_cache_dir).expanduser().resolve()
            if semantic_cache_dir is not None
            else None
        )
        # Cached targets are transported through the existing metadata path.
        self.return_metadata = return_metadata or self.semantic_cache_dir is not None
        self.return_teacher_view = return_teacher_view
        self.teacher_transform = teacher_transform
        self._semantic_cache_manifest = None
        self._semantic_cache_cities = None
        self._semantic_cache_arrays = None
        if self.semantic_cache_dir is not None:
            self._load_semantic_cache_manifest()
        if self.return_teacher_view and not self.return_metadata:
            raise ValueError(
                "return_teacher_view requires return_metadata so the legacy "
                "training tuple remains unambiguous"
            )
        if self.return_teacher_view and self.teacher_transform is None:
            raise ValueError(
                "teacher_transform is required when return_teacher_view is true"
            )
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_semantic_cache_manifest(self):
        manifest_path = self.semantic_cache_dir / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"semantic cache manifest not found: {manifest_path}"
            )
        with manifest_path.open('r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        if manifest.get('schema') != 'openvpr_clip_sparse_affinity':
            raise ValueError(
                f"unsupported semantic cache schema in {manifest_path}"
            )
        if manifest.get('version') != 1:
            raise ValueError(
                f"unsupported semantic cache version in {manifest_path}: "
                f"{manifest.get('version')}"
            )
        if not manifest.get('complete', False):
            raise ValueError(
                f"semantic cache is incomplete: {manifest_path}; resume the "
                "precomputation before training"
            )
        city_entries = manifest.get('cities')
        if not isinstance(city_entries, list) or not city_entries:
            raise ValueError("semantic cache manifest has no city entries")
        cities = {entry.get('name'): entry for entry in city_entries}
        if None in cities or len(cities) != len(city_entries):
            raise ValueError("semantic cache manifest contains invalid/duplicate cities")
        image_count = int(manifest.get('num_images', -1))
        ordered_entries = sorted(
            city_entries, key=lambda entry: int(entry.get('offset', -1))
        )
        expected_offset = 0
        for entry in ordered_entries:
            offset = int(entry.get('offset', -1))
            count = int(entry.get('count', -1))
            if offset != expected_offset or count < 0:
                raise ValueError(
                    "semantic cache city offsets do not form a contiguous index"
                )
            expected_offset += count
        if expected_offset != image_count:
            raise ValueError(
                "semantic cache city counts do not match manifest num_images"
            )
        patch_count = int(manifest.get('patch_count', -1))
        topk = int(manifest.get('topk', -1))
        if not (1 <= topk < patch_count <= 256):
            raise ValueError(
                "semantic cache requires 1 <= topk < patch_count <= 256"
            )
        for filename in ('indices.npy', 'weights.npy', 'confidence.npy'):
            if not (self.semantic_cache_dir / filename).is_file():
                raise FileNotFoundError(
                    f"semantic cache array not found: {self.semantic_cache_dir / filename}"
                )
        self._semantic_cache_manifest = manifest
        self._semantic_cache_cities = cities

    def _open_semantic_cache_arrays(self):
        """Open read-only memmaps lazily inside each DataLoader worker."""
        if self._semantic_cache_arrays is not None:
            return self._semantic_cache_arrays
        manifest = self._semantic_cache_manifest
        patch_count = int(manifest['patch_count'])
        topk = int(manifest['topk'])
        image_count = int(manifest['num_images'])
        arrays = {
            'indices': np.load(
                self.semantic_cache_dir / 'indices.npy', mmap_mode='r'
            ),
            'weights': np.load(
                self.semantic_cache_dir / 'weights.npy', mmap_mode='r'
            ),
            'confidence': np.load(
                self.semantic_cache_dir / 'confidence.npy', mmap_mode='r'
            ),
        }
        expected = {
            'indices': ((image_count, patch_count, topk), np.dtype('uint8')),
            'weights': ((image_count, patch_count, topk), np.dtype('float16')),
            'confidence': ((image_count, patch_count), np.dtype('float16')),
        }
        for name, array in arrays.items():
            shape, dtype = expected[name]
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(
                    f"invalid semantic cache {name}: expected {shape}/{dtype}, "
                    f"found {array.shape}/{array.dtype}"
                )
        self._semantic_cache_arrays = arrays
        return arrays

    def _read_semantic_cache(self, cache_indices):
        arrays = self._open_semantic_cache_arrays()
        rows = np.asarray(cache_indices, dtype=np.int64)
        # Fancy indexing followed by copy detaches tensors from read-only
        # memmaps and keeps their lifetime independent of the worker mapping.
        return {
            'semantic_indices': torch.from_numpy(
                np.asarray(arrays['indices'][rows]).copy()
            ),
            'semantic_weights': torch.from_numpy(
                np.asarray(arrays['weights'][rows]).copy()
            ),
            'semantic_confidence': torch.from_numpy(
                np.asarray(arrays['confidence'][rows]).copy()
            ),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        # Never pickle/fork a mapping opened by a different worker process.
        state['_semantic_cache_arrays'] = None
        return state
        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing one DataFrame
            for each city in self.cities
        '''
        dataframes = []
        for i, city in enumerate(self.cities):
            csv_path = self.base_path / 'Dataframes' / f'{city}.csv'
            df = pd.read_csv(csv_path)
            if self.semantic_cache_dir is not None:
                cache_city = self._semantic_cache_cities.get(city)
                if cache_city is None:
                    raise ValueError(
                        f"city {city!r} is absent from semantic cache manifest"
                    )
                expected_count = int(cache_city['count'])
                if len(df) != expected_count:
                    raise ValueError(
                        f"semantic cache row-count mismatch for {city}: "
                        f"manifest={expected_count}, csv={len(df)}"
                    )
                actual_hash = self._sha256(csv_path)
                if actual_hash != cache_city.get('sha256'):
                    raise ValueError(
                        f"semantic cache CSV hash mismatch for {csv_path}; "
                        "rebuild the cache before training"
                    )
                # Assign before city-level shuffle.  This is the stable cache
                # key: manifest city offset + original CSV row ordinal.
                df['_semantic_cache_index'] = (
                    int(cache_city['offset'])
                    + np.arange(len(df), dtype=np.int64)
                )
            df['place_id'] += i * 10**5 # to avoid place_id conflicts between cities
            df = df.sample(frac=1) # we always shuffle in city level
            dataframes.append(df)
        
        df = pd.concat(dataframes)
        # keep only places depicted by at least img_per_place images
        df = df[df.groupby('place_id')['place_id'].transform('size') >= self.img_per_place]
        return df.set_index('place_id')
        
    def __getitem__(self, index):
        if self.hard_mining:
            place_id = index
        else:
            place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample k images
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place) 
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        imgs = []
        imgs_aug = []
        imgs_teacher = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path / 'Images' / row['city_id'] / img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img_t = self.transform(img)
            else:
                img_t = img

            if self.return_augmented:
                # Photometric-only augmentation to preserve spatial alignment
                aug_t = self.aug_transform if self.aug_transform is not None else self.transform
                img_aug = aug_t(img)
                imgs_aug.append(img_aug)

            if self.return_teacher_view:
                # This path must stay deterministic.  CLIP disagreement should
                # describe the sampled photographs, not RandAugment artefacts.
                imgs_teacher.append(self.teacher_transform(img))

            imgs.append(img_t)

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        labels = torch.tensor(place_id).repeat(self.img_per_place)
        metadata = None
        if self.return_metadata:
            # Preserve the exact order of the sampled rows/images. float64
            # avoids discarding GPS precision while adding negligible memory
            # compared with the corresponding image tensors.
            coordinates = torch.as_tensor(
                place.loc[:, ['lat', 'lon']].to_numpy(dtype='float64'),
                dtype=torch.float64,
            )
            def numeric_metadata(column):
                if column not in place.columns:
                    return torch.full(
                        (self.img_per_place,), float('nan'), dtype=torch.float32
                    )
                return torch.as_tensor(
                    place.loc[:, column].to_numpy(dtype='float32'),
                    dtype=torch.float32,
                )

            metadata = {
                'coordinates': coordinates,
                'years': numeric_metadata('year'),
                'months': numeric_metadata('month'),
                'headings': numeric_metadata('northdeg'),
            }
            if self.return_teacher_view:
                metadata['teacher_images'] = torch.stack(imgs_teacher)
            if self.semantic_cache_dir is not None:
                metadata.update(
                    self._read_semantic_cache(
                        place.loc[:, '_semantic_cache_index'].to_numpy(
                            dtype=np.int64
                        )
                    )
                )

        if self.return_augmented:
            if self.return_metadata:
                return torch.stack(imgs), torch.stack(imgs_aug), labels, metadata
            return torch.stack(imgs), torch.stack(imgs_aug), labels
        if self.return_metadata:
            return torch.stack(imgs), labels, metadata
        return torch.stack(imgs), labels

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)
    
    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        """
            Given a row from the dataframe
            return the corresponding image name
        """
        city = row['city_id']
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
        return name
