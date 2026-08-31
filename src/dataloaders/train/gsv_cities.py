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
  author={Ali-bey, Amar and Chaib-draa, Brahim and Gigu{\\`e}re, Philippe},
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
from src.query_semantic_cache import (
    QUERY_SEMANTIC_CACHE_SCHEMA,
    QUERY_SEMANTIC_CACHE_VERSION,
    QUERY_SEMANTIC_SHUFFLE_ALGORITHM,
    build_cross_place_bijection,
)



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
                 return_crop_semantic_view=False,
                 teacher_transform=None,
                 semantic_cache_dir=None,
                 query_semantic_cache_dir=None,
                 query_semantic_selection="aligned",
                 rscd_cache_dir=None,
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
            return_crop_semantic_view (bool): Add exactly one clean full-image
                teacher view per place. It uses the first view from the
                seed-controlled sampled set and is consumed by the online
                crop-CLS semantic target.
            teacher_transform (callable): Deterministic transform for the clean
                teacher view. Required when either teacher-view option is true.
            semantic_cache_dir (path-like): Directory containing the sparse
                CLIP semantic-affinity cache.  Enabling it appends the sampled
                cache rows to metadata in exactly the same order as the K
                returned images.
            query_semantic_cache_dir (path-like): Directory containing the
                independent ADE20K patch-label cache used by Query-conditioned
                Semantic BoQ.
            query_semantic_selection (str): ``aligned`` reads each image's own
                labels, ``shuffled`` reads the manifest's within-city donor,
                and ``random`` transports aligned rows for deterministic
                random-target construction in the training framework.
            rscd_cache_dir (path-like): The same immutable ADE20K cache, read
                as both receiver-aligned and cross-place donor grids for the
                RSCD matched-mask experiment. It reuses the query-semantic
                stable index and does not create a second cache format.
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
        query_semantic_cache_path = (
            Path(query_semantic_cache_dir).expanduser().resolve()
            if query_semantic_cache_dir is not None
            else None
        )
        self.rscd_cache_dir = (
            Path(rscd_cache_dir).expanduser().resolve()
            if rscd_cache_dir is not None
            else None
        )
        if (
            query_semantic_cache_path is not None
            and self.rscd_cache_dir is not None
            and query_semantic_cache_path != self.rscd_cache_dir
        ):
            raise ValueError(
                "query_semantic_cache_dir and rscd_cache_dir must reference "
                "the same immutable ADE20K cache when both are enabled"
            )
        self.query_semantic_cache_dir = (
            query_semantic_cache_path or self.rscd_cache_dir
        )
        self.query_semantic_selection = str(query_semantic_selection).lower()
        if self.query_semantic_selection not in {
            'aligned', 'shuffled', 'random'
        }:
            raise ValueError(
                "query_semantic_selection must be aligned, shuffled, or random"
            )
        if (
            self.query_semantic_cache_dir is None
            and self.query_semantic_selection != 'aligned'
        ):
            raise ValueError(
                "query_semantic_selection requires query_semantic_cache_dir"
            )
        self.return_teacher_view = bool(return_teacher_view)
        self.return_crop_semantic_view = bool(return_crop_semantic_view)
        # Cached targets and online crop-teacher inputs are transported through
        # the existing metadata path.
        self.return_metadata = (
            return_metadata
            or self.return_crop_semantic_view
            or self.semantic_cache_dir is not None
            or self.query_semantic_cache_dir is not None
        )
        self.teacher_transform = teacher_transform
        self._semantic_cache_manifest = None
        self._semantic_cache_cities = None
        self._semantic_cache_arrays = None
        self._query_semantic_cache_manifest = None
        self._query_semantic_cache_cities = None
        self._query_semantic_cache_arrays = None
        self.query_semantic_grid_size = None
        self.query_semantic_num_classes = None
        self.query_semantic_eligible_min_views = None
        if self.semantic_cache_dir is not None:
            self._load_semantic_cache_manifest()
        if self.query_semantic_cache_dir is not None:
            self._load_query_semantic_cache_manifest()
            if self.img_per_place != self.query_semantic_eligible_min_views:
                raise ValueError(
                    "ADE20K semantic cache was built for "
                    f"img_per_place={self.query_semantic_eligible_min_views}, "
                    f"but the dataset uses {self.img_per_place}"
                )
        if self.return_teacher_view and self.return_crop_semantic_view:
            raise ValueError(
                "return_teacher_view and return_crop_semantic_view are "
                "mutually exclusive"
            )
        if (
            self.return_teacher_view or self.return_crop_semantic_view
        ) and not self.return_metadata:
            raise ValueError(
                "teacher views require return_metadata so the legacy training "
                "tuple remains unambiguous"
            )
        if (
            self.return_teacher_view or self.return_crop_semantic_view
        ) and self.teacher_transform is None:
            raise ValueError(
                "teacher_transform is required when a teacher view is enabled"
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

    @staticmethod
    def _query_semantic_grid_size(manifest):
        grid_size = manifest.get('grid_size')
        if (
            not isinstance(grid_size, list)
            or len(grid_size) != 2
            or any(isinstance(value, bool) for value in grid_size)
        ):
            raise ValueError(
                "query semantic cache grid_size must be [height, width]"
            )
        try:
            height, width = (int(value) for value in grid_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "query semantic cache grid_size must contain integers"
            ) from exc
        if [height, width] != grid_size or height < 1 or width < 1:
            raise ValueError(
                "query semantic cache grid_size must contain positive integers"
            )
        return height, width

    def _load_query_semantic_cache_manifest(self):
        manifest_path = self.query_semantic_cache_dir / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"query semantic cache manifest not found: {manifest_path}"
            )
        with manifest_path.open('r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        if manifest.get('schema') != QUERY_SEMANTIC_CACHE_SCHEMA:
            raise ValueError(
                f"unsupported query semantic cache schema in {manifest_path}"
            )
        if manifest.get('version') != QUERY_SEMANTIC_CACHE_VERSION:
            raise ValueError(
                f"unsupported query semantic cache version in {manifest_path}: "
                f"{manifest.get('version')}"
            )
        if not manifest.get('complete', False):
            raise ValueError(
                f"query semantic cache is incomplete: {manifest_path}; resume "
                "the precomputation before training"
            )

        image_count = manifest.get('num_images')
        if (
            isinstance(image_count, bool)
            or not isinstance(image_count, int)
            or image_count < 1
        ):
            raise ValueError("query semantic cache num_images must be positive")
        grid_size = self._query_semantic_grid_size(manifest)
        num_classes = manifest.get('num_classes')
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or not 1 <= num_classes <= 256
        ):
            raise ValueError(
                "query semantic cache num_classes must be in [1, 256]"
            )
        classes = manifest.get('classes')
        if (
            not isinstance(classes, list)
            or len(classes) != num_classes
            or any(not isinstance(name, str) or not name for name in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValueError(
                "query semantic cache classes must contain num_classes unique names"
            )
        declared_dtypes = {
            'labels_dtype': 'uint8',
            'confidence_dtype': 'uint8',
            'shuffled_indices_dtype': 'int32',
        }
        for field, expected in declared_dtypes.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"query semantic cache {field} must be {expected!r}"
                )

        eligible_min_views = manifest.get('eligible_min_views')
        if (
            isinstance(eligible_min_views, bool)
            or not isinstance(eligible_min_views, int)
            or eligible_min_views < 2
        ):
            raise ValueError(
                "query semantic cache eligible_min_views must be at least 2"
            )
        if (
            manifest.get('shuffle_algorithm')
            != QUERY_SEMANTIC_SHUFFLE_ALGORITHM
        ):
            raise ValueError(
                "query semantic cache has an unsupported shuffle algorithm"
            )

        city_entries = manifest.get('cities')
        if not isinstance(city_entries, list) or not city_entries:
            raise ValueError("query semantic cache manifest has no city entries")
        cities = {}
        expected_offset = 0
        for entry in city_entries:
            if not isinstance(entry, dict):
                raise ValueError(
                    "query semantic cache city entries must be mappings"
                )
            name = entry.get('name')
            if not isinstance(name, str) or not name or name in cities:
                raise ValueError(
                    "query semantic cache contains invalid/duplicate cities"
                )
            offset = entry.get('offset')
            count = entry.get('count')
            eligible_count = entry.get('eligible_count')
            rotation = entry.get('eligible_shuffle_rotation')
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (offset, count, eligible_count, rotation)
            ):
                raise ValueError(
                    f"query semantic cache city {name!r} has non-integer "
                    "offset/count/eligible shuffle metadata"
                )
            if offset != expected_offset or count < 2:
                raise ValueError(
                    "query semantic cache city offsets must form a contiguous "
                    "index and every city must contain at least two images"
                )
            if (
                not 2 <= eligible_count <= count
                or not 0 < rotation < eligible_count
            ):
                raise ValueError(
                    f"query semantic cache city {name!r} has invalid eligible "
                    "shuffle metadata"
                )
            csv_hash = entry.get('sha256')
            if not isinstance(csv_hash, str) or len(csv_hash) != 64:
                raise ValueError(
                    f"query semantic cache city {name!r} has invalid CSV sha256"
                )
            cities[name] = entry
            expected_offset += count
        if expected_offset != image_count:
            raise ValueError(
                "query semantic cache city counts do not match num_images"
            )

        self._validate_query_semantic_cache_arrays(
            manifest=manifest,
            city_entries=city_entries,
            grid_size=grid_size,
        )
        self._query_semantic_cache_manifest = manifest
        self._query_semantic_cache_cities = cities
        self.query_semantic_grid_size = grid_size
        self.query_semantic_num_classes = num_classes
        self.query_semantic_eligible_min_views = eligible_min_views

    def _validate_query_semantic_cache_arrays(
        self, manifest, city_entries, grid_size
    ):
        image_count = int(manifest['num_images'])
        height, width = grid_size
        specs = {
            'labels': ((image_count, height, width), np.dtype('uint8')),
            'confidence': ((image_count, height, width), np.dtype('uint8')),
            'shuffled_indices': ((image_count,), np.dtype('int32')),
        }
        arrays = {}
        for name, (shape, dtype) in specs.items():
            path = self.query_semantic_cache_dir / f'{name}.npy'
            if not path.is_file():
                raise FileNotFoundError(
                    f"query semantic cache array not found: {path}"
                )
            array = np.load(path, mmap_mode='r')
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(
                    f"invalid query semantic cache {name}: expected "
                    f"{shape}/{dtype}, found {array.shape}/{array.dtype}"
                )
            arrays[name] = array

        shuffled = arrays['shuffled_indices']
        for entry in city_entries:
            offset = int(entry['offset'])
            count = int(entry['count'])
            actual = np.asarray(
                shuffled[offset:offset + count], dtype=np.int64
            )
            if (
                bool(np.any(actual < offset))
                or bool(np.any(actual >= offset + count))
                or np.unique(actual).size != count
            ):
                raise ValueError(
                    f"query semantic cache shuffled_indices for {entry['name']!r} "
                    "must be a within-city bijection"
                )

    def _open_query_semantic_cache_arrays(self):
        """Open query-semantic memmaps lazily inside each DataLoader worker."""
        if self._query_semantic_cache_arrays is not None:
            return self._query_semantic_cache_arrays
        arrays = {
            name: np.load(
                self.query_semantic_cache_dir / f'{name}.npy', mmap_mode='r'
            )
            for name in ('labels', 'confidence', 'shuffled_indices')
        }
        self._query_semantic_cache_arrays = arrays
        return arrays

    def _read_query_semantic_cache(self, cache_indices):
        arrays = self._open_query_semantic_cache_arrays()
        rows = np.asarray(cache_indices, dtype=np.int64)
        source_rows = rows
        if self.query_semantic_selection == 'shuffled':
            source_rows = np.asarray(
                arrays['shuffled_indices'][rows], dtype=np.int64
            )
        labels = np.asarray(arrays['labels'][source_rows]).copy()
        if labels.size and int(labels.max()) >= self.query_semantic_num_classes:
            raise ValueError(
                "query semantic cache label is outside the configured class range"
            )
        confidence = np.asarray(
            arrays['confidence'][source_rows], dtype=np.float32
        ).copy()
        confidence /= 255.0
        result = {
            'query_semantic_labels': torch.from_numpy(labels),
            'query_semantic_confidence': torch.from_numpy(confidence),
            # Preserve the sampled image's stable identity even when its target
            # comes from a shuffled donor. Random controls key off this index.
            'query_semantic_cache_indices': torch.from_numpy(rows.copy()),
        }
        if self.rscd_cache_dir is not None:
            donor_rows = np.asarray(
                arrays['shuffled_indices'][rows], dtype=np.int64
            )
            donor_labels = np.asarray(
                arrays['labels'][donor_rows]
            ).copy()
            if (
                donor_labels.size
                and int(donor_labels.max()) >= self.query_semantic_num_classes
            ):
                raise ValueError(
                    "RSCD donor cache label is outside the configured class range"
                )
            donor_confidence = np.asarray(
                arrays['confidence'][donor_rows], dtype=np.float32
            ).copy()
            donor_confidence /= 255.0
            # RSCD always receives both grids.  The controller chooses the
            # experimental mode while keeping receiver/donor I/O identical.
            result.update(
                {
                    'rscd_labels': torch.from_numpy(
                        np.asarray(arrays['labels'][rows]).copy()
                    ),
                    'rscd_confidence': torch.from_numpy(
                        (
                            np.asarray(
                                arrays['confidence'][rows], dtype=np.float32
                            ).copy()
                            / 255.0
                        )
                    ),
                    'rscd_donor_labels': torch.from_numpy(donor_labels),
                    'rscd_donor_confidence': torch.from_numpy(
                        donor_confidence
                    ),
                    'rscd_cache_indices': torch.from_numpy(rows.copy()),
                    'rscd_donor_cache_indices': torch.from_numpy(
                        donor_rows.copy()
                    ),
                }
            )
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        # Never pickle/fork a mapping opened by a different worker process.
        state['_semantic_cache_arrays'] = None
        state['_query_semantic_cache_arrays'] = None
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
        query_shuffled = None
        if self.query_semantic_cache_dir is not None:
            query_shuffled = np.load(
                self.query_semantic_cache_dir / 'shuffled_indices.npy',
                mmap_mode='r',
            )
        for i, city in enumerate(self.cities):
            csv_path = self.base_path / 'Dataframes' / f'{city}.csv'
            df = pd.read_csv(csv_path)
            actual_hash = None
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
            if self.query_semantic_cache_dir is not None:
                cache_city = self._query_semantic_cache_cities.get(city)
                if cache_city is None:
                    raise ValueError(
                        f"city {city!r} is absent from query semantic cache manifest"
                    )
                expected_count = int(cache_city['count'])
                if len(df) != expected_count:
                    raise ValueError(
                        f"query semantic cache row-count mismatch for {city}: "
                        f"manifest={expected_count}, csv={len(df)}"
                    )
                if actual_hash is None:
                    actual_hash = self._sha256(csv_path)
                if actual_hash != cache_city.get('sha256'):
                    raise ValueError(
                        f"query semantic cache CSV hash mismatch for {csv_path}; "
                        "rebuild the cache before training"
                    )
                place_ids = df.loc[:, 'place_id'].to_numpy()
                place_counts = df.groupby('place_id')['place_id'].transform(
                    'size'
                ).to_numpy()
                eligible_positions = np.flatnonzero(
                    place_counts >= self.query_semantic_eligible_min_views
                )
                if eligible_positions.size != int(cache_city['eligible_count']):
                    raise ValueError(
                        f"query semantic eligible row count changed for {city!r}"
                    )
                eligible_donors, expected_rotation = (
                    build_cross_place_bijection(
                        place_ids[eligible_positions],
                        context=f"city {city!r}",
                    )
                )
                recorded_rotation = int(
                    cache_city['eligible_shuffle_rotation']
                )
                if recorded_rotation != expected_rotation:
                    raise ValueError(
                        f"query semantic shuffle rotation mismatch for {city!r}"
                    )
                expected_donors = np.arange(len(df), dtype=np.int64)
                expected_donors[eligible_positions] = eligible_positions[
                    eligible_donors
                ]
                offset = int(cache_city['offset'])
                actual_donors = np.asarray(
                    query_shuffled[offset:offset + len(df)],
                    dtype=np.int64,
                ) - offset
                if not np.array_equal(actual_donors, expected_donors):
                    raise ValueError(
                        f"query semantic eligible shuffle mismatch for {city!r}"
                    )
                eligible_donors = actual_donors[eligible_positions]
                if bool(
                    np.any(
                        place_ids[eligible_positions]
                        == place_ids[eligible_donors]
                    )
                ):
                    raise ValueError(
                        f"query semantic cache shuffled donor for {city!r} "
                        "contains an image from the same place; rebuild the cache"
                    )
                df['_query_semantic_cache_index'] = (
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

        crop_teacher_view_index = None
        crop_teacher_image = None
        if self.return_crop_semantic_view:
            # The K sampled views already change with the seeded sampler. Use
            # the first sampled view so matched runs consume exactly the same
            # clean photograph without maintaining worker-local epoch state.
            crop_teacher_view_index = 0
            
        imgs = []
        imgs_aug = []
        imgs_teacher = []
        for view_index, (_, row) in enumerate(place.iterrows()):
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
            elif (
                self.return_crop_semantic_view
                and view_index == crop_teacher_view_index
            ):
                # Return one clean full image per place. Cropping happens in
                # the main training process so all places share exactly the
                # same quadrant at a given global step.
                crop_teacher_image = self.teacher_transform(img)

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
            if self.return_crop_semantic_view:
                if crop_teacher_image is None:
                    raise RuntimeError("failed to construct crop teacher view")
                metadata['crop_semantic_teacher_image'] = crop_teacher_image
                metadata['crop_semantic_view_index'] = torch.tensor(
                    crop_teacher_view_index, dtype=torch.long
                )
            if self.semantic_cache_dir is not None:
                metadata.update(
                    self._read_semantic_cache(
                        place.loc[:, '_semantic_cache_index'].to_numpy(
                            dtype=np.int64
                        )
                    )
                )
            if self.query_semantic_cache_dir is not None:
                metadata.update(
                    self._read_query_semantic_cache(
                        place.loc[:, '_query_semantic_cache_index'].to_numpy(
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
