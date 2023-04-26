# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
import numpy as np
import collections
from typing import Sequence, Union

from mmyolo.datasets import YOLOv5CocoDataset
from mmengine.dataset import BaseDataset, force_full_init
from mmcv.transforms import BaseTransform
from mmyolo.registry import TRANSFORMS, DATASETS
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness

from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks

import mmengine


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.
    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.
    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None. It is deprecated.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Default: 15.
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 pipeline: Sequence[str],
                 skip_type_keys: Union[Sequence[str], None] = None,
                 max_refetch: int = 15,
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')

        self._metainfo = self.dataset.metainfo
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self.num_samples = len(self.dataset)
        self.max_refetch = max_refetch

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.
        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        if not hasattr(self, 'metainfo'):
            self.metainfo = copy.deepcopy(self._metainfo)
        return self.metainfo


    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.
        Args:
            idx (int): Global index of ``ConcatDataset``.
        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                for i in range(self.max_refetch):
                    # Make sure the results passed the loading pipeline
                    # of the original dataset is not None.
                    indexes = transform.get_indexes(self.dataset)
                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]
                    mix_results = [
                        copy.deepcopy(self.dataset[index]) for index in indexes
                    ]
                    if None not in mix_results:
                        results['mix_results'] = mix_results
                        break
                else:
                    raise RuntimeError(
                        'The loading pipeline of the original dataset'
                        ' always return None. Please check the correctness '
                        'of the dataset and its pipeline.')

            for i in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                updated_results = transform(copy.deepcopy(results))
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                raise RuntimeError(
                    'The training pipeline of the dataset wrapper'
                    ' always return None.Please check the correctness '
                    'of the dataset and its pipeline.')

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.
        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys


@TRANSFORMS.register_module()
class FineTuneCopyPaste(BaseTransform):
    """This is a variant for Simple Copy-Paste, that takes a secondary dataset of 
    images of new classes, and copies the instances from this set into the main 
    dataset.
    Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:
    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.
    Required Keys:
    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks) (optional)
    Modified Keys:
    - img
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)
    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Defaults to 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Defaults to 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Defaults to 300.
    """

    def __init__(
            self,
            supl_dataset_cfg=None,
            copy_paste_chance=0.3,
            max_num_pasted: int = 100,
            bbox_occluded_thr: int = 10,
            mask_occluded_thr: int = 300,
        ) -> None:
        if supl_dataset_cfg is None:
            raise Exception('Supplementary Dataset Path is required for fine-tune copy paste augmentation')

        self._get_supl_dataset(supl_dataset_cfg)
        self.copy_paste_chance = copy_paste_chance
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr

    def _get_supl_dataset(self, dataset_cfg):
        cfg_dict = dict(
            type='mmdet.MultiImageMixDataset',
            dataset=dict(
                type='YOLOv5CocoDataset',
                ann_file=dataset_cfg['ann_file'],
                data_root=dataset_cfg['data_root'],
                data_prefix=dict(img=dataset_cfg['img_prefix']),
                pipeline=dataset_cfg['pipeline'],
                metainfo=dict(classes=dataset_cfg['classes']),
                filter_cfg=dict(filter_empty_gt=False),
            ),
            pipeline=dataset_cfg['pipeline'],
            skip_type_keys=['Resize'],
        )
        cfg = mmengine.Config(cfg_dict)
        self.supl_dataset = DATASETS.build(cfg_dict)

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.s.
        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(dataset))

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to make a copy-paste of image.
        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        # only apply to limited amount of images
        if random.uniform(0, 1) > self.copy_paste_chance:
            return results
            
        supl_idx = random.randint(0, len(self.supl_dataset)-1)
        supl_item = self.supl_dataset.__getitem__(supl_idx)
        return self._copy_paste(results, supl_item)

    @cache_randomness
    def _get_selected_inds(self, num_bboxes: int) -> np.ndarray:
        max_num_pasted = min(num_bboxes + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        return np.random.choice(num_bboxes, size=num_pasted, replace=False)

    def _select_object(self, results: dict) -> dict:
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_bboxes_labels']
        masks = results['gt_masks']
        ignore_flags = results['gt_ignore_flags']

        selected_inds = self._get_selected_inds(bboxes.shape[0])

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]
        selected_ignore_flags = ignore_flags[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_bboxes_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        results['gt_ignore_flags'] = selected_ignore_flags
        return results

    def _copy_paste(self, dst_results: dict, src_results: dict) -> dict:
        """CopyPaste transform function.
        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_bboxes_labels']
        dst_masks = dst_results['gt_masks']
        dst_ignore_flags = dst_results['gt_ignore_flags']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = src_results['gt_masks']
        src_ignore_flags = src_results['gt_ignore_flags']

        if len(src_bboxes) == 0:
            return dst_results

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        # TODO investigate how these bboxes have different shapes......
        updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        try:
            l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        except:
            print(f'{updated_dst_bboxes.tensor.v=}')
            print(f'{dst_bboxes.tensor.v=}')
            quit()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(
            dim=-1).numpy()
        masks_inds = updated_dst_masks.masks.sum(
            axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]
        ) + src_img * composed_mask[..., np.newaxis]
        bboxes = src_bboxes.cat([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate(
            [updated_dst_masks.masks[valid_inds], src_masks.masks])
        ignore_flags = np.concatenate(
            [dst_ignore_flags[valid_inds], src_ignore_flags])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_bboxes_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                              masks.shape[2])
        dst_results['gt_ignore_flags'] = ignore_flags

        # masks seem problematic...
        dst_results.pop('gt_masks')

        return dst_results

    def _get_updated_masks(self, masks: BitmapMasks,
                           composed_mask: np.ndarray) -> BitmapMasks:
        """Update masks with composed mask."""
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
        'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        return repr_str

