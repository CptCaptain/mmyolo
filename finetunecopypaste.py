import random
import numpy as np

from mmdet.datasets import PIPELINES, build_dataset
from mmdet.core import BitmapMasks
import mmcv

import lovely_tensors as lt
lt.monkey_patch()

@PIPELINES.register_module()
class FineTuneCopyPaste:
    """
    This is a variant for Simple Copy-Paste, that takes a secondary dataset of 
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
    Args:
        supl_dataset_cfg (dict): Dataset config to be used for the supplemental
            Dataset. Must include Pipeline.
            Default: None.      (set this yourself)
        copy_paste_chance (float): Apply this only to a subset of images to avoid
            severely unbalancing the dataset.
            Default: 0.3.
        max_num_pasted (int): The maximum number of pasted objects.
            Default: 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Default: 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Default: 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Default: True.
    """

    def __init__(
        self,
        supl_dataset_cfg=None,
        copy_paste_chance=0.3,
        max_num_pasted=100,
        bbox_occluded_thr=10,
        mask_occluded_thr=300,
        selected=True,
    ):
        if supl_dataset_cfg is None:
            raise Exception('Supplementary Dataset Path is required for fine-tune copy paste augmentation')

        self._get_supl_dataset(supl_dataset_cfg)
        self.copy_paste_chance = copy_paste_chance
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected

    def _get_supl_dataset(self, dataset_cfg):
        cfg_dict = dict(
            type='MultiImageMixDataset',
            dataset=dict(
                type='CocoDataset',
                ann_file=dataset_cfg['ann_file'],
                data_root=dataset_cfg['data_root'],
                img_prefix=dataset_cfg['img_prefix'],
                pipeline=dataset_cfg['pipeline'],
                classes=dataset_cfg['classes'],
                filter_empty_gt=False,
            ),
            pipeline=dataset_cfg['pipeline'],
            skip_type_keys=['Resize'],
        )
        cfg = mmcv.Config(cfg_dict)
        self.supl_dataset = build_dataset(cfg_dict)

    def get_indexes(self, dataset):
        """Call function to collect indexes.s.
        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        # return random.randint(0, len(dataset))
        # FIXME len -1 due to IndexErrors in the iterator. But Why?
        # https://wandb.ai/nkoch-aitastic/van-detection/runs/3lvi3hql/logs?
        return random.randint(0, len(dataset)-1)

    def __call__(self, results):
        """Call function to make a copy-paste of image.
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

    def _select_object(self, results):
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        masks = results['gt_masks']
        max_num_pasted = min(bboxes.shape[0] + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        selected_inds = np.random.choice(
            bboxes.shape[0], size=num_pasted, replace=False)

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        return results

    def _copy_paste(self, dst_results, src_results):
        """CopyPaste transform function.
        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_labels']
        dst_masks = dst_results['gt_masks']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_labels']
        src_masks = src_results['gt_masks']

        if len(src_bboxes) == 0:
            return dst_results

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self.get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes()
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        bboxes_inds = np.all(
            np.abs(
                (updated_dst_bboxes - dst_bboxes)) <= self.bbox_occluded_thr,
            axis=-1)
        masks_inds = updated_dst_masks.masks.sum(
            axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]
                         ) + src_img * composed_mask[..., np.newaxis]
        bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate(
            [updated_dst_masks.masks[valid_inds], src_masks.masks])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                              masks.shape[2])

        return dst_results

    def get_updated_masks(self, masks, composed_mask):
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'selected={self.selected}, '
        return repr_str

