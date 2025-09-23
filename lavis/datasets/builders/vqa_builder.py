"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset, GQAInstructDataset

@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults.yaml",
        "balanced_val": "configs/datasets/gqa/balanced_val.yaml",
        "balanced_testdev": "configs/datasets/gqa/balanced_testdev.yaml",
    }

@registry.register_builder("gqa_instruct")
class GQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = GQAInstructDataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults_instruct.yaml",
        "balanced_val": "configs/datasets/gqa/balanced_val_instruct.yaml",
        "balanced_testdev": "configs/datasets/gqa/balanced_testdev_instruct.yaml",
    }
