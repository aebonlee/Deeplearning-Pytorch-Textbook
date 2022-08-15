import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = get_tokenizer('basic_english')

import os
from functools import partial
from pathlib import Path

from torch.utils import (
    _add_docstring_header,
    _create_dataset_directory,
    _wrap_split_argument,
)
from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper

URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

MD5 = "7c2ac02c03563afcf9b574c7e56c153a"

NUM_LINES = {
    "train": 25000,
    "test": 25000,
}

_PATH = "aclImdb_v1.tar.gz"

DATASET_NAME = "IMDB"


def _path_fn(root, path):
    return os.path.join(root, os.path.basename(path))


def _filter_fn(split, t):
    return Path(t[0]).parts[-3] == split and Path(t[0]).parts[-2] in ["pos", "neg"]


def _file_to_sample(t):
    return Path(t[0]).parts[-2], t[1].read().decode("utf-8")


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def IMDB(root, split):
    """Demonstrates complex use case where each sample is stored in separate file and compressed in tar file
    Here we show some fancy filtering and mapping operations.
    Filtering is needed to know which files belong to train/test and neg/pos label
    Mapping is needed to yield proper data samples by extracting label from file name
        and reading data from file
    """

    url_dp = IterableWrapper([URL])
    # cache data on-disk
    cache_dp = url_dp.on_disk_cache(
        filepath_fn=partial(_path_fn, root),
        hash_dict={_path_fn(root, URL): MD5},
        hash_type="md5",
    )
    cache_dp = HttpReader(cache_dp).end_caching(mode="wb", same_filepath_fn=True)

    cache_dp = FileOpener(cache_dp, mode="b")

    # stack TAR extractor on top of load files data pipe
    extracted_files = cache_dp.load_from_tar()

    # filter the files as applicable to create dataset for given split (train or test)
    filter_files = extracted_files.filter(partial(_filter_fn, split))

    # map the file to yield proper data samples
    return filter_files.map(_file_to_sample)
