# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from libai.data.data_utils import indexed_dataset
from libai.tokenizer import BertTokenizer


def test_indexed_dataset(args):
    ds = indexed_dataset.make_dataset(args.data, args.dataset_impl)
    tokenizer = BertTokenizer(vocab_file="bert-vocab.txt")
    print(len(ds.doc_idx))
    print(len(ds))
    print(ds.doc_idx[-1])
    print(ds.doc_idx.dtype, ds.sizes.dtype)
    if ds.supports_prefetch:
        # just prefetch the whole thing in test (so assume it is small)
        ds.prefetch(range(len(ds)))
    if args.count > len(ds.doc_idx) - 1:
        args.count = len(ds.doc_idx) - 1

    for i in range(args.count):
        start = ds.doc_idx[i]
        end = ds.doc_idx[i + 1]
        ids = ds[start:end]
        print(f"Document {i}:")
        print("--------------")
        for s in ids:
            assert len(s) > 0
            l = s.data.tolist()
            text = tokenizer.decode(l)
            print(text)
            print("---")


def test_indexed_dataset_get(args):
    ds = indexed_dataset.make_dataset(args.data, args.dataset_impl)
    tokenizer = BertTokenizer(vocab_file="bert-vocab.txt")
    size = ds.sizes[0]
    print(f"size: {size}")

    full = ds.get(0)
    print(full)
    print(tokenizer.decode(full.data.tolist()))
    print("---")

    start = ds.get(0, length=10)
    print(start)
    print(tokenizer.decode(start.data.tolist()))
    print("---")

    part = ds.get(0, offset=2, length=8)
    print(part)
    print(tokenizer.decode(part.data.tolist()))
    print("---")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="test_samples_lazy_text_sentence", type=str, help="prefix to data files",
    )
    parser.add_argument(
        "--dataset-impl", type=str, default="infer", choices=["lazy", "cached", "mmap", "infer"],
    )
    parser.add_argument(
        "--count", type=int, default=10, help="Number of samples/documents to print"
    )
    args = parser.parse_args()

    if args.dataset_impl == "infer":
        args.dataset_impl = indexed_dataset.infer_dataset_impl(args.data)

    print(args.dataset_impl)
    test_indexed_dataset(args)
    test_indexed_dataset_get(args)


if __name__ == "__main__":
    main()
