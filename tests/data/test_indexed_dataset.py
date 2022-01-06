
from libai.data import indexed_dataset
from libai.tokenizer import build_tokenizer
import argparse
import os
import sys

import oneflow as flow

# script_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(script_dir, "../../../"))


def test_indexed_dataset(args):
    ds = indexed_dataset.make_dataset(args.data, args.dataset_impl)
    tokenizer = build_tokenizer(args)
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
    tokenizer = build_tokenizer(args)
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
    parser.add_argument('--data', type=str, help='prefix to data files')
    parser.add_argument('--dataset-impl', type=str, default='infer', choices=['lazy', 'cached', 'mmap', 'infer'])
    parser.add_argument('--count', type=int, default=10, help='Number of samples/documents to print')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, choices=['BertWordPieceLowerCase', 'GPT2BPETokenizer'], help='what type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, help='Path to the BPE merge file.')
    group.add_argument('--append-eod', action='store_true', help='Append an <eod> token to the end of a document.')

    args = parser.parse_args()
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1

    if args.dataset_impl == 'infer':
        args.dataset_impl = indexed_dataset.infer_dataset_impl(args.data)
    
    print(args.dataset_impl)
    test_indexed_dataset(args)
    test_indexed_dataset_get(args)

if __name__ == "__main__":
    main()
