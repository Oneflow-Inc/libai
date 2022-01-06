
import argparse
from oneflow.utils.data import DataLoader
from libai.data.samplers import PretrainingSampler

from libai.data import build_dataset
from libai.tokenizer import build_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_prefix', nargs='*', help='prefix to data files')
    parser.add_argument('--dataset-impl', type=str, default='infer', choices=['lazy', 'cached', 'mmap', 'infer'])
    parser.add_argument('--max-seq-length', type=int, default=20, help='maximum sequence length')
    parser.add_argument('--mask-lm-prob', type=float, default=.15, help='probability of mask')
    parser.add_argument('--binary-head', action='store_true', help='enable bert binary head')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, choices=['BertWordPieceLowerCase', 'GPT2BPETokenizer'], help='what type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, help='Path to the BPE merge file.')
    group.add_argument('--append-eod', action='store_true', help='Append an <eod> token to the end of a document.')
    
    args = parser.parse_args()
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1

    tokenizer = build_tokenizer(args)
    
    train, valid, test = build_dataset(args, tokenizer, args.data_prefix, args.dataset_impl, split=[.8, .2, .0], data_type='gpt')

    print(len(train))

    sampler = PretrainingSampler(train, micro_batch_size=4, num_epochs=10, shuffle=True)

    data_loader = DataLoader(train, batch_sampler=sampler)

    for batch in data_loader:
        print(batch)


if __name__ == "__main__":
    main()
