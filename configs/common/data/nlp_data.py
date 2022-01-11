from re import L
from libai.config import LazyCall
from libai.libai import tokenizer
from libai.libai.config import arguments
from libai.libai.data.build import build_nlp_test_loader, build_nlp_train_val_test_loader
from libai.libai.data.datasets.bert_dataset import BertDataset


dataloader = dict(
    # Pad the vocab size to be divisible by this value
    # This is added for computational efficiency reasons.
    train = LazyCall(build_nlp_train_val_test_loader)(
        datasets = [
            LazyCall(BertDataset)(
                data_path="/workspace/idea_model/idea_bert/output_data/loss_compara_content_sentence",
                vocab_file="/workspace/idea_model/idea_bert/bert-base-chinese-vocab.txt",
                tokenizer_type="BertCNWWMTokenizer",
                dataset_type="bert",
                data_impl="mmap",
                extra=...,
            ),
            LazyCall(BertDataset)(
                data_path="/workspace/idea_model/idea_bert/output_data/loss_compara_content_sentence",
                vocab_file="/workspace/idea_model/idea_bert/bert-base-chinese-vocab.txt",
                tokenizer_type="BertCNWWMTokenizer",
                dataset_type="bert",
                data_impl="mmap",
                extra=...,
            ),
        ],
        splits=["949,50,1", "900,99,1"],
        weight=[0.5, 0.5],
        batch_size=128,
        num_workers=4,
    ),
    test = [
        LazyCall(build_nlp_test_loader)(
            dataset=LazyCall(testDataset1)(
                root="...",
            ),
            batch_size=128,
        ),
        LazyCall(build_nlp_test_loader)(
            dataset=LazyCall(testDataset2)(
                root="...",
            ),
            batch_size=128
        )
    ],
    
    make_vocab_size_divisible_by=128,
    merge_file=None,
    vocab_extra_ids=0,
    seq_length=512,
    encoder_seq_length=None,
    decoder_seq_length=None,
    sample_rate=1.0,
    mask_prob=0.15,
    short_seq_prob=0.1,
    mmap_warmup=True,
    tokenizer_setup=True,
    # What type of tokenizer to use
    # "BertWordPieceLowerCase",
    # "BertWordPieceCase",
    # "GPT2BPETokenizer",
    # "BertCNWWMTokenizer",
    # What type of dataset to use
    # Implementation of indexed datasets, choose from `lazy`, `cached`, `mmap`, `infer`.
    reset_position_ids=False,
    reset_attention_mask=False,
    eod_mask_loss=False,
    use_external_dataset=False,
    # Dataloader type and number of workers
    dataloader_type="single",
    num_workers=4,
)

