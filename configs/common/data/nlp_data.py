data = dict(
    # Pad the vocab size to be divisible by this value
    # This is added for computational efficiency reasons.
    make_vocab_size_divisible_by=128,
    data_path=[
        "/workspace/idea_model/idea_bert/output_data/loss_compara_content_sentence"
    ],
    split="949,50,1",
    vocab_file="/workspace/idea_model/idea_bert/bert-base-chinese-vocab.txt",
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
    tokenizer_type="BertCNWWMTokenizer",
    # What type of dataset to use
    dataset_type="bert",
    # Implementation of indexed datasets, choose from `lazy`, `cached`, `mmap`, `infer`.
    data_impl="mmap",
    reset_position_ids=False,
    reset_attention_mask=False,
    eod_mask_loss=False,
    use_external_dataset=True,
    # Dataloader type and number of workers
    dataloader_type="single",
    num_workers=4,
)
