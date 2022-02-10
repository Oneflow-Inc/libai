tokenization = dict(
    tokenizer_cfg=dict(
        name="BertTokenizer",
        vocab_file="bert-vocab.txt",
        do_lower_case=True,
    ),
    append_eod=False,
    make_vocab_size_divisible_by=1,
)
