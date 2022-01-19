tokenizer = dict(
    tokenizer_name="BertTokenizer",
    tokenizer_cfg=dict(
        vocab_file="bert-vocab.txt",
        do_lower_case=True,
    ),
    append_eod=False,
)
