from omegaconf import DictConfig

parallel_config = DictConfig(
    dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        pipeline_num_layers=32,
    )
)

eval_config = DictConfig(
    dict(
        pretrained_model_path="/root/models/Llama-2-7b-chat-hf",
        hf_tokenizer_path="/root/models/Llama-2-7b-chat-hf",
        model_type="llama",
        model_weight_type="huggingface",  # libai or huggingface
        eval_tasks=["lambada_openai", "gsm8k"],
        batch_size_per_gpu=1,
    )
)
