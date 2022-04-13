# Frequently Asked Questions

We list some common problems encountered by users and their corresponding solutions here. Feel free to enrich the lis if you find any frequent issues and have ways to help others to solve them.

## Training

- "Loss goes NaN or very large"
  1. Check if the dataset annotations are valid. Mask must be `{0, 1}` where `1` for tokens that are **not masked** and `0` for tokens that are **masked**.
  2. Check `initializer_range` in config file. It can be safely set to `0.02` nn most cases. If the model is very large, decreasing it is a good choice. For example, `initializer_range` can be set to `0.006` when training 175 billion parameter configuration GPT-3 model.

- "AMP enabled goes NaN"
  Set `ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS=1` to check what triggered an overflow of the value range in fp16.

- "GPU out of memory when validation"
  Decrease `test_micro_batch_size` and use `--fast-dev` for quickly running through training and evaluation to check if memory is sufficient.


## Model

- "`apply_query_key_layer_scaling` in MultiheadAttention"
  As the number of attention heads increases, some of the GEMMS inside the self-attention layer become smaller and also the number of elements in the self attention softmax increases.  

- "QKV implementation is not consistent with Hugging Face in self attention"
  ```python
  # query_key_value：[batch_size, seq_len, 3*hidden_size]

  # QKV in LiBai
  query_key_value = query_key_value.view(bsz, -1, self.num_heads, 3 * self.head_size)   
  query_key_value = query_key_value.permute(0, 2, 1, 3)                                
  query, key, value = flow.chunk(query_key_value, chunks=3, dim=-1)                    

  # QKV in Huggingface
  query, key, value = flow.chunk(query_key_value, chunks=3, dim=-1)
  query = query.view(query.size(0), query.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
  key = key.view(key.size(0), key.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
  value = value.view(value.size(0), value.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
  ```
  In tensor parallelism, `chunk` dimension and `flow.sbp.split` dimension will be same in Huggingface's implementation which will occur unexpected behavior.
  
- "the order of layer normalization and the residual connection"
  This is critical to enable the scaling of the BERT-style models beyond BERT-Large. The architecture with `apply_residual_post_layernorm=False` eliminates instabilities observed using the origin BERT architecture with `apply_residual_post_layernorm=True` and also has a lower training loss according to [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf).
