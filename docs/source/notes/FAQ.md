# Frequently Asked Questions

We list some common problems encountered by users and the corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others solve them.

## Training

- "Loss goes NaN or very large"
  1. Check if the dataset annotations are valid. Mask must be `{0, 1}` where `1` for tokens that are **not masked** and `0` for tokens that are **masked**.
  2. Check `initializer_range` in config file. It can be safely set to `0.02` in most cases. If the model size is very large, decreasing `initializer_range` is a good choice. For example, `initializer_range` can be set to `0.006` when training 175 billion parameter configuration GPT-3 model.

- "AMP enabled goes NaN"  
  Set `ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS=1` to check what triggers an overflow of the value range in fp16.

- "GPU out of memory when validation"  
  Decrease `test_micro_batch_size` and use `--fast-dev-run` for quickly running through training and evaluation to check if memory is sufficient.


## Model

- "`apply_query_key_layer_scaling` in MultiheadAttention"  
  As the number of attention heads increases, some of the GEMMS inside the self-attention layer become smaller and the number of elements in the self attention softmax also increases.  

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
  In tensor parallelism, `chunk` dimension and `flow.sbp.split` dimension will be the same in Huggingface's implementation which will occur some unexpected behaviors (i.e., changing the tensor's SBP unexpectedly).  

  We also provide a tutorial about how to load Huggingface weights correctly. Please refer to [How to use Huggingface's pretrained weights in LiBai](https://libai.readthedocs.io/en/latest/notes/How_to_implement_huggingface%27s_weights_in_LiBai.html) for more details.
  
- "the order of layer normalization and the residual connection"  
  This is critical to enable the scaling of the BERT-style models beyond BERT-Large. The architecture with `apply_residual_post_layernorm=False` eliminates instabilities observed using the origin BERT architecture with `apply_residual_post_layernorm=True` and also has a lower training loss according to [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf).

If you find some troubles hard to understand, feel free to open an issue to collect feedbacks in [OneFlow](https://github.com/Oneflow-Inc/oneflow).