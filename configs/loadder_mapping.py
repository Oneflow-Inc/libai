loader_mapping_models = dict(

    llama=dict(
        loader_prefix="projects.Llama.utils.llama_loader",
        huggingface_loader="LlamaLoaderHuggerFace",
    ),

    chatglm=dict(
        loader_prefix="projects.ChatGLM.utils.chatglm_loader",
        huggingface_loader="ChatGLMLoaderHuggerFace",
    ),

    qwen2=dict(
        loader_prefix="projects.Qwen2.utils.qwen_loader",
        huggingface_loader="Qwen2LoaderHuggerFace",
    ),

    aquila=dict(
        loader_prefix="projects.Aquila.utils.aquila_loader",
        huggingface_loader="AquilaLoaderHuggerFace",
    )

)
