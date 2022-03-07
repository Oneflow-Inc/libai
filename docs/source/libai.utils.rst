libai.utils
##############################

libai.utils.distributed module
---------------------------------
.. currentmodule:: libai.utils
.. automodule:: libai.utils.distributed
    :members:
        get_dist_util,
        setup_dist_util,
        ttol,
        tton,
        synchronize,
        convert_to_distributed_default_setting,
        get_nd_sbp,
        get_layer_placement,
        get_world_size,
        get_num_nodes,
        get_rank,
        get_local_rank,
        same_sbp,
        get_data_parallel_size,
        get_data_parallel_rank,
        get_hidden_sbp
        
libai.utils.events module
---------------------------------
.. currentmodule:: libai.utils
.. automodule:: libai.utils.events
    :members:
        get_event_storage,
        JSONWriter,
        CommonMetricPrinter,
        EventStorage,


libai.utils.logger module
---------------------------------
.. currentmodule:: libai.utils
.. automodule:: libai.utils.logger
    :members:
        setup_logger,
        log_first_n,
        log_every_n,
        log_every_n_seconds


libai.utils.checkpoint module
---------------------------------
.. currentmodule:: libai.utils
.. automodule:: libai.utils.checkpoint
    :members:
        Checkpointer,
        PeriodicCheckpointer
