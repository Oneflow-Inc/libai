from abc import ABCMeta, abstractclassmethod, abstractmethod

from libai.config import LazyConfig, try_get_key
from libai.engine.default import DefaultTrainer
from libai.utils.checkpoint import Checkpointer


class BasePipeline(metaclass=ABCMeta):
    """
    Base class for all task pipeline
    """

    def __init__(
        self,
        config_file,
        **kwargs,
    ):
        self.cfg = LazyConfig.load(config_file)
        self.model = self.load_model(config_file)
        self.tokenier = self.build_tokenizer(config_file)

    @abstractmethod
    def load_model(self, cfg):
        model = DefaultTrainer.build_model(cfg).eval()
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=False
        )
        return model

    @abstractmethod
    def build_tokenizer(self, cfg):
        tokenizer = None
        if try_get_key(cfg, "tokenization") is not None:
            tokenizer = DefaultTrainer.build_tokenizer(cfg)
        return tokenizer

    @abstractclassmethod
    def __call__(self, inputs_dict) -> dict:
        model_inputs_dict = self.preprocess(inputs_dict)
        model_outputs_dict = self.forward(model_inputs_dict)
        outputs_dict = self.postprocess(model_outputs_dict)
        return outputs_dict

    @abstractclassmethod
    def preprocess(self, inputs_dict) -> dict:
        raise NotImplementedError

    @abstractclassmethod
    def forward(self, model_inputs_dict) -> dict:
        raise NotImplementedError

    @abstractclassmethod
    def postprocess(self, model_outputs_dict) -> dict:
        raise NotImplementedError
