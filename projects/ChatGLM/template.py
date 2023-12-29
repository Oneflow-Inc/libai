from dataclasses import dataclass
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from libai.tokenizer.tokenization_base import PreTrainedTokenizer
from libai.utils.logger import setup_logger


logger = setup_logger()
@dataclass
class Template:

    prefix: List[Union[str, Dict[str, str]]]
    prompt: List[Union[str, Dict[str, str]]]
    system: str
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool
    efficient_eos: bool
    replace_eos: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids, answer_ids = prompt_ids + encoded_pairs[-1][0], encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        return encoded_pairs

    def _format(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        r"""
        Aligns inputs to the standard format.
        """
        system = system or self.system # use system if provided
        history = history if (history and self.use_history) else []
        history = history + [(query, resp)]
        return system, history

    def _get_special_ids(
        self,
        tokenizer: "PreTrainedTokenizer"
    ) -> Tuple[List[int], List[int]]:
        if tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", True):
            bos_ids = [tokenizer.bos_token_id]
        else: # baichuan, qwen and gpt2 models have no bos token
            bos_ids = []

        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token is required.")

        if self.efficient_eos: # used in baichuan, qwen, chatglm, etc.
            eos_ids = []
        else:
            eos_ids = [tokenizer.eos_token_id]

        return bos_ids, eos_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + sep + query    resp + eos
        Turn t: sep + bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self._convert_inputs_to_ids(tokenizer, context=self.sep)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0:
                prefix_ids = self._convert_inputs_to_ids(tokenizer, context=self.prefix, system=system)
                if len(prefix_ids) != 0: # has prefix
                    prefix_ids = bos_ids + prefix_ids + sep_ids
                else:
                    prefix_ids = bos_ids
            else:
                prefix_ids = sep_ids + bos_ids

            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query, idx=str(turn_idx+1))
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((prefix_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs

    def _convert_inputs_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        context: List[Union[str, Dict[str, str]]],
        system: Optional[str] = None,
        query: Optional[str] = None,
        idx: Optional[str] = None
    ) -> List[int]:
        r"""
        Converts context to token ids.
        """
        kwargs = dict(add_special_tokens=False)

        token_ids = []
        for elem in context:
            if isinstance(elem, str):
                elem = elem.replace("{{system}}", system, 1) if system is not None else elem
                elem = elem.replace("{{query}}", query, 1) if query is not None else elem
                elem = elem.replace("{{idx}}", idx, 1) if idx is not None else elem
                if len(elem) != 0:
                    token_ids = token_ids + tokenizer.encode(elem, **kwargs)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            else:
                raise ValueError("Input must be string or dict[str, str], got {}".format(type(elem)))

        return token_ids

templates: Dict[str, Template] = {}

def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if name is None: # for pre-training
        return None

    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)

    stop_words = deepcopy(template.stop_words)
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        tokenizer.eos_token = stop_words.pop(0)
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))
    from transformers import PreTrainedTokenizer
    if stop_words:
        tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words)
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))

    return template

def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    system: str,
    sep: List[Union[str, Dict[str, str]]],
    stop_words: Optional[List[str]] = [],
    use_history: Optional[bool] = True,
    efficient_eos: Optional[bool] = False,
    replace_eos: Optional[bool] = False
) -> None:
    template_class = Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        system=system,
        sep=sep,
        stop_words=stop_words,
        use_history=use_history,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos
    )


register_template(
    name="chatglm2",
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        "{{system}}"
    ],
    prompt=[
        "[Round {{idx}}]\n\n问：{{query}}\n\n答："
    ],
    system="",
    sep=[
        "\n\n"
    ],
    efficient_eos=True
)


register_template(
    name="chatglm3",
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        {"token": "<|system|>"},
        "\n",
        "{{system}}"
    ],
    prompt=[
        {"token": "<|user|>"},
        "\n",
        "{{query}}",
        {"token": "<|assistant|>"},
        "\n" # add an extra newline to avoid error in ChatGLM's process_response method
    ],
    system=(
        "You are ChatGLM3, a large language model trained by Zhipu.AI. "
        "Follow the user's instructions carefully. Respond using markdown."
    ),
    sep=[],
    stop_words=[
        "<|user|>",
        "<|observation|>"
    ],
    efficient_eos=True
)


register_template(
    name="chatglm3_raw", # the raw template for tool tuning
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        {"token": "<|system|>"},
        "\n",
        "{{system}}"
    ],
    prompt=[
        {"token": "<|user|>"},
        "\n",
        "{{query}}",
        {"token": "<|assistant|>"}
    ],
    system=(
        "You are ChatGLM3, a large language model trained by Zhipu.AI. "
        "Follow the user's instructions carefully. Respond using markdown."
    ),
    sep=[],
    stop_words=[
        "<|user|>",
        "<|observation|>"
    ],
    efficient_eos=True
)
