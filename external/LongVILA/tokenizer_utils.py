# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers

from constants import IGNORE_INDEX, SENTINEL_TOKEN
from mm_utils import tokenizer_image_token
import dataclasses
from enum import Enum, auto
from typing import List

# from llava.utils.logging import logger


class SeparatorStyle(Enum):
    """Different separator style."""

    AUTO = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_3 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    sep_style: SeparatorStyle = SeparatorStyle.AUTO
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            ret = self.system + self.sep
            for rid, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    sep = self.sep if rid < len(messages) - 1 else self.sep2
                    ret += role + message + sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )


conv_auto = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    sep_style=SeparatorStyle.AUTO,
    sep="\n",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

hermes_2 = Conversation(
    system="<|im_start|>system\nAnswer the questions.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
    messages=(),
    version="hermes-2",
)

# Template added by Yukang. Note (kentang-mit@): sep is <|eot_id|> for official template.
llama_3_chat = Conversation(
    system="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama_v3",
    messages=(),
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|eot_id|>",
    sep2="<|end_of_text|>",
)


default_conversation = conv_auto
conv_templates = {
    "auto": conv_auto,
    "hermes-2": hermes_2,
    "llama_3": llama_3_chat,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "plain": conv_llava_plain,
}


CONVERSATION_MODE_MAPPING = {
    "vila1.5-3b": "vicuna_v1",
    "vila1.5-8b": "llama_3",
    "vila1.5-13b": "vicuna_v1",
    "vila1.5-40b": "hermes-2",
    "llama-3": "llama_3",
    "llama3": "llama_3",
}


def auto_set_conversation_mode(model_name_or_path: str) -> str:
    global default_conversation
    for k, v in CONVERSATION_MODE_MAPPING.items():
        if k in model_name_or_path.lower():
            print(f"Setting conversation mode to `{v}` based on model name/path `{model_name_or_path}`.")
            default_conversation = conv_templates[v]
            return


DUMMY_CONVERSATION = [
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "answer"},
] * 10


def tokenize_conversation_legacy(
    messages: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    add_generation_prompt: bool = False,
    overrides: Optional[Dict[str, str]] = None,
    no_system_prompt: bool = False,
) -> torch.Tensor:
    conv = default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    if no_system_prompt:
        conv.system = ""

    # Skip the first message if it is not from human
    if messages[0]["from"] != "human":
        messages = messages[1:]

    # Add a generation prompt if needed
    if add_generation_prompt:
        messages.append({"from": "gpt", "value": None})

    conv.messages = []
    for turn, message in enumerate(messages):
        role = roles[message["from"]]
        assert role == conv.roles[turn % 2]
        if overrides is not None and message["from"] in overrides:
            conv.append_message(role, overrides[message["from"]])
        else:
            conv.append_message(role, message["value"])

    return tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt")


def tokenize_conversation(
    messages: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    add_generation_prompt: bool = False,
    overrides: Optional[Dict[str, str]] = None,
    no_system_prompt: bool = False,
    return_ids_only=True,
) -> torch.Tensor:
    # Normalize the conversation before tokenization
    for message in messages:
        message["value"] = message["value"].strip()

    if default_conversation.sep_style != SeparatorStyle.AUTO:
        return tokenize_conversation_legacy(
            messages,
            tokenizer,
            add_generation_prompt=add_generation_prompt,
            overrides=overrides,
            no_system_prompt=no_system_prompt,
        )

    conversation = []
    for m in messages:
        message = {}
        if m["from"] == "human":
            message["role"] = "user"
        elif m["from"] == "gpt":
            message["role"] = "assistant"
        elif m["from"] == "system":
            message["role"] = "system"
            if no_system_prompt:
                raise ValueError("message[role]=system is not allowed when no_system_prompt is set to True.")
        else:
            raise ValueError(f"Unexpected sender '{m['from']}' in conversation entry.")

        message["content"] = m["value"]
        if overrides is not None and m["from"] in overrides:
            message["content"] = overrides[m["from"]]
        conversation.append(message)

    if no_system_prompt:
        conversation = [{"role": "system", "content": ""}] + conversation

    text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tokenizer_image_token(text, tokenizer, return_tensors="pt", return_ids=return_ids_only)


def _maybe_add_sentinel_token(tokenizer: transformers.PreTrainedTokenizer) -> None:
    if not hasattr(tokenizer, "sentinel_token"):
        tokenizer.add_tokens([SENTINEL_TOKEN], special_tokens=True)
        tokenizer.sentinel_token = SENTINEL_TOKEN
        tokenizer.sentinel_token_id = tokenizer.convert_tokens_to_ids(SENTINEL_TOKEN)


def preprocess_conversation(
    conversation: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    no_system_prompt: bool = False,
    retried: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    inputs = tokenize_conversation(conversation, tokenizer, no_system_prompt=no_system_prompt)
    labels = torch.ones_like(inputs) * IGNORE_INDEX

    # Generate the template by replacing the assistant's response with a sentinel.
    _maybe_add_sentinel_token(tokenizer)
    template = tokenize_conversation(
        conversation, tokenizer, overrides={"gpt": SENTINEL_TOKEN}, no_system_prompt=no_system_prompt
    )

    # Remove sentinel tokens from the template.
    mask = torch.ones_like(template, dtype=torch.bool)
    for k in range(template.size(0) - 1):
        if template[k] == tokenizer.sentinel_token_id:
            mask[k : k + 2] = False
            if k > 0 and retried:
                mask[k - 1] = False
    template = template[mask]

    # Match the tokenized conversation with the template (with no assistant's response).
    # Every token that is not matched will be included in the label for training.
    p = 0
    for k in range(inputs.size(0)):
        if p < template.size(0) and inputs[k] == template[p]:
            p += 1
        else:
            labels[k] = inputs[k]

    # Mask all tokens in the label if the template is not fully matched.
    if p < template.size(0):
        if not retried:
            return preprocess_conversation(
                conversation,
                tokenizer,
                no_system_prompt=no_system_prompt,
                retried=True,
            )
        print(f"Failed to process the conversation: '{conversation}'. All tokens will be masked in the label.")
        labels[:] = IGNORE_INDEX

    return {"input_ids": inputs, "labels": labels}


def infer_stop_tokens(tokenizer: transformers.PreTrainedTokenizer) -> List[str]:
    _maybe_add_sentinel_token(tokenizer)
    template = tokenize_conversation(DUMMY_CONVERSATION, tokenizer, overrides={"gpt": SENTINEL_TOKEN})

    stop_tokens = {tokenizer.eos_token}
    for k in range(template.size(0) - 1):
        if template[k] == tokenizer.sentinel_token_id:
            stop_token = tokenizer.decode(template[k + 1])
            stop_tokens.add(stop_token)
    return list(stop_tokens)