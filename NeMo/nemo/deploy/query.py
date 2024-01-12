# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import typing

import numpy as np
from pytriton.client import ModelClient

from .utils import str_list2numpy


class NemoQuery:
    """
    Sends a query to Triton for LLM inference

    Example:
        from nemo.deploy import NemoQuery

        nq = NemoQuery(url="localhost", model_name="GPT-2B")

        prompts = ["hello, testing GPT inference", "another GPT inference test?"]
        output = nq.query_llm(
            prompts=prompts,
            max_output_len=100,
            top_k=1,
            top_p=0.0,
            temperature=0.0,
        )
        print("prompts: ", prompts)
    """

    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name

    def query_llm(
        self, prompts, max_output_token=512, top_k=1, top_p=0.0, temperature=1.0, init_timeout=600.0,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            prompts (List(str)): list of sentences.
            max_output_token (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            init_timeout (flat): timeout for the connection.
        """

        prompts = str_list2numpy(prompts)
        max_output_token = np.full(prompts.shape, max_output_token, dtype=np.int_)
        top_k = np.full(prompts.shape, top_k, dtype=np.int_)
        top_p = np.full(prompts.shape, top_p, dtype=np.single)
        temperature = np.full(prompts.shape, temperature, dtype=np.single)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            result_dict = client.infer_batch(
                prompts=prompts, max_output_token=max_output_token, top_k=top_k, top_p=top_p, temperature=temperature,
            )
            output_type = client.model_config.outputs[0].dtype

        if output_type == np.bytes_:
            sentences = np.char.decode(result_dict["outputs"].astype("bytes"), "utf-8")
            return sentences
        else:
            return result_dict["outputs"]
