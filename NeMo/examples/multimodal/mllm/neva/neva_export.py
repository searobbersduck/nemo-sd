# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from omegaconf.omegaconf import OmegaConf
from polygraphy.backend.trt import CreateConfig, Profile, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.logger import G_LOGGER

from nemo.core.classes.exportable import Exportable
from nemo.core.config import hydra_runner
from nemo.core.neural_types import ChannelType, LogitsType, NeuralType

G_LOGGER.module_severity = G_LOGGER.EXTRA_VERBOSE
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor, CLIPVisionModel

from nemo.core.classes.exportable import Exportable
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.core.neural_types import ChannelType, LogitsType, NeuralType
from nemo.export import TensorRTLLM

LOGGER = logging.getLogger(__name__)


def build_vision_encoder(model_path, clip_path, precision, bs_min, bs_opt, bs_max, out_dir):
    torch_precision = torch.bfloat16 if precision in ['bf16', 'bf16-mixed'] else torch.float16

    with tempfile.TemporaryDirectory() as temp:
        LOGGER.info('Extracting model')
        connector = SaveRestoreConnector()
        connector._unpack_nemo_file(path2file=model_path, out_folder=temp)
        config_yaml = os.path.join(temp, connector.model_config_yaml)
        config = OmegaConf.load(config_yaml)
        if config.tensor_model_parallel_size > 1:
            path = os.path.join(temp, 'mp_rank_00', connector.model_weights_ckpt)
        else:
            path = os.path.join(temp, connector.model_weights_ckpt)
        state_dict = connector._load_state_dict_from_disk(path)
        LOGGER.info('Done')

    vision_connector = torch.nn.Linear(config.mm_cfg.vision_encoder.hidden_size, config.hidden_size, bias=True,)
    vision_encoder = CLIPVisionModel.from_pretrained(clip_path, torch_dtype=torch_precision)
    image_size = vision_encoder.vision_model.config.image_size

    if 'model.vision_connector.weight' in state_dict:
        new_state_dict = {
            'weight': state_dict['model.vision_connector.weight'],
            'bias': state_dict['model.vision_connector.bias'],
        }
    else:
        new_state_dict = {
            'weight': state_dict[
                'model.language_model.embedding.word_embeddings.adapter_layer.mm_linear_adapter.linear.weight'
            ],
            'bias': state_dict[
                'model.language_model.embedding.word_embeddings.adapter_layer.mm_linear_adapter.linear.bias'
            ],
        }

    vision_connector.load_state_dict(new_state_dict)
    vision_connector = vision_connector.to(dtype=torch_precision)

    class VisionEncoderWrapper(torch.nn.Module, Exportable):
        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            vision_x = self.encoder(images, output_hidden_states=True)
            vision_x = vision_x.hidden_states[-2]
            vision_x = vision_x[:, 1:]
            vision_x = self.connector(vision_x)
            return vision_x

        # For onnx export
        def input_example(self, max_batch=8):
            sample = next(self.parameters())
            images = torch.randn(max_batch, 3, image_size, image_size, device=sample.device, dtype=sample.dtype)
            return (images,)

        @property
        def input_types(self):
            return {'images': NeuralType(('B', 'C', 'H', 'W'), ChannelType())}

        @property
        def output_types(self):
            return {'features': NeuralType(('B', 'S', 'D'), LogitsType())}

        @property
        def input_names(self):
            return ['images']

        @property
        def output_names(self):
            return ['features']

    wrapper = VisionEncoderWrapper(vision_encoder, vision_connector)

    os.makedirs(f'/tmp/onnx/', exist_ok=True)
    dynamic_axes = {'images': {0: 'B'}}

    LOGGER.info('Exporting ONNX')
    wrapper.export(f'/tmp/onnx/vision_encoder.onnx', dynamic_axes=dynamic_axes, onnx_opset_version=17)
    LOGGER.info('Done')

    bsmin_example = wrapper.input_example(max_batch=bs_min)
    bsopt_example = wrapper.input_example(max_batch=bs_opt)
    bsmax_example = wrapper.input_example(max_batch=bs_max)

    input_profile = {}
    input_profile['images'] = [
        tuple(bsmin_example[0].shape),
        tuple(bsopt_example[0].shape),
        tuple(bsmax_example[0].shape),
    ]

    p = Profile()
    if input_profile:
        for name, dims in input_profile.items():
            assert len(dims) == 3
            p.add(name, min=dims[0], opt=dims[1], max=dims[2])

    LOGGER.info('Exporting TRT')
    engine = engine_from_network(
        network_from_onnx_path('/tmp/onnx/vision_encoder.onnx'),
        config=CreateConfig(
            tf32=precision in [32, '32', '32-true'],
            fp16=precision in [16, '16', '16-mixed'],
            bf16=precision in ['bf16', 'bf16-mixed'],
            profiles=[p],
        ),
    )
    save_engine(engine, path=os.path.join(out_dir, 'vision_encoder.plan'))

    processor = CLIPImageProcessor.from_pretrained(clip_path)
    processor.save_pretrained(out_dir)
    LOGGER.info('Done')


def build_trtllm_engines(in_file, out_dir, tensor_parallelism, max_input_len, max_output_len, max_batch_size):
    trt_llm_exporter = TensorRTLLM(model_dir=out_dir)
    trt_llm_exporter.export(
        nemo_checkpoint_path=in_file,
        model_type="llama",
        n_gpus=tensor_parallelism,
        max_input_token=max_input_len,
        max_output_token=max_output_len,
        max_batch_size=max_batch_size,
        max_prompt_embedding_table_size=max_batch_size * max_input_len,
    )
    LOGGER.info('Done')


@hydra_runner(config_path='conf', config_name='neva_export')
def main(cfg):
    precision = cfg.model.get('precision', 16)
    assert precision != 32, 'FP32 export not supported'
    plan_dir = os.path.join(cfg.infer.out_dir, 'plan')

    os.makedirs(plan_dir, exist_ok=True)
    LOGGER.info('Building TRT-LLM engines')
    build_trtllm_engines(
        cfg.model.restore_from_path,
        plan_dir,
        cfg.infer.llm.get('tensor_parallelism', 1),
        cfg.infer.llm.get('max_input_len', 2048),
        cfg.infer.llm.get('max_output_len', 2048),
        cfg.infer.llm.get('max_batch_size', 1),
    )

    LOGGER.info('Building vision TRT engine')
    build_vision_encoder(
        cfg.model.restore_from_path,
        cfg.infer.vision.clip,
        32,
        cfg.infer.vision.get('min_batch_size', 1),
        cfg.infer.vision.get('opt_batch_size', 1),
        cfg.infer.vision.get('max_batch_size', 1),
        plan_dir,
    )


if __name__ == '__main__':
    main()
