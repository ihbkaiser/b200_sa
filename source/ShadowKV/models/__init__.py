################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

from .glm import GLM
from .llama import Llama
from .qwen import Qwen2
from .qwen3 import Qwen3
from .phi3 import Phi3

# model_type -> wrapper. Checked before the substring heuristics below, because
# a local checkpoint directory is named by whoever downloaded it: a path such as
# /storage/.../qwen3-4b-instruct-2507 matches 'qwen' and would silently land on
# the Qwen2 wrapper, which computes the wrong head_dim for Qwen3.
MODEL_TYPE_TO_CLASS = {
    'llama': Llama,
    'qwen2': Qwen2,
    'qwen3': Qwen3,
    'phi3': Phi3,
    'chatglm': GLM,
    'glm': GLM,
}

def choose_model_class(model_name):
    try:
        from transformers import AutoConfig
        model_type = AutoConfig.from_pretrained(model_name, trust_remote_code=False).model_type
    except Exception:
        model_type = None
    if model_type in MODEL_TYPE_TO_CLASS:
        return MODEL_TYPE_TO_CLASS[model_type]

    if 'llama' in model_name.lower():
        return Llama
    elif 'glm' in model_name.lower():
        return GLM
    elif 'yi' in model_name.lower():
        return Llama
    elif 'qwen' in model_name.lower():
        return Qwen2
    elif 'phi' in model_name.lower():
        return Phi3
    else:
        raise ValueError(f"Model {model_name} not found")