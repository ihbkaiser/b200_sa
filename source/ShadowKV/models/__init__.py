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

"""Model registry with lazy imports for optional CUDA-backed components.

Artifact validators and CPU reference tests should be able to import a small
router module without first loading the ShadowKV CUDA extension.  The public
class names remain available through module-level lazy attribute resolution.
"""

from importlib import import_module


_MODEL_MODULES = {
    "GLM": ".glm",
    "Llama": ".llama",
    "Qwen2": ".qwen",
    "Qwen3": ".qwen3",
    "Phi3": ".phi3",
}


def __getattr__(name):
    module_name = _MODEL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

# model_type -> wrapper. Checked before the substring heuristics below, because
# a local checkpoint directory is named by whoever downloaded it: a path such as
# /storage/.../qwen3-4b-instruct-2507 matches 'qwen' and would silently land on
# the Qwen2 wrapper, which computes the wrong head_dim for Qwen3.
MODEL_TYPE_TO_CLASS = {
    'llama': 'Llama',
    'qwen2': 'Qwen2',
    'qwen3': 'Qwen3',
    'phi3': 'Phi3',
    'chatglm': 'GLM',
    'glm': 'GLM',
}


def _resolve_model_class(name):
    return globals().get(name) or getattr(__import__(__name__, fromlist=[name]), name)

def choose_model_class(model_name):
    try:
        from transformers import AutoConfig
        model_type = AutoConfig.from_pretrained(model_name, trust_remote_code=False).model_type
    except Exception:
        model_type = None
    if model_type in MODEL_TYPE_TO_CLASS:
        return _resolve_model_class(MODEL_TYPE_TO_CLASS[model_type])

    if 'llama' in model_name.lower():
        return _resolve_model_class('Llama')
    elif 'glm' in model_name.lower():
        return _resolve_model_class('GLM')
    elif 'yi' in model_name.lower():
        return _resolve_model_class('Llama')
    elif 'qwen' in model_name.lower():
        return _resolve_model_class('Qwen2')
    elif 'phi' in model_name.lower():
        return _resolve_model_class('Phi3')
    else:
        raise ValueError(f"Model {model_name} not found")
