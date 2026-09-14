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

import os

import torch
from datasets import load_dataset
from termcolor import colored
import random
import numpy as np

# RULER
from .metrics import needle_score, string_match_part, multi_number, multi_words
from .benchmark_metrics import (
    LONG_BENCH_METRICS,
    score_aime,
    score_gpqa,
    score_longbench_v2,
    score_math500,
)

# NIAH
from data.utils import (
    NIAH_TEMPLATE,
    RANDOM_NEEDLE_CITIES,
    create_contexts,
    generate_random_number,
    read_context_files,
    truncate_input,
)

METRICS_FN = {
    'niah': needle_score,
    'multi': multi_number,
    'vt': multi_words,
    'cwe': multi_words,
    'fwe': multi_words,
    'qa': string_match_part,
}

GEN_LEN = {
    'niah': 64,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32,
}

DATADIR = {
    'ruler': os.environ.get('SHADOWKV_RULER_DATA_ROOT', 'data/ruler/data'),
    'niah': 'data/niah/data',
}

DEFAULT_LONG_BENCH_PATH = (
    "/storage/baonn/huggingface/hub/"
    "datasets--Xnhyacinth--LongBench/snapshots/"
    "2e9ade51ebf45d98942056c0716234f9d5d257d5"
)
DEFAULT_AIME25_PATH = "/storage/baonn/kvpress_aime25_local"
DEFAULT_GPQA_PATH = "/storage/baonn/kvpress_gpqa"
DEFAULT_LONG_BENCH_V2_PATH = "simonjegou/LongBench-v2"
DEFAULT_MATH500_PATH = "alessiodevoto/math500"


class Dataset:
    def __init__(self, dataset_name, tokenizer, datalen, num_samples, rank=0, world_size=1):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.datalen = datalen
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.is_sharded = False
        self.classes = None
        self.metadata = None
        self._row_gen_len = None

        if dataset_name == 'niah':
            self.tokenized_prompts, self.gt, self.ctx_len, self.depth_pct = self.get_dataset()
        else:
            self.tokenized_prompts, self.gt = self.get_dataset()
        
        self.num_samples = len(self.tokenized_prompts)
        self.gen_len = self.get_gen_len()
        self.metric = self.get_metric()

    def __str__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __repr__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __len__(self) -> int:
        return self.num_samples

    def shard(self, rank, world_size):
        if world_size > 1:
            shard_size = self.num_samples // world_size
            start = rank * shard_size
            end = start + shard_size if rank != world_size - 1 else self.num_samples
            shard_tokenized_prompts, shard_gt = self.tokenized_prompts[start:end], self.gt[start:end]
            self.tokenized_prompts = shard_tokenized_prompts
            self.gt = shard_gt
            if self.classes is not None:
                self.classes = self.classes[start:end]
            if self.metadata is not None:
                self.metadata = self.metadata[start:end]
            self.num_samples = len(shard_tokenized_prompts)

        self.is_sharded = True

    def get_gen_len(self):
        if self._row_gen_len is not None:
            return self._row_gen_len
        if 'niah' == self.dataset_name:
            return 10
        elif 'niah' in self.dataset_name:
            return 128
        elif 'vt' in self.dataset_name:
            return 30
        elif 'cwe' in self.dataset_name:
            return 120
        elif 'fwe' in self.dataset_name:
            return 50
        elif 'qa' in self.dataset_name:
            return 32
        else:
            raise Exception("Gen len not found")

    def __getitem__(self, idx):
        if 'persona' in self.dataset_name:
            return self.tokenized_prompts[idx], self.queries[idx], self.gt[idx]
        return self.tokenized_prompts[idx], self.gt[idx]

    def get_metric(self):
        if self.dataset_name.startswith('longbench/'):
            task = self.dataset_name.split('/', 1)[1]
            return LONG_BENCH_METRICS[task]
        elif self.dataset_name == 'aime25':
            return score_aime
        elif self.dataset_name == 'longbench-v2':
            return score_longbench_v2
        elif self.dataset_name == 'math500':
            return score_math500
        elif self.dataset_name.startswith('gpqa'):
            return score_gpqa
        elif 'multiquery' in self.dataset_name or 'multivalue' in self.dataset_name:
            return METRICS_FN['multi']
        elif 'niah' in self.dataset_name:
            return METRICS_FN['niah']
        elif 'vt' in self.dataset_name:
            return METRICS_FN['vt']
        elif 'cwe' in self.dataset_name:
            return METRICS_FN['cwe']
        elif 'fwe' in self.dataset_name:
            return METRICS_FN['fwe']
        elif 'qa' in self.dataset_name:
            return METRICS_FN['qa']
        else:
            raise Exception("Metric not found")

    def get_dataset(self):
        if 'ruler' in self.dataset_name: # ruler/xxx
            task = self.dataset_name.split('/')[-1]
            assert self.datalen in [4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024], "Only support datalen of 4k, 8k, 16k, 32k, 64k, 128k, or 256k"

            tokenizer_name = self.tokenizer.name_or_path.lower()
            if 'llama-3' in tokenizer_name or 'deepseek' in tokenizer_name:
                model_dir = 'llama-3'
            elif 'yi' in self.tokenizer.name_or_path.lower():
                model_dir = 'yi'
            elif 'lwm' in self.tokenizer.name_or_path.lower():
                model_dir = 'lwm'
            elif 'glm' in self.tokenizer.name_or_path.lower():
                model_dir = 'glm'
            elif 'qwen' in self.tokenizer.name_or_path.lower():
                model_dir = 'qwen'
            elif 'phi' in self.tokenizer.name_or_path.lower():
                model_dir = 'phi'
            else:
                raise Exception("Model not found", self.tokenizer.name_or_path)

            dataset = load_dataset("json", data_files=f'{DATADIR["ruler"]}/{model_dir}/{self.datalen}/{task}/validation.jsonl', split='train')
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            gt = []

            for i in range(self.num_samples):
                input_text = dataset[i]['input']
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
                tokenized_prompts.append(input_ids)
                gt.append(dataset[i]['outputs'])

            return tokenized_prompts, gt

        elif self.dataset_name.startswith('longbench/'):
            task = self.dataset_name.split('/', 1)[1]
            path = os.environ.get('SHADOWKV_LONGBENCH_PATH', DEFAULT_LONG_BENCH_PATH)
            dataset = load_dataset(path, data_dir=task, split='test')
            return self._load_benchmark_rows(dataset, answer_column='answers', with_classes=True)

        elif self.dataset_name == 'longbench-v2':
            path = os.environ.get(
                'SHADOWKV_LONGBENCH_V2_PATH', DEFAULT_LONG_BENCH_V2_PATH
            )
            dataset = load_dataset(path, split='test')
            length_filter = os.environ.get(
                'SHADOWKV_LONGBENCH_V2_LENGTH_FILTER', ''
            ).strip()
            if length_filter:
                allowed = {
                    item.strip().lower()
                    for item in length_filter.split(',') if item.strip()
                }
                invalid = allowed - {'short', 'medium', 'long'}
                if invalid:
                    raise ValueError(
                        'invalid LongBench-v2 length filter: '
                        + ','.join(sorted(invalid))
                    )
                indices = [
                    idx for idx, value in enumerate(dataset['length'])
                    if str(value).lower() in allowed
                ]
                dataset = dataset.select(indices)
            return self._load_benchmark_rows(
                dataset,
                answer_column='answer',
                metadata_columns=(
                    '_id', 'domain', 'sub_domain', 'difficulty', 'length'
                ),
            )

        elif self.dataset_name == 'aime25':
            path = os.environ.get('SHADOWKV_AIME25_PATH', DEFAULT_AIME25_PATH)
            dataset = load_dataset(path, split='test')
            return self._load_benchmark_rows(dataset, answer_column='answer')

        elif self.dataset_name == 'math500':
            path = os.environ.get('SHADOWKV_MATH500_PATH', DEFAULT_MATH500_PATH)
            dataset = load_dataset(path, split='test')
            return self._load_benchmark_rows(
                dataset,
                answer_column='answer',
                metadata_columns=('unique_id', 'subject', 'level'),
            )

        elif self.dataset_name.startswith('gpqa'):
            parts = self.dataset_name.split('/', 1)
            subset = parts[1] if len(parts) == 2 else 'diamond'
            path = os.environ.get('SHADOWKV_GPQA_PATH', DEFAULT_GPQA_PATH)
            dataset = load_dataset(path, data_dir=subset, split='test')
            return self._load_benchmark_rows(dataset, answer_column='answer')

        elif self.dataset_name == 'niah':
            print(colored(f"[Warning] NIAH dataset cannot set # samples, it is up to world_size, which is set to {self.world_size}", 'red'))
            
            haystack_file = f'{DATADIR["niah"]}/pg19_mini.jsonl'
            context_lengths_min = 16*1024
            context_lengths_max = self.datalen
            n_context_length_intervals = 15
            n_document_depth_intervals = 10  # position of the needle in the haystack
            n_rounds = 1 # max(1, 4 // self.world_size) # 8 rounds in total assume we have 8xGPUs
            needle = "\nThe special magic {city} number is: {rnd_number}\n"
            retrieval_question="What is the special magic {} number?"
            rnd_number_digits = 7

            context_lengths = np.round(
                np.linspace(
                    context_lengths_min,
                    context_lengths_max,
                    num=n_context_length_intervals,
                    endpoint=True,
                )
            ).astype(int)

            document_depth_percents = np.round( # we use linear scale here
                np.linspace(
                    0,
                    100,
                    num=n_document_depth_intervals,
                    endpoint=True,
                )
            ).astype(int)

            self.is_sharded = True # we shard the data during init dataset
            
            full_contexts = read_context_files(n=n_rounds, context_lengths=context_lengths, haystack_file=haystack_file, tokenizer=self.tokenizer)
            full_tokens = [
                self.tokenizer.encode(full_context, add_special_tokens=False) for full_context in full_contexts
            ]

            tokenized_prompts = []
            gt = []
            ctx_len = []
            depth_pct = []

            for context_length in context_lengths:
                trim_contexts = [
                    self.tokenizer.decode(full_token[:context_length], skip_special_tokens=True)
                    for full_token in full_tokens
                ]
                contexts = []
                for depth_percent in document_depth_percents:
                    for i in range(n_rounds):
                        random_city = random.choice(RANDOM_NEEDLE_CITIES)
                        insert_needle = True
                        needle_rnd_number = str(generate_random_number(rnd_number_digits))
                        context = create_contexts(
                            needle_rnd_number=needle_rnd_number,
                            insert_needle=insert_needle,
                            random_city=random_city,
                            trim_context=trim_contexts[i],
                            context_length=context_length,
                            depth_percent=depth_percent,
                            needle=needle,
                            retrieval_question=retrieval_question,
                            tokenizer=self.tokenizer,
                            final_context_length_buffer=32,
                        )
                        contexts.append(context)

                for context in contexts:
                    prompt = NIAH_TEMPLATE.format(
                        context=context["context"], question=context["question"]
                    )
                    input_tensor = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
                    tokenized_prompts.append(input_tensor.input_ids)
                    gt.append(context["needle_rnd_number"])
                    ctx_len.append(context["context_length"])
                    depth_pct.append(context["depth_percent"])
            
            return tokenized_prompts, gt, ctx_len, depth_pct

        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not found; supported families are "
                "ruler/<task>, longbench/<task>, longbench-v2, aime25, "
                "math500, gpqa[/subset], and niah"
            )

    def _tokenize_benchmark_prompt(self, context, question, answer_prefix):
        """Render a benchmark row, keeping context and question ID spans separate."""
        context = str(context)
        question = str(question)
        answer_prefix = str(answer_prefix)
        if self.tokenizer.chat_template is None:
            context_text = (getattr(self.tokenizer, 'bos_token', '') or '') + context
            question_text = question + "\n" + answer_prefix
        else:
            separator = '#' * (len(context) + 10)
            kwargs = dict(
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}], **kwargs
            )
            context_text, found, question_suffix = rendered.partition(separator)
            if not found:
                raise RuntimeError("chat template did not preserve the KVPress separator")
            question_text = question + question_suffix + answer_prefix

        context_ids = self.tokenizer.encode(
            context_text, return_tensors='pt', add_special_tokens=False
        )
        question_ids = self.tokenizer.encode(
            question_text, return_tensors='pt', add_special_tokens=False
        )

        # LongBench-v2's official evaluator truncates the *complete prompt* in
        # the middle, retaining both its beginning and its end.  The latter is
        # important because it contains the question and answer choices.  Do
        # not use KVPress's context-prefix truncation here: that changes which
        # evidence is visible and is not the benchmark's published protocol.
        if self.dataset_name == 'longbench-v2':
            prompt_ids = torch.cat((context_ids, question_ids), dim=1)
            return truncate_input(prompt_ids, max_length=self.datalen, manner='middle')

        if context_ids.shape[1] > self.datalen:
            context_ids = context_ids[:, :self.datalen]
        return torch.cat((context_ids, question_ids), dim=1)

    def _load_benchmark_rows(
        self, dataset, answer_column, with_classes=False, metadata_columns=()
    ):
        limit = min(self.num_samples, len(dataset)) if self.num_samples > 0 else len(dataset)
        rows = dataset.select(range(limit))
        gen_lens = {int(value) for value in rows['max_new_tokens']}
        if len(gen_lens) != 1:
            raise ValueError(
                f"{self.dataset_name} must have one max_new_tokens value, got {sorted(gen_lens)}"
            )
        self._row_gen_len = gen_lens.pop()
        tokenized_prompts = []
        gt = []
        classes = []
        metadata = []
        for row in rows:
            tokenized_prompts.append(self._tokenize_benchmark_prompt(
                row['context'], row['question'], row['answer_prefix']
            ))
            gt.append(row[answer_column])
            if with_classes:
                classes.append(row.get('all_classes'))
            if metadata_columns:
                metadata.append({key: row.get(key) for key in metadata_columns})
        if with_classes:
            self.classes = classes
        if metadata_columns:
            self.metadata = metadata
        return tokenized_prompts, gt
