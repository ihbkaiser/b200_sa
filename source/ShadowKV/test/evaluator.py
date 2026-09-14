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
from termcolor import colored
from tqdm import tqdm
import torch.distributed as dist
import pandas as pd
import json
import datetime

from data.dataset import Dataset
from models.base import LLM


class Evaluator:
    def __init__(self, dist_config):

        self.dist_config = dist_config

        # init final report
        self.all_stats = []

    def test(
        self,
        llm: LLM,
        dataset: Dataset,
        output_path: str,
        setting: str = 'baseline',
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        sample_seed: int | None = None,
    ):

        # mkdir if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self.dist_config.master_process:
            print(colored(f"[Test] {llm.model_name} on {dataset.dataset_name}, results saved to {output_path}", 'green'))

        if dataset.is_sharded == False:
            dataset.shard(self.dist_config.rank, self.dist_config.world_size)

        bsz = llm.batch_size
        scores = []
        preds = []

        # clear the file
        open(output_path, 'w').close()
        if self.dist_config.is_distributed:
            dist.barrier()

        progress_bar = tqdm(range(dataset.num_samples // bsz), desc='Testing', disable=self.dist_config.is_distributed and not self.dist_config.master_process)
        for i in range(dataset.num_samples // bsz):
            # KVPress resets the configured seed for every independent AIME
            # problem.  Match that protocol exactly instead of letting one
            # problem's sampled length perturb all later problems.
            if sample_seed is not None:
                torch.manual_seed(sample_seed)
            prompt = torch.cat([dataset.tokenized_prompts[i*bsz+j] for j in range(bsz)], dim=0)
            if 'persona' in dataset.dataset_name:
                assert bsz == 1
                llm.generate(prompt.to(llm.device), gen_len=0, verbose=False, top_p=0.9, temperature=0.0) # prefill ctx
                queries = dataset.queries[i]
                gts_list = dataset.gt[i]
                rets_list = []
                for query, gts in zip(queries, gts_list):
                    rets = llm.generate(llm.encode(query, template="chat"), cont=True, gen_len=dataset.gen_len, top_p=0.9, temperature=0.0)
                    rets_list.append(rets)
                    pred = rets[0]
                    if isinstance(gts, list):
                        if len(gts) == 1:
                            gts = gts[0]
                    # print(pred, gts, dataset.metric(pred, gts))
                    scores.append(dataset.metric(pred, gts))
                
            elif dataset.dataset_name.startswith('longbench/'):
                rets = llm.generate(
                    prompt.to(llm.device), gen_len=dataset.gen_len,
                    verbose=False, top_p=top_p, top_k=top_k,
                    temperature=temperature,
                )
                for (pred, gt, classes) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz], dataset.classes[i*bsz:(i+1)*bsz]):
                    scores.append(max([
                        dataset.metric(pred, g, all_classes=classes) for g in gt
                    ]))

            else:
                # LLM.generate samples the first output token from the prefill
                # logits before entering its `gen_len` decode loop.  Subtract
                # one here so AIME's published 32768 cap means 32768 output
                # tokens rather than 32769.  Other established evaluations
                # retain their historical behavior.
                generation_steps = (
                    dataset.gen_len - 1
                    if dataset.dataset_name == 'aime25' else dataset.gen_len
                )
                rets = llm.generate(
                    prompt.to(llm.device), gen_len=generation_steps,
                    verbose=False, top_p=top_p, top_k=top_k,
                    temperature=temperature,
                )
                for (pred, gt) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz]):
                    if isinstance(gt, list):
                        if len(gt) == 1:
                            gt = gt[0]
                    scores.append(dataset.metric(pred, gt))

            # Optional per-prompt trace for free-rate adaptive allocators.
            # This is intentionally evaluator-side: component_count is reset
            # for every prompt, so inspecting it only after the full dataset
            # silently reports the final prompt rather than the task mean.
            allocation_trace = os.environ.get("STREAMING_ALLOCATION_TRACE_OUT")
            if allocation_trace and hasattr(llm.kv_cache, "component_count"):
                histogram = {}
                total_blocks = 0
                total_components = 0
                layer_means = []
                layer_histograms = []
                layer_head_means = []
                head_histograms = None
                head_component_sums = None
                head_block_counts = None
                for layer_idx, state in enumerate(llm.kv_cache.block_state):
                    first, last = state.candidate_block_range
                    counts = llm.kv_cache.component_count[
                        layer_idx, :, :, first:last
                    ].long()
                    if counts.numel() == 0:
                        continue
                    layer_histogram = {}
                    values, frequencies = counts.unique(return_counts=True)
                    for value, frequency in zip(values.tolist(), frequencies.tolist()):
                        histogram[str(value)] = histogram.get(str(value), 0) + frequency
                        layer_histogram[str(value)] = frequency
                    total_blocks += counts.numel()
                    total_components += int(counts.sum().item())
                    layer_means.append(float(counts.float().mean().item()))
                    layer_histograms.append(layer_histogram)
                    per_head = counts.float().mean(dim=(0, 2))
                    layer_head_means.append(per_head.tolist())
                    num_heads = counts.shape[1]
                    if head_histograms is None:
                        head_histograms = [dict() for _ in range(num_heads)]
                        head_component_sums = [0 for _ in range(num_heads)]
                        head_block_counts = [0 for _ in range(num_heads)]
                    for head_idx in range(num_heads):
                        head_counts = counts[:, head_idx].reshape(-1)
                        head_component_sums[head_idx] += int(
                            head_counts.sum().item()
                        )
                        head_block_counts[head_idx] += head_counts.numel()
                        head_values, head_frequencies = head_counts.unique(
                            return_counts=True
                        )
                        for value, frequency in zip(
                            head_values.tolist(), head_frequencies.tolist()
                        ):
                            key = str(value)
                            head_histograms[head_idx][key] = (
                                head_histograms[head_idx].get(key, 0) + frequency
                            )
                record = {
                    "dataset": dataset.dataset_name,
                    "sample_index": i,
                    "total_blocks": total_blocks,
                    "total_components": total_components,
                    "mean_components_per_block": (
                        total_components / total_blocks if total_blocks else None
                    ),
                    "histogram": histogram,
                    "layer_means": layer_means,
                    "layer_histograms": layer_histograms,
                    "layer_head_means": layer_head_means,
                    "head_means": (
                        [
                            component_sum / block_count
                            for component_sum, block_count in zip(
                                head_component_sums, head_block_counts
                            )
                        ]
                        if head_component_sums is not None else []
                    ),
                    "head_histograms": head_histograms or [],
                    "layer_mean_min": min(layer_means) if layer_means else None,
                    "layer_mean_max": max(layer_means) if layer_means else None,
                }
                os.makedirs(os.path.dirname(allocation_trace) or ".", exist_ok=True)
                with open(allocation_trace, "a", encoding="utf8") as handle:
                    handle.write(json.dumps(record) + "\n")
            
            progress_bar.update(1)
            avg_score = sum(scores) / len(scores)
            progress_bar.set_postfix({'avg_score': avg_score})

            if dataset.dataset_name == 'niah':
                preds = {
                        "context_length": dataset.ctx_len[i*bsz:(i+1)*bsz],
                        "depth_percent": dataset.depth_pct[i*bsz:(i+1)*bsz],
                        "response": rets,
                        "answer": dataset.gt[i*bsz:(i+1)*bsz],
                        "correct": scores[i*bsz:(i+1)*bsz],
                        "avg_score": avg_score,
                    }
            elif 'persona' in dataset.dataset_name:
                preds = {
                        "query": queries,
                        "response": rets_list,
                        "answer": dataset.gt[i*bsz:(i+1)*bsz],
                        "correct": scores,
                        "avg_score": avg_score,
                    }
            else:
                preds = {
                        "prediction": rets,
                        "ground_truth": dataset.gt[i*bsz:(i+1)*bsz],
                        "correct": scores,
                        "avg_score": avg_score,
                    }
                if dataset.metadata is not None:
                    preds["metadata"] = dataset.metadata[i*bsz:(i+1)*bsz]

            with open(output_path, "a", encoding="utf8") as fout:
                fout.write(json.dumps(preds, ensure_ascii=False) + "\n")
            # if self.dist_config.is_distributed:
            #     dist.barrier()

        progress_bar.close()
        avg_score = sum(scores) / len(scores)

        self.all_stats.append(
            {
                'model': llm.model_name,
                'dataset': dataset.dataset_name,
                'samples': dataset.num_samples,
                f'{setting}': avg_score,
            }
        )
        if self.dist_config.is_distributed:
            dist.barrier()

    def summarize(self):
        df = pd.DataFrame(self.all_stats)

        if self.dist_config.is_distributed:
            dist.barrier()
            output = [None for _ in range(self.dist_config.world_size)]
            dist.gather_object(df, output if self.dist_config.master_process else None, dst=0)
            dist.barrier()
            if self.dist_config.master_process:
                df = pd.concat(output)
                # Define the columns you want to calculate the mean for (excluding 'samples')
                setting_columns = [col for col in df.columns if col not in ['model', 'dataset', 'samples']]

                # Calculate the sum for 'samples'
                samples_sum = df.groupby(['model', 'dataset'])['samples'].sum()

                # Define the aggregation dictionary for other settings
                agg_dict = {col: 'mean' for col in setting_columns}

                # Calculate the weighted mean for each setting column based on 'samples'
                weighted_means = df.groupby(['model', 'dataset']).apply(lambda x: pd.Series({
                    col: (x[col] * x['samples']).sum() / x['samples'].sum() for col in setting_columns
                }))

                # Combine the weighted means with the sum of samples
                df = weighted_means.join(samples_sum).reset_index()
        
        if self.dist_config.master_process:
            # add a row for the average
            numeric_columns = df.select_dtypes(include='number')
            mean_values = numeric_columns.mean()
            mean_row = pd.DataFrame({col: [mean_values[col] if col in mean_values else 'mean'] for col in df.columns})
            df_with_mean = pd.concat([df, mean_row], ignore_index=True)
            print(df_with_mean.to_markdown(index=False))
