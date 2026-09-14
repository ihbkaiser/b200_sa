#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <climits>

namespace {

__global__ void gather_kv_16byte_kernel(
    const uint4* __restrict__ source_k,
    const uint4* __restrict__ source_v,
    const int64_t* __restrict__ positions,
    uint4* __restrict__ destination_k,
    uint4* __restrict__ destination_v,
    int heads,
    int source_length,
    int destination_length,
    int selected_count,
    int vectors_per_token) {
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  const int selected = blockIdx.z;
  const int vector = threadIdx.x;
  if (vector >= vectors_per_token) return;

  const int64_t position = positions[
      (static_cast<int64_t>(batch) * heads + head) * selected_count + selected];
  const int64_t source_offset =
      ((static_cast<int64_t>(batch) * heads + head) * source_length + position) *
      vectors_per_token + vector;
  const int64_t destination_offset =
      ((static_cast<int64_t>(batch) * heads + head) * destination_length + selected) *
      vectors_per_token + vector;
  destination_k[destination_offset] = source_k[source_offset];
  destination_v[destination_offset] = source_v[source_offset];
}

__global__ void append_kv_16byte_kernel(
    const uint4* __restrict__ source_k,
    const uint4* __restrict__ source_v,
    uint4* __restrict__ destination_k,
    uint4* __restrict__ destination_v,
    uint4* __restrict__ staging_k,
    int heads,
    int incoming,
    int destination_length,
    int staging_length,
    int destination_start,
    int staging_start,
    int vectors_per_token) {
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  const int token = blockIdx.z;
  const int vector = threadIdx.x;
  if (vector >= vectors_per_token) return;

  const int64_t source_offset =
      ((static_cast<int64_t>(batch) * heads + head) * incoming + token) *
      vectors_per_token + vector;
  const int64_t destination_offset =
      ((static_cast<int64_t>(batch) * heads + head) * destination_length +
       destination_start + token) * vectors_per_token + vector;
  const int64_t staging_offset =
      ((static_cast<int64_t>(batch) * heads + head) * staging_length +
       staging_start + token) * vectors_per_token + vector;
  const uint4 key = source_k[source_offset];
  destination_k[destination_offset] = key;
  destination_v[destination_offset] = source_v[source_offset];
  staging_k[staging_offset] = key;
}

__global__ void gather_blocks_reuse_16byte_kernel(
    const uint4* __restrict__ source_k,
    const uint4* __restrict__ source_v,
    const uint4* __restrict__ previous_k,
    const uint4* __restrict__ previous_v,
    const int64_t* __restrict__ previous_blocks,
    const int64_t* __restrict__ current_blocks,
    uint4* __restrict__ destination_k,
    uint4* __restrict__ destination_v,
    int64_t* __restrict__ destination_blocks,
    int heads,
    int source_length,
    int destination_length,
    int selected_blocks,
    int block_size,
    int vectors_per_token,
    int range0_start,
    int range0_count,
    int range1_start,
    int range1_count,
    bool with_keys) {
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  const int task = blockIdx.z;
  const int thread = threadIdx.x;
  const int64_t bh = static_cast<int64_t>(batch) * heads + head;
  const int exact_count = range0_count + range1_count;
  __shared__ int previous_slot;

  if (task < selected_blocks) {
    const int64_t block = current_blocks[bh * selected_blocks + task];
    // Matching used to be a serial scan by thread 0.  One thread scanning
    // `selected_blocks` entries makes the whole launch quadratic in the slot
    // count: fine at a block-sized retrieval unit (512 slots at budget 4096),
    // but at a one-token unit there are 4096 slots and the scan costs more
    // than the transfer it saves.  Scan strided across the thread block
    // instead; atomicMin keeps the lowest match, which is what the break did.
    if (thread == 0) {
      previous_slot = INT_MAX;
      destination_blocks[bh * selected_blocks + task] = block;
    }
    __syncthreads();
    for (int slot = thread; slot < selected_blocks; slot += blockDim.x) {
      if (previous_blocks[bh * selected_blocks + slot] == block) {
        atomicMin(&previous_slot, slot);
      }
    }
    __syncthreads();
    const int match = (previous_slot == INT_MAX) ? -1 : previous_slot;

    const int vectors_per_block = block_size * vectors_per_token;
    if (thread >= vectors_per_block) return;
    const int token = thread / vectors_per_token;
    const int vector = thread - token * vectors_per_token;
    const int64_t destination_offset =
        (bh * destination_length + task * block_size + token) *
            vectors_per_token + vector;
    if (match >= 0) {
      const int64_t previous_offset =
          (bh * destination_length + match * block_size + token) *
              vectors_per_token + vector;
      if (with_keys) {
        destination_k[destination_offset] = previous_k[previous_offset];
      }
      destination_v[destination_offset] = previous_v[previous_offset];
    } else {
      const int64_t position = block * block_size + token;
      const int64_t source_offset =
          (bh * source_length + position) * vectors_per_token + vector;
      if (with_keys) {
        destination_k[destination_offset] = source_k[source_offset];
      }
      destination_v[destination_offset] = source_v[source_offset];
    }
    return;
  }

  const int exact = task - selected_blocks;
  if (exact >= exact_count || thread >= vectors_per_token) return;
  const int position = exact < range0_count
      ? range0_start + exact
      : range1_start + exact - range0_count;
  const int output_token = selected_blocks * block_size + exact;
  const int64_t source_offset =
      (bh * source_length + position) * vectors_per_token + thread;
  const int64_t destination_offset =
      (bh * destination_length + output_token) * vectors_per_token + thread;
  if (with_keys) {
    destination_k[destination_offset] = source_k[source_offset];
  }
  destination_v[destination_offset] = source_v[source_offset];
}

}  // namespace

void gather_kv(
    const torch::Tensor& source_k,
    const torch::Tensor& source_v,
    const torch::Tensor& positions,
    torch::Tensor& destination_k,
    torch::Tensor& destination_v) {
  TORCH_CHECK(source_k.device().is_cpu() && source_v.device().is_cpu(),
              "sources must be CPU tensors");
  TORCH_CHECK(source_k.is_pinned() && source_v.is_pinned(),
              "sources must be pinned");
  TORCH_CHECK(destination_k.is_cuda() && destination_v.is_cuda(),
              "destinations must be CUDA tensors");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64,
              "positions must be CUDA int64");
  TORCH_CHECK(source_k.is_contiguous() && source_v.is_contiguous() &&
                  destination_k.is_contiguous() && destination_v.is_contiguous() &&
                  positions.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(source_k.scalar_type() == destination_k.scalar_type() &&
                  source_v.scalar_type() == destination_v.scalar_type(),
              "source and destination dtypes must match");
  TORCH_CHECK(source_k.element_size() == 2,
              "kernel supports FP16/BF16 (two-byte elements)");

  const int batch = source_k.size(0);
  const int heads = source_k.size(1);
  const int source_length = source_k.size(2);
  const int head_dim = source_k.size(3);
  const int selected_count = positions.size(2);
  const int destination_length = destination_k.size(2);
  TORCH_CHECK((head_dim * 2) % 16 == 0,
              "head dimension byte width must be divisible by 16");
  const int vectors_per_token = head_dim * 2 / 16;
  TORCH_CHECK(vectors_per_token <= 1024, "head dimension is too large");

  c10::cuda::CUDAGuard guard(destination_k.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(batch, heads, selected_count);
  gather_kv_16byte_kernel<<<grid, vectors_per_token, 0, stream>>>(
      reinterpret_cast<const uint4*>(source_k.data_ptr()),
      reinterpret_cast<const uint4*>(source_v.data_ptr()),
      positions.data_ptr<int64_t>(),
      reinterpret_cast<uint4*>(destination_k.data_ptr()),
      reinterpret_cast<uint4*>(destination_v.data_ptr()),
      heads,
      source_length,
      destination_length,
      selected_count,
      vectors_per_token);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "UVA KV gather failed: ",
              cudaGetErrorString(error));
}

void append_kv(
    const torch::Tensor& source_k,
    const torch::Tensor& source_v,
    torch::Tensor& destination_k,
    torch::Tensor& destination_v,
    torch::Tensor& staging_k,
    int64_t destination_start,
    int64_t staging_start) {
  TORCH_CHECK(source_k.is_cuda() && source_v.is_cuda(),
              "append sources must be CUDA tensors");
  TORCH_CHECK(destination_k.device().is_cpu() && destination_v.device().is_cpu(),
              "append destinations must be CPU tensors");
  TORCH_CHECK(destination_k.is_pinned() && destination_v.is_pinned(),
              "append destinations must be pinned");
  TORCH_CHECK(staging_k.is_cuda(), "append staging tensor must be CUDA");
  TORCH_CHECK(source_k.is_contiguous() && source_v.is_contiguous() &&
                  destination_k.is_contiguous() && destination_v.is_contiguous() &&
                  staging_k.is_contiguous(),
              "append tensors must be contiguous");
  TORCH_CHECK(source_k.scalar_type() == source_v.scalar_type() &&
                  source_k.scalar_type() == destination_k.scalar_type() &&
                  source_k.scalar_type() == destination_v.scalar_type() &&
                  source_k.scalar_type() == staging_k.scalar_type(),
              "append dtypes must match");
  TORCH_CHECK(source_k.element_size() == 2,
              "append supports FP16/BF16 (two-byte elements)");
  TORCH_CHECK(source_k.dim() == 4 && source_k.sizes() == source_v.sizes(),
              "append sources must have matching [B,H,Q,D] shapes");

  const int batch = source_k.size(0);
  const int heads = source_k.size(1);
  const int incoming = source_k.size(2);
  const int head_dim = source_k.size(3);
  const int destination_length = destination_k.size(2);
  const int staging_length = staging_k.size(2);
  TORCH_CHECK(destination_start >= 0 &&
                  destination_start + incoming <= destination_length,
              "append destination range is out of bounds");
  TORCH_CHECK(staging_start >= 0 && staging_start + incoming <= staging_length,
              "append staging range is out of bounds");
  TORCH_CHECK((head_dim * 2) % 16 == 0,
              "head dimension byte width must be divisible by 16");
  const int vectors_per_token = head_dim * 2 / 16;

  c10::cuda::CUDAGuard guard(source_k.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(batch, heads, incoming);
  append_kv_16byte_kernel<<<grid, vectors_per_token, 0, stream>>>(
      reinterpret_cast<const uint4*>(source_k.data_ptr()),
      reinterpret_cast<const uint4*>(source_v.data_ptr()),
      reinterpret_cast<uint4*>(destination_k.data_ptr()),
      reinterpret_cast<uint4*>(destination_v.data_ptr()),
      reinterpret_cast<uint4*>(staging_k.data_ptr()),
      heads,
      incoming,
      destination_length,
      staging_length,
      static_cast<int>(destination_start),
      static_cast<int>(staging_start),
      vectors_per_token);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "UVA KV append failed: ",
              cudaGetErrorString(error));
}

void gather_blocks_reuse_kv(
    const torch::Tensor& source_k,
    const torch::Tensor& source_v,
    const torch::Tensor& previous_k,
    const torch::Tensor& previous_v,
    const torch::Tensor& previous_blocks,
    const torch::Tensor& current_blocks,
    torch::Tensor& destination_k,
    torch::Tensor& destination_v,
    torch::Tensor& destination_blocks,
    int64_t block_size,
    int64_t range0_start,
    int64_t range0_count,
    int64_t range1_start,
    int64_t range1_count) {
  TORCH_CHECK(source_k.device().is_cpu() && source_v.device().is_cpu(),
              "reuse sources must be CPU tensors");
  TORCH_CHECK(source_k.is_pinned() && source_v.is_pinned(),
              "reuse sources must be pinned");
  TORCH_CHECK(previous_k.is_cuda() && previous_v.is_cuda() &&
                  destination_k.is_cuda() && destination_v.is_cuda(),
              "reuse buffers must be CUDA tensors");
  TORCH_CHECK(previous_blocks.is_cuda() && current_blocks.is_cuda() &&
                  destination_blocks.is_cuda(),
              "reuse block ids must be CUDA tensors");
  TORCH_CHECK(previous_blocks.scalar_type() == torch::kInt64 &&
                  current_blocks.scalar_type() == torch::kInt64 &&
                  destination_blocks.scalar_type() == torch::kInt64,
              "reuse block ids must be int64");
  TORCH_CHECK(source_k.is_contiguous() && source_v.is_contiguous() &&
                  previous_k.is_contiguous() && previous_v.is_contiguous() &&
                  current_blocks.is_contiguous() &&
                  previous_blocks.is_contiguous() &&
                  destination_k.is_contiguous() && destination_v.is_contiguous() &&
                  destination_blocks.is_contiguous(),
              "reuse tensors must be contiguous");
  TORCH_CHECK(source_k.scalar_type() == source_v.scalar_type() &&
                  source_k.scalar_type() == previous_k.scalar_type() &&
                  source_k.scalar_type() == previous_v.scalar_type() &&
                  source_k.scalar_type() == destination_k.scalar_type() &&
                  source_k.scalar_type() == destination_v.scalar_type(),
              "reuse dtypes must match");
  TORCH_CHECK(source_k.element_size() == 2,
              "reuse supports FP16/BF16 (two-byte elements)");
  TORCH_CHECK(source_k.dim() == 4 && source_k.sizes() == source_v.sizes(),
              "reuse sources must have matching [B,H,N,D] shapes");
  TORCH_CHECK(current_blocks.dim() == 3 &&
                  current_blocks.sizes() == previous_blocks.sizes() &&
                  current_blocks.sizes() == destination_blocks.sizes(),
              "reuse block ids must have matching [B,H,K] shapes");
  const int batch = source_k.size(0);
  const int heads = source_k.size(1);
  const int source_length = source_k.size(2);
  const int head_dim = source_k.size(3);
  const int selected_blocks = current_blocks.size(2);
  const int destination_length = destination_k.size(2);
  TORCH_CHECK(previous_k.sizes() == destination_k.sizes() &&
                  previous_v.sizes() == destination_v.sizes() &&
                  previous_k.sizes() == previous_v.sizes() &&
                  previous_k.sizes() == destination_v.sizes(),
              "reuse K/V buffers must have matching shapes");
  TORCH_CHECK(block_size > 0 && (head_dim * 2) % 16 == 0,
              "invalid reuse block/head dimensions");
  const int vectors_per_token = head_dim * 2 / 16;
  const int copy_threads = static_cast<int>(block_size) * vectors_per_token;
  TORCH_CHECK(copy_threads <= 1024, "reuse block copy exceeds CUDA block limit");
  // The id scan is strided over the whole thread block, so a narrow copy width
  // would leave it nearly serial.  Widen the launch; the surplus threads exit
  // before the copy.
  const int threads = copy_threads < 256 ? 256 : copy_threads;
  TORCH_CHECK(range0_start >= 0 && range0_count >= 0 &&
                  range1_start >= 0 && range1_count >= 0 &&
                  range0_start + range0_count <= source_length &&
                  range1_start + range1_count <= source_length,
              "reuse exact range is out of bounds");
  TORCH_CHECK(selected_blocks * block_size + range0_count + range1_count <=
                  destination_length,
              "reuse destination capacity is too small");

  c10::cuda::CUDAGuard guard(destination_k.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(
      batch, heads, selected_blocks + range0_count + range1_count);
  gather_blocks_reuse_16byte_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(source_k.data_ptr()),
      reinterpret_cast<const uint4*>(source_v.data_ptr()),
      reinterpret_cast<const uint4*>(previous_k.data_ptr()),
      reinterpret_cast<const uint4*>(previous_v.data_ptr()),
      previous_blocks.data_ptr<int64_t>(),
      current_blocks.data_ptr<int64_t>(),
      reinterpret_cast<uint4*>(destination_k.data_ptr()),
      reinterpret_cast<uint4*>(destination_v.data_ptr()),
      destination_blocks.data_ptr<int64_t>(),
      heads,
      source_length,
      destination_length,
      selected_blocks,
      static_cast<int>(block_size),
      vectors_per_token,
      static_cast<int>(range0_start),
      static_cast<int>(range0_count),
      static_cast<int>(range1_start),
      static_cast<int>(range1_count),
      true);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "UVA reuse gather failed: ",
              cudaGetErrorString(error));
}

// ShadowKV reconstructs keys from a low-rank factorisation on the GPU and
// never moves them across PCIe.  Gathering K alongside V would double its
// traffic for bytes it throws away, so expose a values-only entry point that
// runs the same block-reuse kernel with the key side switched off.
void gather_blocks_reuse_values(
    const torch::Tensor& source_v,
    const torch::Tensor& previous_v,
    const torch::Tensor& previous_blocks,
    const torch::Tensor& current_blocks,
    torch::Tensor& destination_v,
    torch::Tensor& destination_blocks,
    int64_t block_size) {
  TORCH_CHECK(source_v.device().is_cpu(), "reuse source must be a CPU tensor");
  TORCH_CHECK(source_v.is_pinned(), "reuse source must be pinned");
  TORCH_CHECK(previous_v.is_cuda() && destination_v.is_cuda(),
              "reuse buffers must be CUDA tensors");
  TORCH_CHECK(previous_blocks.is_cuda() && current_blocks.is_cuda() &&
                  destination_blocks.is_cuda(),
              "reuse block ids must be CUDA tensors");
  TORCH_CHECK(previous_blocks.scalar_type() == torch::kInt64 &&
                  current_blocks.scalar_type() == torch::kInt64 &&
                  destination_blocks.scalar_type() == torch::kInt64,
              "reuse block ids must be int64");
  TORCH_CHECK(source_v.is_contiguous() && previous_v.is_contiguous() &&
                  destination_v.is_contiguous() &&
                  previous_blocks.is_contiguous() &&
                  current_blocks.is_contiguous() &&
                  destination_blocks.is_contiguous(),
              "reuse tensors must be contiguous");
  TORCH_CHECK(source_v.element_size() == 2,
              "kernel supports FP16/BF16 (two-byte elements)");
  TORCH_CHECK(source_v.dim() == 4, "reuse source must be [B,H,N,D]");
  TORCH_CHECK(current_blocks.dim() == 3 &&
                  current_blocks.sizes() == previous_blocks.sizes() &&
                  current_blocks.sizes() == destination_blocks.sizes(),
              "reuse block ids must have matching [B,H,K] shapes");
  TORCH_CHECK(previous_v.sizes() == destination_v.sizes(),
              "reuse V buffers must have matching shapes");
  const int batch = source_v.size(0);
  const int heads = source_v.size(1);
  const int source_length = source_v.size(2);
  const int head_dim = source_v.size(3);
  const int selected_blocks = current_blocks.size(2);
  const int destination_length = destination_v.size(2);
  TORCH_CHECK(block_size > 0 && (head_dim * 2) % 16 == 0,
              "invalid reuse block/head dimensions");
  const int vectors_per_token = head_dim * 2 / 16;
  const int copy_threads = static_cast<int>(block_size) * vectors_per_token;
  TORCH_CHECK(copy_threads <= 1024, "reuse block copy exceeds CUDA block limit");
  const int threads = copy_threads < 256 ? 256 : copy_threads;
  TORCH_CHECK(selected_blocks * block_size <= destination_length,
              "reuse destination capacity is too small");

  c10::cuda::CUDAGuard guard(destination_v.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(batch, heads, selected_blocks);
  gather_blocks_reuse_16byte_kernel<<<grid, threads, 0, stream>>>(
      nullptr,
      reinterpret_cast<const uint4*>(source_v.data_ptr()),
      nullptr,
      reinterpret_cast<const uint4*>(previous_v.data_ptr()),
      previous_blocks.data_ptr<int64_t>(),
      current_blocks.data_ptr<int64_t>(),
      nullptr,
      reinterpret_cast<uint4*>(destination_v.data_ptr()),
      destination_blocks.data_ptr<int64_t>(),
      heads,
      source_length,
      destination_length,
      selected_blocks,
      static_cast<int>(block_size),
      vectors_per_token,
      0,
      0,
      0,
      0,
      false);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "UVA reuse value gather failed: ",
              cudaGetErrorString(error));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gather_kv", &gather_kv, "Pinned-CPU fused K/V gather");
  module.def("append_kv", &append_kv, "Pinned-CPU fused K/V append");
  module.def("gather_blocks_reuse_kv", &gather_blocks_reuse_kv,
             "Pinned-CPU block gather with GPU reuse");
  module.def("gather_blocks_reuse_values", &gather_blocks_reuse_values,
             "Pinned-CPU block gather with GPU reuse, values only");
}
