"""
Model runner for efficient inference execution.

Design Philosophy:
- CUDA graphs for zero-overhead decode phase
- KV cache allocation based on available GPU memory
- Tensor parallelism via multiprocessing and shared memory
- Separate preparation logic for prefill vs decode

Optimization Notes (from benchmark report):
- CUDA graphs provide 2-3x speedup by eliminating kernel launch overhead
- Captured for batch sizes [1, 2, 4, 8, 16, 32, ..., 512]
- Only used for decode phase (prefill has variable shapes)
- Flash Attention provides additional 1.5-2x speedup

Architecture:
- Rank 0: Main process that coordinates and samples tokens
- Rank 1-N: Worker processes for tensor parallelism (if tp_size > 1)
- Communication via shared memory (IPC) for low latency
"""
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import List

from config import Config
from engine.types import ModelRunner as ModelRunnerABC
from engine.sequence import Sequence
from models.qwen3 import Qwen3ForCausalLM
from layers.sampler import Sampler
from utils.context import set_context, get_context, reset_context
from utils.loader import load_model


class ModelRunner(ModelRunnerABC):
    """
    Model runner with CUDA graph support and tensor parallelism.

    Responsibilities:
    1. Load model and allocate KV cache
    2. Capture CUDA graphs for efficient decode
    3. Prepare input tensors for prefill/decode phases
    4. Execute forward passes
    5. Sample next tokens
    6. Coordinate across multiple GPUs (if tensor parallel)
    """

    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | List[Event]
    ):
        """
        Initialize model runner.

        Args:
            config: System configuration
            rank: GPU rank (0 for main, 1+ for workers)
            event: Event for synchronization (list if rank 0, single if rank > 0)
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager or not config.enable_cuda_graphs
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Initialize distributed backend
        if self.world_size > 1:
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:2333",
                world_size=self.world_size,
                rank=rank
            )

        # Set device and default dtype
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        # Load model
        print(f"[Rank {rank}] Loading model...")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.model.eval()

        # Create sampler (only rank 0 samples)
        self.sampler = Sampler()

        # Setup pipeline
        print(f"[Rank {rank}] Warming up model...")
        self.warmup_model()

        print(f"[Rank {rank}] Allocating KV cache...")
        self.allocate_kv_cache()

        if not self.enforce_eager:
            print(f"[Rank {rank}] Capturing CUDA graphs...")
            self.capture_cudagraph()

        # Reset defaults
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # Setup IPC for tensor parallelism
        if self.world_size > 1:
            if rank == 0:
                # Rank 0 creates shared memory
                self.shm = SharedMemory(name="allmos_v2", create=True, size=2**20)
                dist.barrier()
                print(f"[Rank 0] Shared memory created, workers ready")
            else:
                # Workers attach to shared memory
                dist.barrier()
                self.shm = SharedMemory(name="allmos_v2")
                print(f"[Rank {rank}] Attached to shared memory, entering event loop")
                self.loop()  # Worker loop (never returns)

    def exit(self) -> None:
        """Cleanup resources before shutdown."""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()

        if not self.enforce_eager:
            del self.graphs, self.graph_pool

        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    def loop(self) -> None:
        """
        Worker process event loop.

        Workers wait for commands from rank 0 via shared memory.
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self) -> tuple:
        """Read command from shared memory (worker side)."""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name: str, *args) -> None:
        """Write command to shared memory (coordinator side)."""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name: str, *args):
        """
        Call a method on this runner.

        If tensor parallel, broadcasts call to all workers.
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self) -> None:
        """
        Warm up model to measure memory usage.

        Runs a large batch through the model to trigger all allocations,
        then measures peak memory to compute KV cache budget.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(
            max_num_batched_tokens // max_model_len,
            self.config.max_num_seqs
        )

        # Create dummy sequences at max length
        seqs = [
            Sequence([0] * max_model_len)
            for _ in range(num_seqs)
        ]

        # Run prefill to trigger all allocations
        self.run(seqs, is_prefill=True)

        torch.cuda.empty_cache()

    def allocate_kv_cache(self) -> None:
        """
        Allocate KV cache based on available GPU memory.

        Strategy:
        1. Measure current memory usage
        2. Reserve gpu_memory_utilization fraction of total memory
        3. Subtract model memory and activations
        4. Allocate remaining space for KV cache blocks
        """
        config = self.config
        hf_config = config.hf_config

        # Get memory stats
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # Compute number of blocks that fit
        config.num_kvcache_blocks = config.compute_num_kvcache_blocks(
            total, used, peak, current
        )

        print(f"[Rank {self.rank}] Allocating {config.num_kvcache_blocks} KV cache blocks")

        # Allocate KV cache tensor
        # Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        self.kv_cache = torch.empty(
            2,  # K and V
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=hf_config.torch_dtype,
            device="cuda"
        )

        # Assign cache slices to attention layers
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: List[Sequence]) -> torch.Tensor:
        """
        Prepare block tables for batch of sequences.

        Block tables map sequence positions to KV cache blocks.
        Padded to same length with -1 for unused entries.

        Args:
            seqs: List of sequences

        Returns:
            Block tables tensor [batch_size, max_num_blocks]
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables,
            dtype=torch.int32,
            pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: List[Sequence]) -> tuple:
        """
        Prepare input tensors for prefill phase.

        Prefill processes all prompt tokens at once (variable length per sequence).
        Uses varlen (variable length) Flash Attention for efficiency.

        Args:
            seqs: Sequences to process

        Returns:
            Tuple of (input_ids, positions)
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # Cumulative sequence lengths for queries
        cu_seqlens_k = [0]  # Cumulative sequence lengths for keys
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)

            # Only process uncached tokens
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            seqlen_q = seqlen - seq.num_cached_tokens  # New tokens
            seqlen_k = seqlen  # All tokens (including cached)

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # Skip slot mapping for warmup (empty block table)
            if not seq.block_table:
                continue

            # Map tokens to KV cache slots
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    # Full block
                    end = start + self.block_size
                else:
                    # Last block (may be partial)
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        # Check if we have prefix caching (more keys than queries)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Set attention context
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables
        )

        return input_ids, positions

    def prepare_decode(self, seqs: List[Sequence]) -> tuple:
        """
        Prepare input tensors for decode phase.

        Decode processes one token per sequence (same length for all sequences).
        Can use CUDA graphs since batch size and shapes are fixed.

        Args:
            seqs: Sequences to process

        Returns:
            Tuple of (input_ids, positions)
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))

            # Map to KV cache slot
            slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            slot_mapping.append(slot)

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # Set attention context
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables
        )

        return input_ids, positions

    def prepare_sample(self, seqs: List[Sequence]) -> torch.Tensor:
        """
        Prepare sampling parameters.

        Args:
            seqs: Sequences to sample for

        Returns:
            Temperature tensor [batch_size]
        """
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(
            temperatures,
            dtype=torch.float32,
            pin_memory=True
        ).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool
    ) -> torch.Tensor:
        """
        Run model forward pass.

        Uses CUDA graphs for decode phase if available.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            is_prefill: Whether this is prefill or decode

        Returns:
            Logits tensor
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Eager mode (no CUDA graph)
            hidden_states = self.model(input_ids, positions)
            return self.model.compute_logits(hidden_states)
        else:
            # CUDA graph mode
            bs = input_ids.size(0)
            context = get_context()

            # Find appropriate graph (round up to next graph size)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # Copy inputs into graph memory
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # Replay graph
            graph.replay()

            # Extract outputs
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        """
        Run model on batch of sequences.

        Args:
            seqs: Sequences to process
            is_prefill: Whether this is prefill or decode phase

        Returns:
            List of sampled token IDs (None for worker ranks)
        """
        # Prepare inputs
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        # Prepare sampling parameters (rank 0 only)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # Run model
        logits = self.run_model(input_ids, positions, is_prefill)

        # Sample tokens (rank 0 only)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # Reset context
        reset_context()

        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        """
        Capture CUDA graphs for decode phase.

        Captures graphs for batch sizes [1, 2, 4, 8, 16, ..., max_bs].
        Graphs eliminate kernel launch overhead (2-3x speedup).
        """
        config = self.config
        hf_config = config.hf_config

        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # Allocate graph memory (shared across all graphs via pool)
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        positions = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device="cuda")
        outputs = torch.zeros(max_bs, hf_config.hidden_size, dtype=hf_config.torch_dtype, device="cuda")

        # Batch sizes to capture
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # Capture graphs in reverse order (largest first for memory allocation)
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            # Set context for this batch size
            set_context(
                is_prefill=False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs]
            )

            # Warmup
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # Capture
            with torch.cuda.graph(graph, pool=self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # Create pool from first graph
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # Save references to graph variables
        self.graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
        }

        print(f"[Rank {self.rank}] Captured {len(self.graphs)} CUDA graphs")
