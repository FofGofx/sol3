"""Protect H3's position-wise FFN while retaining reusable attention workspace."""

from contextlib import ExitStack, contextmanager
from functools import partial, wraps
from time import perf_counter

import torch


def _budget(device, driver_free):
    stats = torch.cuda.memory_stats(device)
    reserved = stats.get("reserved_bytes.all.current", 0)
    active = stats.get("active_bytes.all.current", reserved)
    # Split fragments and blocks pending completion on another stream cannot
    # be assumed to satisfy the next large projection.
    split = stats.get("inactive_split_bytes.all.current", 0)
    reusable = max(0, reserved - active - split)
    return driver_free + reusable, reusable


def _chunk_forward(forward, x, tokens, args, kwargs, out_features):
    result = x.new_empty((*x.shape[:-1], out_features))
    for start in range(0, x.shape[-2], tokens):
        dest = result.narrow(-2, start, min(tokens, x.shape[-2] - start))
        piece = forward(x.narrow(-2, start, dest.shape[-2]), *args, **kwargs)
        dest.copy_(piece)
        del piece
    return result


class MemoryPolicy:
    def __init__(self, active_job, release, reserve_bytes):
        self.active_job = active_job
        self.release = release
        self.reserve_bytes = int(reserve_bytes)
        self.reset()

    def reset(self):
        self.calls = self.driver_low = self.cache_sufficient = self.chunks = 0
        self.ooms = self.releases = self.freed = 0
        self.release_seconds = 0.0
        self.pressure_shapes = set()
        self.oom_tiles = {}
        self.first_cache_hit = None
        self.notes = []

    def _release(self, job, x, reason, free, available, required):
        allocated = torch.cuda.memory_allocated(x.device)
        started = perf_counter()
        note = self.release(job)
        with torch.cuda.device(x.device):
            torch.cuda.empty_cache()
        elapsed = perf_counter() - started
        freed = max(0, allocated - torch.cuda.memory_allocated(x.device))
        self.releases += 1
        self.freed += freed
        self.release_seconds += elapsed
        if len(self.notes) < 8:
            self.notes.append(dict(reason=reason, shape=list(x.shape), free_before=free,
                                   available_before=available, required=required,
                                   attention_scratch_bytes_released=freed, seconds=elapsed,
                                   release=note))

    def _tile(self, x, layer, available, required):
        length = x.shape[-2]
        out_features = layer.net[-1].out_features
        output = x.numel() // x.shape[-1] * out_features * x.element_size()
        # The full output and both the projection and its activated half can
        # coexist. Include the chunk's own output before copying into it.
        per_token = (required - self.reserve_bytes + output) // length
        cap = (available - self.reserve_bytes - output) // max(1, per_token)
        if cap < 1:
            raise torch.cuda.OutOfMemoryError(
                f"H3 FFN needs {self.reserve_bytes + output + per_token} bytes for its reserve, "
                f"output and one token; only {available} bytes are available")
        cap = max(1, min(cap, max(1, length // 2)))
        return max(1, cap // 128 * 128) if cap >= 128 else cap

    def run(self, layer, forward, hidden_states, *args, **kwargs):
        x = hidden_states
        job = self.active_job()
        if (job is None or job.attention_mode != "sol_attn^3" or torch.is_grad_enabled()
                or layer.training or not x.is_cuda or x.ndim < 2 or x.numel() == 0):
            return forward(x, *args, **kwargs)
        self.calls += 1
        rows = x.numel() // x.shape[-1]
        packed = rows * layer.net[0].proj.out_features * x.element_size()
        required = packed + packed // 2 + self.reserve_bytes
        free, _ = torch.cuda.mem_get_info(x.device)
        available = free
        key = (x.device, tuple(x.shape), layer.net[0].proj.out_features, x.dtype)
        if free < required:
            self.driver_low += 1
            available, reusable = _budget(x.device, free)
            if available >= required:
                self.cache_sufficient += 1
                if self.first_cache_hit is None:
                    self.first_cache_hit = dict(shape=list(x.shape), driver_free=free, reusable=reusable,
                                                required=required, effective_available=available)
        released = False
        if available >= required + self.reserve_bytes:
            self.pressure_shapes.discard(key)
        elif available < required and key not in self.pressure_shapes:
            self.pressure_shapes.add(key)
            self._release(job, x, "insufficient_headroom", free, available, required)
            released = True
            free, _ = torch.cuda.mem_get_info(x.device)
            available, _ = _budget(x.device, free)
        tile = None
        previous_tile = self.oom_tiles.get(key)
        try:
            if available >= required and previous_tile is None:
                return forward(x, *args, **kwargs)
            tile = self._tile(x, layer, available, required)
            if previous_tile is not None:
                tile = min(tile, previous_tile)
            self.chunks += 1
            return _chunk_forward(forward, x, tile, args, kwargs, layer.net[-1].out_features)
        except torch.cuda.OutOfMemoryError:
            # Exit the exception handler before retrying so failed projection
            # tensors held by its traceback can be released.
            self.ooms += 1
        if not released:
            self._release(job, x, "cuda_oom", free, available, required)
        self.pressure_shapes.add(key)
        free, _ = torch.cuda.mem_get_info(x.device)
        available, _ = _budget(x.device, free)
        retry = self._tile(x, layer, available, required)
        if tile is not None:
            retry = min(retry, max(1, tile // 2))
        # A failed full allocation disproves the aggregate estimate for this
        # geometry. Keep the smaller block through the rest of this job.
        self.oom_tiles[key] = retry
        self.chunks += 1
        return _chunk_forward(forward, x, retry, args, kwargs, layer.net[-1].out_features)

    def snapshot(self):
        return dict(checked_FFN_calls=self.calls,
                    driver_low_calls=self.driver_low, allocator_sufficient_calls=self.cache_sufficient,
                    pressure_releases=self.releases, release_seconds=self.release_seconds,
                    attention_scratch_bytes_released=self.freed, chunked_FFN_calls=self.chunks,
                    cuda_oom_retries=self.ooms, reserve_bytes=self.reserve_bytes, samples=self.notes,
                    first_reusable_cache_hit=self.first_cache_hit)


def _restore(obj, name, previous, installed, had_local):
    if getattr(obj, name) is installed:
        if had_local:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def _install(stack, obj, name, installed):
    previous = getattr(obj, name)
    had_local = name in vars(obj)
    setattr(obj, name, installed)
    stack.callback(_restore, obj, name, previous, installed, had_local)


class AttentionScratchGuard(MemoryPolicy):
    """The native H3 hook owns one guard for the lifetime of its transformer."""

    def __init__(self, model, active_job, release, reserve_bytes=2 * 1024**3):
        from diffusers.models.activations import SwiGLU
        from diffusers.models.attention import FeedForward

        super().__init__(active_job, release, reserve_bytes)
        self.stack = ExitStack()
        for layer in model.modules():
            if type(layer) is FeedForward and type(layer.net[0]) is SwiGLU:
                _install(self.stack, layer, "forward", partial(self.run, layer, layer.forward))

    def close(self):
        self.stack.close()


@contextmanager
def existing_guard_scope():
    """Use the FFN memory policy with guards already attached to the transformer."""
    try:
        from diffusers.models.attention import FeedForward
    except ModuleNotFoundError as exc:
        if exc.name != "diffusers":
            raise
        yield
        return

    previous = FeedForward.forward
    layers = {}
    policies = {}
    with ExitStack() as stack:
        def discover(layer):
            # Replace the guard’s SwiGLU pre-hook while preserving the order of unrelated hooks.
            hooks = layer.net[0]._forward_pre_hooks
            for key, callback in hooks.items():
                owner = getattr(callback, "__self__", None)
                cls = type(owner)
                if (cls.__module__ != "h3_attention_scratch_guard" or cls.__name__ != "AttentionScratchGuard"
                        or getattr(callback, "__name__", None) != "before"):
                    continue
                if owner not in policies:
                    policy = MemoryPolicy(owner.active_job, owner.release, owner.reserve_bytes)
                    policies[owner] = policy
                    _install(stack, owner, "reset", policy.reset)
                    _install(stack, owner, "snapshot", policy.snapshot)
                def pass_through(module, args):
                    return None
                def restore_hook(hooks=hooks, key=key, callback=callback):
                    if hooks.get(key) is pass_through:
                        hooks[key] = callback
                stack.callback(restore_hook)
                hooks[key] = pass_through
                return policies[owner]
            return None

        @wraps(previous)
        def guarded(layer, hidden_states, *args, **kwargs):
            policy = layers.get(layer)
            if policy is None:
                policy = discover(layer)
                if policy is None:
                    return previous(layer, hidden_states, *args, **kwargs)
                layers[layer] = policy
            return policy.run(layer, partial(previous, layer), hidden_states, *args, **kwargs)

        _install(stack, FeedForward, "forward", guarded)
        yield
