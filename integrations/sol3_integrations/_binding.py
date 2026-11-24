"""Connect the installed forward at the model's existing, already selected sparse call."""

import ast
from contextlib import ExitStack, contextmanager
from functools import wraps
from importlib import import_module
from inspect import getattr_static, isdatadescriptor
from collections.abc import Mapping
import sys
from textwrap import dedent
from threading import RLock
from types import ModuleType

from sol3_attn import release_workspace, sol3_attn_with_route


_installation = RLock()


def _bound_model(bound, binding=None):
    if getattr_static(bound, "__sol3_sparse_model__", None) is not None:
        return getattr(bound, "__sol3_sparse_model__", None)
    # A host can declare its model marker through the binding module’s REVISION.
    if binding is None:
        binding = sys.modules.get(type(bound).__module__)
        if not isinstance(bound, getattr(binding, "BoundSparse", ())):
            binding = sys.modules.get("sol3_sparse_binding")
    revision = getattr(binding, "REVISION", None)
    if isinstance(revision, str) and revision and isinstance(bound, getattr(binding, "BoundSparse", ())):
        return getattr(bound, f"__sol3_{revision}__", None)
    return None


def _kernel_body(kernel):
    tree = ast.parse(dedent(kernel.src))
    fn = tree.body[0]
    fn.name = "kernel"
    fn.decorator_list = []
    return ast.dump(fn)


class _CountedKernel:
    def __init__(self, kernel, counts, key):
        self.kernel = kernel
        self.counts = counts
        self.key = key

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def measured(*args, **kwargs):
            result = launch(*args, **kwargs)
            self.counts[self.key] = self.counts.get(self.key, 0) + 1
            return result

        return measured


def _replace(stack, obj, key, replacement):
    local = getattr(obj, "__dict__", {})
    restore_value = key in local or isdatadescriptor(getattr_static(type(obj), key, None))
    original = local[key] if key in local else getattr(obj, key)
    setattr(obj, key, replacement)
    if restore_value:
        stack.callback(setattr, obj, key, original)
    else:
        stack.callback(delattr, obj, key)


class Integration:
    def __init__(self, model, fusions, counters, stack):
        self.model = model
        self.fusions = fusions
        self.counters = counters
        self.stack = stack
        self.attached = set()
        self.calls = {}
        self.metadata = {}

    def release(self):
        for native, route in self.metadata.values():
            if native.get("route") is route:
                native.clear()
        self.metadata.clear()
        entries = release_workspace(self.model)
        if entries and self.counters is not None:
            self.counters["workspace_entries_released"] = self.counters.get("workspace_entries_released", 0) + entries

    def release_after(self, obj, name):
        fn = getattr(obj, name, None)
        marker = (id(obj), name)
        descriptor = getattr_static(obj, name, None) if isinstance(obj, type) else None
        if isinstance(descriptor, (classmethod, staticmethod)):
            fn = descriptor.__func__
        if not callable(fn):
            raise TypeError(f"Release point {name!r} must be callable")
        if marker in self.attached:
            return

        @wraps(fn)
        def hooked(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                self.release()
                if name == "warmup" and self.counters is not None:
                    self.counters["warmup_releases"] = self.counters.get("warmup_releases", 0) + 1

        if isinstance(descriptor, (classmethod, staticmethod)):
            hooked = type(descriptor)(hooked)
        _replace(self.stack, obj, name, hooked)
        self.attached.add(marker)

    def attach_fusions(self, role, module):
        """Attach matching JIT kernels before the host first launches them."""
        if not isinstance(module, ModuleType):
            raise TypeError("Pass the host module containing the kernel launch sites")
        for key in self.fusions.KERNELS[role]:
            marker = (id(module), key)
            if marker in self.attached:
                continue
            original = vars(module).get(key)
            target = getattr(self.fusions, key)
            if not hasattr(original, "src") or _kernel_body(original) != _kernel_body(target):
                continue
            if self.counters is not None:
                target = _CountedKernel(target, self.counters, "fusion." + key)
            _replace(self.stack, module, key, target)
            self.attached.add(marker)

    def bind(self, owner, *, metadata=None):
        """Bind a native cfg-based callable or an LTX/H3 kernel instance."""
        if _bound_model(owner) == self.model:
            owner = owner.owner
        key = (id(owner), id(metadata))
        if key in self.calls:
            return self.calls[key]
        core = getattr(owner, "core", None)
        inner = getattr(owner, "inner", None)
        if core is None and inner is not None:
            core = sys.modules.get(inner.forward.__module__)
        if core is None:
            core = sys.modules.get(getattr(owner, "__module__", ""))
        last = metadata if metadata is not None else getattr(core, "LAST", None)
        if last is None and inner is not None:
            last = getattr(inner.forward, "__globals__", {}).get("LAST")
        cfg = inner.config if inner is not None else getattr(owner, "config", None)
        counts = self.counters

        def call(q, k, v, **kwargs):
            if kwargs.get("tau", 1.0) <= -100:
                return owner(q, k, v, **kwargs)
            if counts is not None:
                counts["sparse_calls"] = counts.get("sparse_calls", 0) + 1
            if self.model == "fast":
                return sol3_attn_with_route(q, k, v, profile="fast", config=kwargs.pop("config", None),
                                            counters=counts, **kwargs)
            config = kwargs.pop("cfg", cfg)
            if config is None:
                raise TypeError("The model adapter requires the original Sol3 configuration")
            output, route = sol3_attn_with_route(q, k, v, profile=self.model, config=config, metadata=last,
                                                 counters=counts, **kwargs)
            if last is not None:
                self.metadata[id(last)] = (last, route)
            return output

        self.calls[key] = call
        return call


@contextmanager
def integration(model, module, release_points, *, counters=None, fusion_modules=()):
    with _installation, ExitStack() as stack:
        scope = Integration(model, import_module(module), counters, stack)
        stack.callback(scope.release)
        for owner, name in release_points:
            scope.release_after(owner, name)
        if isinstance(fusion_modules, Mapping):
            for role, host_module in fusion_modules.items():
                scope.attach_fusions(role, host_module)
        else:
            # Existing callers can pass an explicit sequence of host modules.
            for host_module in fusion_modules:
                for role in scope.fusions.KERNELS:
                    scope.attach_fusions(role, host_module)
        yield scope


@contextmanager
def activate(model, module, release_points, *, counters=None, fusion_modules=(), binding=None,
             enabled=None, target=None):
    """Use Sol³ for matching sparse calls and restore the original hooks on exit."""
    with integration(model, module, release_points, counters=counters, fusion_modules=fusion_modules) as scope:
        if target is not None:
            if binding is not None:
                raise ValueError("Pass either a sparse-call target or a host binding")
            owner, name = target
            if isinstance(owner, type):
                raise TypeError("Use a module or instance holding the selected sparse callable")
            original = getattr(owner, name)
            if not callable(original):
                raise TypeError(f"Target {name!r} must be callable")
            _replace(scope.stack, owner, name, scope.bind(original))
            yield scope
            return
        native_binding = import_module("sol3_sparse_binding") if binding is None else binding
        cls = native_binding.BoundSparse
        previous = cls.__call__
        bound_calls = {}

        def dispatch(bound, q, k, v, *args, **kwargs):
            if _bound_model(bound, native_binding) != model or enabled is not None and not enabled():
                return previous(bound, q, k, v, *args, **kwargs)
            if bound not in bound_calls:
                bound_calls[bound] = scope.bind(bound.owner)
            return bound_calls[bound](q, k, v, *args, **kwargs)

        _replace(scope.stack, cls, "__call__", dispatch)
        yield scope
