"""Indexed modulation, residual gating and rotary for H3 joint blocks."""

from ._pointwise import _gate_index32 as _gate, _mod_index32 as _mod, _rope

KERNELS = {'pointwise': ('_mod', '_gate', '_rope')}
