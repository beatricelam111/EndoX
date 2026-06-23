"""Warp / omni.replicator.core compatibility shim.

omni.replicator.core 1.11.x constructs warp arrays with the ``owner``
keyword (``wp.types.array(..., owner=...)``), which was removed in
Warp >= 1.11.  This module patches the constructor once so the old
replicator code works transparently with newer Warp builds.

Import this module *after* ``import warp as wp; wp.init()`` in any
script that calls ``annotator.get_data()``.
"""

import warp as wp

_PATCHED = False


def patch_warp_array_init():
    """Strip the removed ``owner`` kwarg from ``wp.types.array.__init__``.

    Safe to call multiple times; only applies the patch once and only
    when the running Warp version no longer accepts ``owner``.
    """
    global _PATCHED
    if _PATCHED:
        return

    import inspect

    try:
        sig = inspect.signature(wp.types.array.__init__)
    except (ValueError, TypeError):
        _PATCHED = True
        return

    if "owner" in sig.parameters:
        _PATCHED = True
        return

    _original = wp.types.array.__init__

    def _patched(self, *args, **kwargs):
        kwargs.pop("owner", None)
        return _original(self, *args, **kwargs)

    wp.types.array.__init__ = _patched
    _PATCHED = True


patch_warp_array_init()
