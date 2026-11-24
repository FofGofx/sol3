"""Connect workspace release to a two-stage LTX pipeline."""

from sol3_integrations import ltx25


def generate(entry, runner, *args, release_points=(), fusion_modules=(), **kwargs):
    """Use the entry module's original Stage 1 release before Stage 2 proceeds."""
    with ltx25.activate(release_points=release_points, fusion_modules=fusion_modules) as installed:
        installed.release_after(entry, "release_workspace")
        return runner.generate(*args, **kwargs)
