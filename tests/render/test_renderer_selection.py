"""Public renderer selection and preserved NuRec asset resolution."""
import argparse
import ast
from pathlib import Path

import pytest
from navsafe.render import create_renderer, NuRecGrpcSceneRenderer
from navsafe.render.usdz_utils import resolve_usdz_path


def test_default_renderer():
    assert isinstance(create_renderer(), NuRecGrpcSceneRenderer)


@pytest.mark.parametrize('name', ['assets', 'rasterized', '3dgs', 'nurec_usdz', 'worldmodel'])
def test_removed_backends_rejected(name):
    with pytest.raises(ValueError, match='nurec_grpc'):
        create_renderer(name)


def test_cli_default_and_choices():
    # Exercise the actual parser declaration without starting IsaacSim.
    source = Path(__file__).parents[2] / 'navsafe/cli/eval_entry.py'
    tree = ast.parse(source.read_text())
    declaration = next(n for n in tree.body if isinstance(n, ast.Expr)
                       and isinstance(n.value, ast.Call) and n.value.args
                       and isinstance(n.value.args[0], ast.Constant)
                       and n.value.args[0].value == '--render-backend')
    parser = argparse.ArgumentParser()
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(source), 'exec'), {'ap': parser})
    assert parser.parse_args([]).render_backend == 'nurec_grpc'
    with pytest.raises(SystemExit):
        parser.parse_args(['--render-backend', 'assets'])


def test_usdz_resolution(tmp_path):
    asset = tmp_path / 'usd-out/last.usdz'
    asset.parent.mkdir()
    asset.touch()
    assert resolve_usdz_path({'nurec_run_dir': str(tmp_path)}) == str(asset)
    assert resolve_usdz_path({'nurec_usdz_path': str(asset)}) == str(asset)
    assert resolve_usdz_path({}) is None
