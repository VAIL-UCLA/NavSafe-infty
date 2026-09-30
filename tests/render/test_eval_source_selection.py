"""Input selection contracts tested without launching IsaacSim."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).parents[2]


def parser():
    tree = ast.parse((ROOT / 'navsafe/cli/eval_entry.py').read_text())
    declarations = [n for n in tree.body if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute)
                    and isinstance(n.value.func.value, ast.Name)
                    and n.value.func.value.id == 'ap'
                    and n.value.func.attr == 'add_argument']
    ap = argparse.ArgumentParser()
    exec(compile(ast.Module(body=declarations, type_ignores=[]), '<eval-parser>', 'exec'), {'ap': ap, 'argparse': argparse, '_AUTOSELECT_DEFAULT_DIR': '/unused-recipe-default'})
    return ap


def test_default_arrow_input():
    args = parser().parse_args(['--checkpoint', 'none', '--output-dir', '/tmp/eval'])
    assert args.scenario_source == 'py123d'


@pytest.mark.parametrize('flags', [
    ['--scenario-source', 'sd_pickle'], ['--scenario-source', 'pg'],
    ['--scenario-pkl', 'scene.pkl'], ['--num-blocks', '5'], ['--seed', '1'],
])
def test_removed_source_flags_fail(flags):
    with pytest.raises(SystemExit):
        parser().parse_args(['--checkpoint', 'none', '--output-dir', '/tmp/eval', *flags])


def test_programmatic_builder_rejects_old_sources():
    tree = ast.parse((ROOT / 'navsafe/evaluation/eval_env_config.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_eval_env_cfg')
    sentinel = object()
    scope = {'Any': object, 'EnvCfg': object, '_build_py123d_cfg': lambda args: sentinel}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<eval-config>', 'exec'), scope)
    build = scope['build_eval_env_cfg']
    assert build(SimpleNamespace()) is sentinel
    for source in ('sd_pickle', 'pg'):
        with pytest.raises(ValueError, match='py123d'):
            build(SimpleNamespace(scenario_source=source))
