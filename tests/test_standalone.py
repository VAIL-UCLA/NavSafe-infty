"""Extraction contracts that do not require NVIDIA or policy weights."""
import ast
import importlib.resources as resources
from pathlib import Path
import navsafe


def test_no_old_package_imports():
    root = Path(navsafe.__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert all(n != "nexussim" and not n.startswith("nexussim.") for n in names), path


def test_runtime_resources_are_packaged():
    root = resources.files("navsafe")
    assert root.joinpath("benchmark/eval/run_bundle_eval.sh").is_file()
    assert root.joinpath("benchmark/editor/static/index.html").is_file()
    assert root.joinpath("benchmark/recipes/benchmark/R-4.a2c2e046132e5596.yaml").is_file()
    assert root.joinpath("cli/eval_entry.py").is_file()
    assert root.joinpath("gs3d_converter/configs/car2sim_6cam_static.yaml").is_file()
    assert root.joinpath("benchmark/rubric/families/F3_left_turn.yaml").is_file()
    assert root.joinpath("modelzoo/navsim/sparsedrivev2/ops/src/deformable_aggregation_cuda.cu").is_file()


def test_vendored_social_force_is_installable():
    import pysocialforce
    assert callable(pysocialforce.Simulator)


def test_social_force_import_has_no_logging_side_effects(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", "import logging; root=logging.getLogger(); before=(root.level,list(root.handlers)); import pysocialforce; assert before==(root.level,list(root.handlers))"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "file.log").exists()
