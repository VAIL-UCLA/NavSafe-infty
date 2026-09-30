"""Tests for trajectory scorers.

This file contains unit tests and property-based tests for the migrated scorers.
"""

import pytest
import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from navsafe.evaluation.scorers.cls_scorer import ClsScorer


# ---------------------------------------------------------------------------
# CLS Scorer — Unit Tests
# ---------------------------------------------------------------------------


class TestClsScorerUnit:
    """Unit tests for ClsScorer with known inputs and error cases."""

    def test_known_argmax(self):
        """CLS scorer selects the candidate with the highest confidence score."""
        scorer = ClsScorer()
        # B=1, N=4, T=8, 3 coords
        candidates = torch.randn(1, 4, 8, 3)
        # Candidate index 2 has the highest score
        scores = torch.tensor([[0.1, 0.3, 0.9, 0.2]])

        result = scorer.select_best({
            "confidence_scores": scores,
            "all_candidates": candidates,
        })

        assert result["best_idx"].item() == 2
        assert torch.equal(result["trajectory"], candidates[0:1, 2])
        assert torch.equal(result["scores"], scores)

    def test_batch_argmax(self):
        """CLS scorer handles multiple batch elements correctly."""
        scorer = ClsScorer()
        # B=2, N=3, T=4, 3 coords
        candidates = torch.randn(2, 3, 4, 3)
        scores = torch.tensor([
            [0.1, 0.5, 0.3],  # best = 1
            [0.8, 0.2, 0.4],  # best = 0
        ])

        result = scorer.select_best({
            "confidence_scores": scores,
            "all_candidates": candidates,
        })

        assert result["best_idx"][0].item() == 1
        assert result["best_idx"][1].item() == 0
        assert torch.equal(result["trajectory"][0], candidates[0, 1])
        assert torch.equal(result["trajectory"][1], candidates[1, 0])

    def test_missing_confidence_scores_raises_value_error(self):
        """CLS scorer raises ValueError when confidence_scores is missing."""
        scorer = ClsScorer()
        candidates = torch.randn(1, 4, 8, 3)

        with pytest.raises(ValueError, match="confidence_scores not found in model_output"):
            scorer.select_best({"all_candidates": candidates})

    def test_none_confidence_scores_raises_value_error(self):
        """CLS scorer raises ValueError when confidence_scores is None."""
        scorer = ClsScorer()
        candidates = torch.randn(1, 4, 8, 3)

        with pytest.raises(ValueError, match="confidence_scores not found in model_output"):
            scorer.select_best({
                "confidence_scores": None,
                "all_candidates": candidates,
            })


# ---------------------------------------------------------------------------
# CLS Scorer — Property-Based Tests
# ---------------------------------------------------------------------------


# Feature: bridgesim-to-navsafe-migration, Property 1: CLS scorer selects argmax with correct output structure
# **Validates: Requirements 1.2, 1.3**
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    num_candidates=st.integers(min_value=1, max_value=32),
    timesteps=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=100)
def test_cls_scorer_argmax_property(batch_size, num_candidates, timesteps):
    """Property 1: CLS scorer selects argmax with correct output structure.

    For any batch of confidence_scores of shape (B, N) and all_candidates of
    shape (B, N, T, 3), the CLS scorer's select_best shall return:
    - best_idx equal to argmax(confidence_scores, dim=-1)
    - trajectory of shape (B, T, 3) matching all_candidates[b, best_idx[b]]
    - scores of shape (B, N) equal to the input confidence_scores
    """
    scorer = ClsScorer()

    B, N, T = batch_size, num_candidates, timesteps
    candidates = torch.randn(B, N, T, 3)
    scores = torch.randn(B, N)

    result = scorer.select_best({
        "confidence_scores": scores,
        "all_candidates": candidates,
    })

    expected_best_idx = scores.argmax(dim=-1)

    # best_idx matches argmax
    assert torch.equal(result["best_idx"], expected_best_idx), (
        f"best_idx mismatch: got {result['best_idx']}, expected {expected_best_idx}"
    )

    # trajectory shape is (B, T, 3)
    assert result["trajectory"].shape == (B, T, 3), (
        f"trajectory shape mismatch: got {result['trajectory'].shape}, expected ({B}, {T}, 3)"
    )

    # trajectory matches candidates[b, best_idx[b]] for each batch element
    for b in range(B):
        assert torch.equal(
            result["trajectory"][b], candidates[b, expected_best_idx[b]]
        ), f"trajectory mismatch at batch element {b}"

    # scores are passed through unchanged
    assert result["scores"].shape == (B, N), (
        f"scores shape mismatch: got {result['scores'].shape}, expected ({B}, {N})"
    )
    assert torch.equal(result["scores"], scores), "scores should be passed through unchanged"


# ---------------------------------------------------------------------------
# GT Scorer — Unit Tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock

from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (
    EPDMSTrajectoryScorer_Fast,
)
from navsafe.evaluation.scorers.gt_scorer import (
    GtScorer,
    EPDMS_METRIC_KEYS,
    _METRIC_KEY_MAP,
)

# ego_state passed to GT/TTA select_best calls; the mocked engine ignores it.
_EGO_STATE = {"position": np.zeros(3), "speed": 0.0}


def _minimal_scenario():
    """Smallest scenario dict the real EPDMS engine initializes on (env=None)."""
    return {
        "metadata": {"sdc_id": "ego", "timestep": 0.1},
        "tracks": {
            "ego": {
                "state": {
                    "position": np.zeros((10, 3)),
                    "valid": np.ones(10, dtype=bool),
                }
            }
        },
        "map_features": {},
    }


class TestGtScorerUnit:
    """Unit tests for GtScorer with precomputed engine scores and error cases."""

    def _make_mock_epdms(self, batch_returns):
        """Create a mock EPDMSTrajectoryScorer_Fast with canned batch results.

        Args:
            batch_returns: list of (scores, metrics) tuples, one per batch
                element in b order. ``scores`` is an array-like of N composite
                scores; ``metrics`` is a list of N short-key metric dicts.
        """
        mock_epdms = MagicMock()
        mock_epdms.score_candidates = MagicMock(side_effect=[
            (np.asarray(scores, dtype=np.float64), metrics)
            for scores, metrics in batch_returns
        ])
        return mock_epdms

    @staticmethod
    def _metrics(nc=1.0, dac=1.0, ddc=1.0, tlc=1.0, ep=1.0,
                 ttc=1.0, lk=1.0, hc=1.0, ec=1.0):
        """Build a short-key metric dict as returned by the fast engine."""
        return {"nc": nc, "dac": dac, "ddc": ddc, "tlc": tlc,
                "ep": ep, "ttc": ttc, "lk": lk, "hc": hc, "ec": ec}

    def test_known_epdms_values(self):
        """GT scorer passes engine scores through and selects the highest."""
        scorer = GtScorer()

        # Candidate 0: perfect; candidate 1: collision → 0; candidate 2: moderate
        metrics = [
            self._metrics(),
            self._metrics(nc=0.0),
            self._metrics(ep=0.5, ttc=0.8, lk=0.9, hc=0.7, ec=0.6),
        ]
        scorer.epdms = self._make_mock_epdms([([1.0, 0.0, 0.55], metrics)])

        candidates = torch.randn(1, 3, 8, 3)
        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        assert result["best_idx"].item() == 0
        assert result["trajectory"].shape == (1, 8, 3)
        assert result["scores"].shape == (1, 3)
        # Engine scores are passed through unchanged
        assert result["scores"][0, 0].item() == pytest.approx(1.0)
        assert result["scores"][0, 1].item() == pytest.approx(0.0)
        assert result["scores"][0, 2].item() == pytest.approx(0.55)

        # Verify all 9 metric keys present
        for key in EPDMS_METRIC_KEYS:
            assert key in result["per_metric_scores"]
            assert result["per_metric_scores"][key].shape == (1, 3)

    def test_batch_selection(self):
        """GT scorer handles multiple batch elements correctly."""
        scorer = GtScorer()

        # B=2, N=2 → one score_candidates call per batch element
        scorer.epdms = self._make_mock_epdms([
            # Batch 0: candidate 1 is better
            ([0.3, 0.9], [self._metrics(ep=0.3), self._metrics(ep=0.9)]),
            # Batch 1: candidate 0 is better
            ([1.0, 0.0], [self._metrics(), self._metrics(nc=0.0)]),
        ])

        candidates = torch.randn(2, 2, 4, 3)
        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        assert result["best_idx"][0].item() == 1
        assert result["best_idx"][1].item() == 0

    def test_missing_fields_raises_value_error(self):
        """GT scorer raises ValueError when scenario data is missing required fields."""
        scorer = GtScorer()

        # Missing 'tracks' and 'map_features'
        with pytest.raises(ValueError, match="Missing required fields"):
            scorer.initialize({"metadata": {}}, env=MagicMock())

    def test_missing_all_fields_raises_value_error(self):
        """GT scorer lists all missing fields."""
        scorer = GtScorer()

        with pytest.raises(ValueError, match="Missing required fields"):
            scorer.initialize({}, env=MagicMock())

    def test_initialize_minimal_scenario(self):
        """initialize builds a real EPDMS engine from a minimal valid scenario."""
        scorer = GtScorer()
        scorer.initialize(_minimal_scenario(), env=None)

        assert isinstance(scorer.epdms, EPDMSTrajectoryScorer_Fast)

    def test_uninitialized_raises_value_error(self):
        """GT scorer raises ValueError when select_best called without initialize."""
        scorer = GtScorer()
        candidates = torch.randn(1, 2, 8, 3)

        with pytest.raises(ValueError, match="not initialized"):
            scorer.select_best(
                {"all_candidates": candidates},
                ego_state=_EGO_STATE,
                frame_idx=0,
            )

    def test_per_metric_scores_values(self):
        """GT scorer maps short engine metric keys to UPPER report keys."""
        scorer = GtScorer()

        metrics0 = self._metrics(
            nc=0.9, dac=0.8, ddc=0.7, tlc=0.6, ep=0.5, ttc=0.4, lk=0.3, hc=0.2, ec=0.1
        )
        scorer.epdms = self._make_mock_epdms([([0.42], [metrics0])])

        candidates = torch.randn(1, 1, 8, 3)
        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        pms = result["per_metric_scores"]
        assert pms["NC"][0, 0].item() == pytest.approx(0.9)
        assert pms["DAC"][0, 0].item() == pytest.approx(0.8)
        assert pms["DDC"][0, 0].item() == pytest.approx(0.7)
        assert pms["TLC"][0, 0].item() == pytest.approx(0.6)
        assert pms["EP"][0, 0].item() == pytest.approx(0.5)
        assert pms["TTC"][0, 0].item() == pytest.approx(0.4)
        assert pms["LK"][0, 0].item() == pytest.approx(0.3)
        assert pms["HC"][0, 0].item() == pytest.approx(0.2)
        assert pms["EC"][0, 0].item() == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# GT Scorer — Property-Based Tests
# ---------------------------------------------------------------------------


_UNIT_FLOATS = st.floats(
    min_value=0.0, max_value=1.0,
    allow_nan=False, allow_infinity=False,
    allow_subnormal=False,
)


@st.composite
def epdms_metric_values(draw):
    """Strategy that generates a short-key dict of 9 EPDMS metric values in [0, 1].

    Uses a restricted float range to avoid float32 precision issues where
    extremely small scores become indistinguishable from zero.
    """
    return {key: draw(_UNIT_FLOATS) for key in _METRIC_KEY_MAP}


# Feature: bridgesim-to-navsafe-migration, Property 2: GT scorer passes engine scores through and selects highest composite
# **Validates: Requirements 2.2, 2.3, 2.4**
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    num_candidates=st.integers(min_value=1, max_value=8),
    timesteps=st.integers(min_value=2, max_value=8),
    data=st.data(),
)
@settings(max_examples=100)
def test_gt_scorer_epdms_composite_property(batch_size, num_candidates, timesteps, data):
    """Property 2: GT scorer passes engine scores through and selects highest composite.

    For any initialized GT scorer whose engine returns per-candidate composite
    scores and short-key metric dicts, the output per_metric_scores dictionary
    shall contain all 9 EPDMS report keys with values matching the engine's
    short-key metrics, the scores shall be the engine scores passed through,
    the best_idx shall be their argmax, and output shapes shall conform to
    the contract.
    """
    B, N, T = batch_size, num_candidates, timesteps

    # Generate random but known engine scores and metric dicts per (b, n),
    # regrouped into one (scores, metrics) return per batch element.
    all_scores = []
    all_metric_values = []
    batch_returns = []
    for _b in range(B):
        scores_b = [data.draw(_UNIT_FLOATS) for _n in range(N)]
        metrics_b = [data.draw(epdms_metric_values()) for _n in range(N)]
        all_scores.append(scores_b)
        all_metric_values.append(metrics_b)
        batch_returns.append((np.asarray(scores_b, dtype=np.float64), metrics_b))

    scorer = GtScorer()
    scorer.epdms = MagicMock()
    scorer.epdms.score_candidates = MagicMock(side_effect=batch_returns)

    candidates = torch.randn(B, N, T, 3)
    result = scorer.select_best(
        {"all_candidates": candidates},
        ego_state=_EGO_STATE,
        frame_idx=0,
    )

    # 1. All 9 EPDMS report keys present in per_metric_scores
    assert set(result["per_metric_scores"].keys()) == set(EPDMS_METRIC_KEYS), (
        f"Missing metric keys: {set(EPDMS_METRIC_KEYS) - set(result['per_metric_scores'].keys())}"
    )

    # 2. Output shapes conform to contract
    assert result["trajectory"].shape == (B, T, 3), (
        f"trajectory shape: {result['trajectory'].shape}, expected ({B}, {T}, 3)"
    )
    assert result["scores"].shape == (B, N), (
        f"scores shape: {result['scores'].shape}, expected ({B}, {N})"
    )
    assert result["best_idx"].shape == (B,), (
        f"best_idx shape: {result['best_idx'].shape}, expected ({B},)"
    )
    for key in EPDMS_METRIC_KEYS:
        assert result["per_metric_scores"][key].shape == (B, N), (
            f"per_metric_scores[{key}] shape: {result['per_metric_scores'][key].shape}"
        )

    # 3. Engine scores are passed through unchanged (within float32) and
    #    best_idx is their argmax
    for b in range(B):
        for n in range(N):
            assert abs(result["scores"][b, n].item() - all_scores[b][n]) < 1e-5, (
                f"Score at ({b},{n}): got {result['scores'][b, n].item()}, "
                f"expected {all_scores[b][n]}"
            )
        # Use float32 argmax to match the scorer's torch.float32 precision
        expected_best = int(torch.tensor(all_scores[b], dtype=torch.float32).argmax())
        assert result["best_idx"][b].item() == expected_best, (
            f"Batch {b}: best_idx={result['best_idx'][b].item()}, expected={expected_best}, "
            f"scores={all_scores[b]}"
        )

    # 4. Per-metric values match the engine's short-key metrics via _METRIC_KEY_MAP
    for b in range(B):
        for n in range(N):
            for short_key, report_key in _METRIC_KEY_MAP.items():
                actual = result["per_metric_scores"][report_key][b, n].item()
                expected = all_metric_values[b][n][short_key]
                assert abs(actual - expected) < 1e-5, (
                    f"Metric {report_key} at ({b},{n}): got {actual}, expected {expected}"
                )

    # 5. trajectory matches candidates[b, best_idx[b]]
    for b in range(B):
        bi = result["best_idx"][b].item()
        assert torch.equal(result["trajectory"][b], candidates[b, bi]), (
            f"trajectory mismatch at batch {b}"
        )


# ---------------------------------------------------------------------------
# Learned Scorer — Unit Tests
# ---------------------------------------------------------------------------

from navsafe.evaluation.scorers.learned_scorer import LearnedScorer


def _create_mock_checkpoint(path, d_model=16, nhead=2, num_metrics=3):
    """Create a minimal valid checkpoint file at *path* and return the config."""
    config = {"d_model": d_model, "nhead": nhead, "num_metrics": num_metrics}

    # Build a coarse decoder and grab its state dict
    decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, batch_first=True,
    )
    coarse_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=1)
    coarse_decoder_sd = coarse_decoder.state_dict()

    # Build per-metric linear heads and grab their state dict
    fine_heads = torch.nn.ModuleDict()
    for i in range(num_metrics):
        fine_heads[f"metric_{i}"] = torch.nn.Linear(d_model, 1)
    fine_heads_sd = fine_heads.state_dict()

    ckpt = {
        "coarse_decoder_state_dict": coarse_decoder_sd,
        "fine_heads_state_dict": fine_heads_sd,
        "config": config,
    }
    torch.save(ckpt, str(path))
    return config


class TestLearnedScorerUnit:
    """Unit tests for LearnedScorer with mock checkpoint and error cases."""

    def test_basic_select_best(self, tmp_path):
        """LearnedScorer loads a mock checkpoint and returns valid output."""
        ckpt_path = tmp_path / "mock.ckpt"
        _create_mock_checkpoint(ckpt_path, d_model=16, nhead=2, num_metrics=3)

        scorer = LearnedScorer(str(ckpt_path), top_k=3, device="cpu")

        B, N, T = 1, 6, 8
        candidates = torch.randn(B, N, T, 3)
        result = scorer.select_best({"all_candidates": candidates})

        assert result["trajectory"].shape == (B, T, 3)
        assert result["scores"].shape == (B, N)
        assert result["best_idx"].shape == (B,)
        assert 0 <= result["best_idx"].item() < N

        # stage_scores structure
        assert "stage_scores" in result
        assert "coarse" in result["stage_scores"]
        assert "fine" in result["stage_scores"]
        assert result["stage_scores"]["coarse"]["scores"].shape == (B, N)
        assert result["stage_scores"]["coarse"]["topk_idx"].shape == (B, 3)
        assert result["stage_scores"]["fine"]["scores"].shape == (B, 3)

    def test_batch_select_best(self, tmp_path):
        """LearnedScorer handles multiple batch elements."""
        ckpt_path = tmp_path / "mock.ckpt"
        _create_mock_checkpoint(ckpt_path, d_model=16, nhead=2, num_metrics=3)

        scorer = LearnedScorer(str(ckpt_path), top_k=2, device="cpu")

        B, N, T = 2, 5, 4
        candidates = torch.randn(B, N, T, 3)
        result = scorer.select_best({"all_candidates": candidates})

        assert result["trajectory"].shape == (B, T, 3)
        assert result["scores"].shape == (B, N)
        assert result["best_idx"].shape == (B,)
        for b in range(B):
            assert 0 <= result["best_idx"][b].item() < N

    def test_topk_greater_than_n(self, tmp_path):
        """When top_k >= N, coarse stage keeps all candidates."""
        ckpt_path = tmp_path / "mock.ckpt"
        _create_mock_checkpoint(ckpt_path, d_model=16, nhead=2, num_metrics=3)

        scorer = LearnedScorer(str(ckpt_path), top_k=10, device="cpu")

        B, N, T = 1, 3, 4
        candidates = torch.randn(B, N, T, 3)
        result = scorer.select_best({"all_candidates": candidates})

        # top_k clamped to N=3
        assert result["stage_scores"]["coarse"]["topk_idx"].shape == (B, 3)
        assert result["stage_scores"]["fine"]["scores"].shape == (B, 3)

    def test_file_not_found_error(self):
        """LearnedScorer raises FileNotFoundError for missing checkpoint."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            LearnedScorer("/nonexistent/path/model.ckpt", device="cpu")

    def test_runtime_error_incompatible_format(self, tmp_path):
        """LearnedScorer raises RuntimeError for incompatible checkpoint format."""
        bad_ckpt = tmp_path / "bad.ckpt"
        # Save a checkpoint missing required keys
        torch.save({"some_key": "some_value"}, str(bad_ckpt))

        with pytest.raises(RuntimeError, match="Checkpoint format mismatch"):
            LearnedScorer(str(bad_ckpt), device="cpu")

    def test_runtime_error_missing_config_keys(self, tmp_path):
        """LearnedScorer raises RuntimeError when config is missing required keys."""
        bad_ckpt = tmp_path / "bad_config.ckpt"
        torch.save({
            "coarse_decoder_state_dict": {},
            "fine_heads_state_dict": {},
            "config": {"d_model": 16},  # missing nhead and num_metrics
        }, str(bad_ckpt))

        with pytest.raises(RuntimeError, match="Checkpoint format mismatch"):
            LearnedScorer(str(bad_ckpt), device="cpu")

    def test_runtime_error_non_dict_checkpoint(self, tmp_path):
        """LearnedScorer raises RuntimeError when checkpoint is not a dict."""
        bad_ckpt = tmp_path / "not_dict.ckpt"
        torch.save("not a dict", str(bad_ckpt))

        with pytest.raises(RuntimeError, match="Checkpoint format mismatch"):
            LearnedScorer(str(bad_ckpt), device="cpu")

    def test_stage_scores_coarse_fine_entries(self, tmp_path):
        """stage_scores contains both coarse and fine entries with correct shapes."""
        ckpt_path = tmp_path / "mock.ckpt"
        _create_mock_checkpoint(ckpt_path, d_model=16, nhead=2, num_metrics=4)

        scorer = LearnedScorer(str(ckpt_path), top_k=3, device="cpu")

        B, N, T = 1, 8, 6
        candidates = torch.randn(B, N, T, 3)
        result = scorer.select_best({"all_candidates": candidates})

        coarse = result["stage_scores"]["coarse"]
        fine = result["stage_scores"]["fine"]

        assert coarse["scores"].shape == (B, N)
        assert coarse["topk_idx"].shape == (B, 3)
        assert coarse["topk_scores"].shape == (B, 3)
        assert fine["scores"].shape == (B, 3)
        assert fine["best_local_idx"].shape == (B,)
        assert "per_metric" in fine
        assert len(fine["per_metric"]) == 4  # num_metrics


# ---------------------------------------------------------------------------
# Learned Scorer — Property-Based Tests
# ---------------------------------------------------------------------------


# Feature: bridgesim-to-navsafe-migration, Property 3: Learned scorer coarse-to-fine pipeline preserves candidate count invariant
# **Validates: Requirements 3.2, 3.4**
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    num_candidates=st.integers(min_value=2, max_value=16),
    timesteps=st.integers(min_value=1, max_value=8),
    top_k=st.integers(min_value=1, max_value=8),
    data=st.data(),
)
@settings(max_examples=100, deadline=None)
def test_learned_scorer_coarse_fine_property(batch_size, num_candidates, timesteps, top_k, data, tmp_path_factory):
    """Property 3: Learned scorer coarse-to-fine pipeline preserves candidate count invariant.

    For any valid input with N candidates and configured top_k K where K < N,
    the Learned scorer's coarse stage shall reduce to exactly K candidates,
    the fine stage shall rank among those K, stage_scores shall contain both
    coarse and fine entries, and the output best_idx shall index into the
    original N candidates.
    """
    from hypothesis import assume

    B, N, T, K = batch_size, num_candidates, timesteps, top_k
    # We need K < N for the coarse stage to actually filter
    assume(K < N)

    ckpt_path = tmp_path_factory.mktemp("ckpt") / "mock.ckpt"
    _create_mock_checkpoint(ckpt_path, d_model=16, nhead=2, num_metrics=3)

    scorer = LearnedScorer(str(ckpt_path), top_k=K, device="cpu")

    candidates = torch.randn(B, N, T, 3)
    result = scorer.select_best({"all_candidates": candidates})

    # 1. stage_scores contains both coarse and fine entries
    assert "stage_scores" in result, "Missing stage_scores in output"
    assert "coarse" in result["stage_scores"], "Missing coarse in stage_scores"
    assert "fine" in result["stage_scores"], "Missing fine in stage_scores"

    coarse = result["stage_scores"]["coarse"]
    fine = result["stage_scores"]["fine"]

    # 2. Coarse stage scores all N candidates
    assert coarse["scores"].shape == (B, N), (
        f"Coarse scores shape: {coarse['scores'].shape}, expected ({B}, {N})"
    )

    # 3. Coarse stage reduces to exactly K candidates
    assert coarse["topk_idx"].shape == (B, K), (
        f"topk_idx shape: {coarse['topk_idx'].shape}, expected ({B}, {K})"
    )
    assert coarse["topk_scores"].shape == (B, K), (
        f"topk_scores shape: {coarse['topk_scores'].shape}, expected ({B}, {K})"
    )

    # 4. Fine stage ranks among those K candidates
    assert fine["scores"].shape == (B, K), (
        f"Fine scores shape: {fine['scores'].shape}, expected ({B}, {K})"
    )

    # 5. Output best_idx indexes into the original N candidates
    assert result["best_idx"].shape == (B,), (
        f"best_idx shape: {result['best_idx'].shape}, expected ({B},)"
    )
    for b in range(B):
        idx = result["best_idx"][b].item()
        assert 0 <= idx < N, (
            f"best_idx[{b}]={idx} out of range [0, {N})"
        )

    # 6. best_idx is one of the top-K indices
    for b in range(B):
        idx = result["best_idx"][b].item()
        topk_indices = coarse["topk_idx"][b].tolist()
        assert idx in topk_indices, (
            f"best_idx[{b}]={idx} not in topk_idx={topk_indices}"
        )

    # 7. Output trajectory and scores have correct shapes
    assert result["trajectory"].shape == (B, T, 3), (
        f"trajectory shape: {result['trajectory'].shape}, expected ({B}, {T}, 3)"
    )
    assert result["scores"].shape == (B, N), (
        f"scores shape: {result['scores'].shape}, expected ({B}, {N})"
    )


# ---------------------------------------------------------------------------
# TTA Scorer — Unit Tests
# ---------------------------------------------------------------------------

from navsafe.evaluation.scorers.tta_scorer import TtaScorer


class TestTtaScorerUnit:
    """Unit tests for TtaScorer with known continuity/collision values and edge cases."""

    def _make_mock_epdms(self, batch_scores):
        """Create a mock EPDMSTrajectoryScorer_Fast with canned batch scores.

        Args:
            batch_scores: list of per-batch score lists, one per batch element
                in b order (TtaScorer discards the metrics half of the tuple).
        """
        mock_epdms = MagicMock()
        mock_epdms.score_candidates = MagicMock(side_effect=[
            (np.asarray(scores, dtype=np.float64), []) for scores in batch_scores
        ])
        return mock_epdms

    def test_known_continuity_penalty(self):
        """TTA scorer computes correct L2 continuity penalty against previous plan."""
        scorer = TtaScorer(continuity_weight=1.0, collision_weight=0.0)

        # B=1, N=2, T=4, 3 coords
        # Candidate 0: all zeros
        # Candidate 1: all ones
        candidates = torch.zeros(1, 2, 4, 3)
        candidates[0, 1, :, :] = 1.0

        # Previous plan: all zeros → candidate 0 has zero penalty, candidate 1 has nonzero
        scorer.prev_plan = torch.zeros(4, 3)

        # Mock EPDMS to return equal scores so continuity decides
        scorer.epdms = self._make_mock_epdms([[0.5, 0.5]])

        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        # Candidate 0 penalty should be 0
        assert result["continuity_penalty"][0, 0].item() == pytest.approx(0.0)
        # Candidate 1 penalty should be L2 norm of ones vector (4*3=12 elements, each 1.0)
        expected_penalty = torch.norm(torch.ones(12), p=2).item()
        assert result["continuity_penalty"][0, 1].item() == pytest.approx(expected_penalty, rel=1e-5)

        # Candidate 0 should be selected (lower penalty)
        assert result["best_idx"].item() == 0

    def test_first_timestep_zero_continuity(self):
        """TTA scorer sets continuity penalty to zero on first timestep (no prev_plan)."""
        scorer = TtaScorer(continuity_weight=1.0, collision_weight=0.0)
        scorer.prev_plan = None  # First timestep

        candidates = torch.randn(1, 3, 4, 3)
        scorer.epdms = self._make_mock_epdms([[0.5, 0.8, 0.3]])

        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        # All continuity penalties should be zero
        assert torch.all(result["continuity_penalty"] == 0.0)
        # Best should be candidate 1 (highest EPDMS score, no penalty)
        assert result["best_idx"].item() == 1

    def test_collision_scoring_with_agents(self):
        """TTA scorer computes collision scores from extrapolated agent positions."""
        scorer = TtaScorer(continuity_weight=0.0, collision_weight=1.0)
        scorer.prev_plan = None

        # B=1, N=2, T=4
        # Candidate 0: moves away from agent
        # Candidate 1: moves toward agent
        candidates = torch.zeros(1, 2, 4, 3)
        # Candidate 0: at (100, 100) — far from agent
        candidates[0, 0, :, 0] = 100.0
        candidates[0, 0, :, 1] = 100.0
        # Candidate 1: at (0, 0) — right on top of agent
        candidates[0, 1, :, 0] = 0.0
        candidates[0, 1, :, 1] = 0.0

        agent_states = {
            "positions": np.array([[0.0, 0.0]]),  # Agent at origin
            "velocities": np.array([[0.0, 0.0]]),  # Stationary
            "sizes": np.array([[4.0, 2.0]]),  # 4m x 2m vehicle
        }

        scorer.epdms = self._make_mock_epdms([[0.5, 0.5]])

        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
            agent_states=agent_states,
            dt=0.5,
        )

        # Candidate 0 should have zero collision score (far away)
        assert result["collision_scores"][0, 0].item() == 0.0
        # Candidate 1 should have nonzero collision score (overlapping)
        assert result["collision_scores"][0, 1].item() > 0.0
        # Candidate 0 should be selected (no collision)
        assert result["best_idx"].item() == 0

    def test_no_agent_states_zero_collision(self):
        """TTA scorer returns zero collision scores when no agent states provided."""
        scorer = TtaScorer(continuity_weight=0.0, collision_weight=1.0)
        scorer.prev_plan = None

        candidates = torch.randn(1, 2, 4, 3)
        scorer.epdms = self._make_mock_epdms([[0.3, 0.7]])

        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        assert torch.all(result["collision_scores"] == 0.0)

    def test_output_shapes(self):
        """TTA scorer returns all required output keys with correct shapes."""
        scorer = TtaScorer()
        scorer.prev_plan = torch.randn(4, 3)

        B, N, T = 2, 3, 4
        candidates = torch.randn(B, N, T, 3)
        scorer.epdms = self._make_mock_epdms([[0.5] * N for _ in range(B)])

        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        assert result["trajectory"].shape == (B, T, 3)
        assert result["scores"].shape == (B, N)
        assert result["best_idx"].shape == (B,)
        assert result["continuity_penalty"].shape == (B, N)
        assert result["collision_scores"].shape == (B, N)

    def test_uninitialized_raises_value_error(self):
        """TTA scorer raises ValueError when select_best called without initialize."""
        scorer = TtaScorer()
        candidates = torch.randn(1, 2, 4, 3)

        with pytest.raises(ValueError, match="not initialized"):
            scorer.select_best(
                {"all_candidates": candidates},
                ego_state=_EGO_STATE,
                frame_idx=0,
            )

    def test_missing_fields_raises_value_error(self):
        """TTA scorer raises ValueError when scenario data is missing required fields."""
        scorer = TtaScorer()

        with pytest.raises(ValueError, match="Missing required fields"):
            scorer.initialize({"metadata": {}}, env=MagicMock())

    def test_initialize_minimal_scenario(self):
        """initialize builds a real EPDMS engine and resets the plan state."""
        scorer = TtaScorer()
        scorer.prev_plan = torch.randn(4, 3)

        scenario = _minimal_scenario()
        scorer.initialize(scenario, env=None)

        assert isinstance(scorer.epdms, EPDMSTrajectoryScorer_Fast)
        assert scorer.prev_plan is None
        assert scorer.agent_predictions is scenario["tracks"]

    def test_prev_plan_updated_after_select(self):
        """TTA scorer updates prev_plan after each select_best call."""
        scorer = TtaScorer(continuity_weight=0.0, collision_weight=0.0)
        scorer.prev_plan = None

        candidates = torch.randn(1, 2, 4, 3)
        scorer.epdms = self._make_mock_epdms([[0.3, 0.7]])

        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        # prev_plan should now be set to the best trajectory
        assert scorer.prev_plan is not None
        assert scorer.prev_plan.shape == (4, 3)


# ---------------------------------------------------------------------------
# TTA Scorer — Property-Based Tests
# ---------------------------------------------------------------------------


# Feature: bridgesim-to-navsafe-migration, Property 4: TTA scorer augments EPDMS with continuity and collision scores
# **Validates: Requirements 4.2, 4.4, 4.5**
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    num_candidates=st.integers(min_value=1, max_value=8),
    timesteps=st.integers(min_value=2, max_value=8),
    num_agents=st.integers(min_value=0, max_value=4),
    has_prev_plan=st.booleans(),
)
@settings(max_examples=100, deadline=None)
def test_tta_scorer_augmented_output_property(
    batch_size, num_candidates, timesteps, num_agents, has_prev_plan
):
    """Property 4: TTA scorer augments EPDMS with continuity and collision scores.

    For any initialized TTA scorer with valid candidates, previous plan, and
    agent state predictions, the output shall contain continuity_penalty of
    shape (B, N) and collision_scores of shape (B, N) alongside the standard
    scorer outputs (trajectory, scores, best_idx), and the final ranking shall
    incorporate all three signal types.
    """
    B, N, T = batch_size, num_candidates, timesteps

    scorer = TtaScorer(gamma=0.99, continuity_weight=1.0, collision_weight=1.0)

    # Set up previous plan
    if has_prev_plan:
        scorer.prev_plan = torch.randn(T, 3)
    else:
        scorer.prev_plan = None

    # Mock EPDMS — one (scores, metrics) return per batch element
    epdms_scores = torch.rand(B, N)
    mock_results = [
        (epdms_scores[b].numpy().astype(np.float64), []) for b in range(B)
    ]
    scorer.epdms = MagicMock()
    scorer.epdms.score_candidates = MagicMock(side_effect=mock_results)

    candidates = torch.randn(B, N, T, 3)

    # Build agent states
    agent_states = None
    if num_agents > 0:
        agent_states = {
            "positions": np.random.randn(num_agents, 2).astype(np.float32),
            "velocities": np.random.randn(num_agents, 2).astype(np.float32),
            "sizes": np.abs(np.random.randn(num_agents, 2).astype(np.float32)) + 1.0,
        }

    result = scorer.select_best(
        {"all_candidates": candidates},
        ego_state=_EGO_STATE,
        frame_idx=0,
        agent_states=agent_states,
        dt=0.5,
    )

    # 1. Output contains all required keys
    assert "trajectory" in result
    assert "scores" in result
    assert "best_idx" in result
    assert "continuity_penalty" in result
    assert "collision_scores" in result

    # 2. Shapes are correct
    assert result["trajectory"].shape == (B, T, 3), (
        f"trajectory shape: {result['trajectory'].shape}, expected ({B}, {T}, 3)"
    )
    assert result["scores"].shape == (B, N), (
        f"scores shape: {result['scores'].shape}, expected ({B}, {N})"
    )
    assert result["best_idx"].shape == (B,), (
        f"best_idx shape: {result['best_idx'].shape}, expected ({B},)"
    )
    assert result["continuity_penalty"].shape == (B, N), (
        f"continuity_penalty shape: {result['continuity_penalty'].shape}, expected ({B}, {N})"
    )
    assert result["collision_scores"].shape == (B, N), (
        f"collision_scores shape: {result['collision_scores'].shape}, expected ({B}, {N})"
    )

    # 3. best_idx is valid
    for b in range(B):
        idx = result["best_idx"][b].item()
        assert 0 <= idx < N, f"best_idx[{b}]={idx} out of range [0, {N})"

    # 4. trajectory matches candidates[b, best_idx[b]]
    for b in range(B):
        bi = result["best_idx"][b].item()
        assert torch.equal(result["trajectory"][b], candidates[b, bi]), (
            f"trajectory mismatch at batch {b}"
        )

    # 5. First timestep edge case: continuity penalty is zero when no prev_plan
    if not has_prev_plan:
        assert torch.all(result["continuity_penalty"] == 0.0), (
            "continuity_penalty should be zero when no previous plan"
        )

    # 6. Final ranking incorporates all three signal types
    # Verify composite = epdms - continuity_weight * penalty - collision_weight * collision
    expected_composite = (
        epdms_scores
        - scorer.continuity_weight * result["continuity_penalty"]
        - scorer.collision_weight * result["collision_scores"]
    )
    assert torch.allclose(result["scores"], expected_composite, atol=1e-5), (
        "Final scores should be composite of EPDMS, continuity, and collision"
    )

    # 7. Continuity penalties are non-negative
    assert torch.all(result["continuity_penalty"] >= 0.0), (
        "continuity_penalty should be non-negative"
    )

    # 8. Collision scores are non-negative
    assert torch.all(result["collision_scores"] >= 0.0), (
        "collision_scores should be non-negative"
    )


# Feature: bridgesim-to-navsafe-migration, Property 5: TTA continuity penalty equals L2 distance to previous plan
# **Validates: Requirements 4.3**
@given(
    timesteps=st.integers(min_value=1, max_value=16),
    data=st.data(),
)
@settings(max_examples=100, deadline=None)
def test_tta_continuity_l2_property(timesteps, data):
    """Property 5: TTA continuity penalty equals L2 distance to previous plan.

    For any previous plan trajectory of shape (T, 3) and any candidate
    trajectory of shape (T, 3), the continuity penalty for that candidate
    shall equal the L2 norm of the difference between the candidate and
    the previous plan.
    """
    T = timesteps

    # Generate random previous plan and candidate
    prev_plan = torch.randn(T, 3)
    candidate = torch.randn(T, 3)

    scorer = TtaScorer()
    scorer.prev_plan = prev_plan

    # Wrap candidate as (B=1, N=1, T, 3)
    candidates = candidate.unsqueeze(0).unsqueeze(0)

    penalty = scorer._compute_continuity_penalty(candidates)

    # Expected: L2 norm of flattened difference
    diff = candidate - prev_plan  # (T, 3)
    expected = torch.norm(diff.reshape(-1), p=2)

    assert penalty.shape == (1, 1), f"penalty shape: {penalty.shape}, expected (1, 1)"
    assert torch.allclose(penalty[0, 0], expected, atol=1e-5), (
        f"penalty={penalty[0, 0].item()}, expected={expected.item()}"
    )

    # Also test with multiple candidates
    num_candidates = data.draw(st.integers(min_value=1, max_value=8))
    multi_candidates = torch.randn(1, num_candidates, T, 3)
    multi_penalty = scorer._compute_continuity_penalty(multi_candidates)

    assert multi_penalty.shape == (1, num_candidates)

    for n in range(num_candidates):
        diff_n = multi_candidates[0, n] - prev_plan
        expected_n = torch.norm(diff_n.reshape(-1), p=2)
        assert torch.allclose(multi_penalty[0, n], expected_n, atol=1e-5), (
            f"candidate {n}: penalty={multi_penalty[0, n].item()}, expected={expected_n.item()}"
        )

    # Edge case: candidate equals prev_plan → zero penalty
    same_candidates = prev_plan.unsqueeze(0).unsqueeze(0)
    zero_penalty = scorer._compute_continuity_penalty(same_candidates)
    assert torch.allclose(zero_penalty, torch.zeros(1, 1), atol=1e-7), (
        f"Same candidate as prev_plan should have zero penalty, got {zero_penalty.item()}"
    )


# ---------------------------------------------------------------------------
# Cross-Cutting Scorer Tests — Unit Tests (Task 5)
# ---------------------------------------------------------------------------


class TestScorerCrossCutting:
    """Cross-cutting tests that apply to all four scorers."""

    def test_cls_single_candidate_returns_best_idx_zero(self):
        """CLS scorer returns best_idx=0 for a single candidate (N=1)."""
        scorer = ClsScorer()
        # B=1, N=1, T=8
        candidates = torch.randn(1, 1, 8, 3)
        scores = torch.tensor([[0.42]])

        result = scorer.select_best({
            "confidence_scores": scores,
            "all_candidates": candidates,
        })

        assert result["best_idx"].item() == 0
        assert result["trajectory"].shape == (1, 8, 3)
        assert torch.equal(result["trajectory"], candidates[:, 0])

    def test_gt_single_candidate_returns_best_idx_zero(self):
        """GT scorer returns best_idx=0 for a single candidate (N=1)."""
        scorer = GtScorer()

        # Mock EPDMS to return any score
        mock_epdms = MagicMock()
        mock_epdms.score_candidates = MagicMock(side_effect=[
            (np.array([0.42]), [{
                "nc": 0.9, "dac": 0.8, "ddc": 0.7, "tlc": 0.6, "ep": 0.5,
                "ttc": 0.4, "lk": 0.3, "hc": 0.2, "ec": 0.1,
            }]),
        ])
        scorer.epdms = mock_epdms

        candidates = torch.randn(1, 1, 8, 3)
        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        assert result["best_idx"].item() == 0
        assert result["trajectory"].shape == (1, 8, 3)
        assert torch.equal(result["trajectory"], candidates[:, 0])

    def test_learned_single_candidate_returns_best_idx_zero(self, tmp_path):
        """Learned scorer returns best_idx=0 for a single candidate (N=1)."""
        ckpt_path = tmp_path / "mock.ckpt"
        _create_mock_checkpoint(ckpt_path, d_model=16, nhead=2, num_metrics=3)

        scorer = LearnedScorer(str(ckpt_path), top_k=5, device="cpu")

        candidates = torch.randn(1, 1, 8, 3)
        result = scorer.select_best({"all_candidates": candidates})

        assert result["best_idx"].item() == 0
        assert result["trajectory"].shape == (1, 8, 3)

    def test_tta_single_candidate_returns_best_idx_zero(self):
        """TTA scorer returns best_idx=0 for a single candidate (N=1)."""
        scorer = TtaScorer(continuity_weight=1.0, collision_weight=1.0)
        scorer.prev_plan = torch.randn(8, 3)

        # Mock EPDMS
        mock_epdms = MagicMock()
        mock_epdms.score_candidates = MagicMock(side_effect=[
            (np.array([0.75]), []),
        ])
        scorer.epdms = mock_epdms

        candidates = torch.randn(1, 1, 8, 3)
        result = scorer.select_best(
            {"all_candidates": candidates},
            ego_state=_EGO_STATE,
            frame_idx=0,
        )

        assert result["best_idx"].item() == 0
        assert result["trajectory"].shape == (1, 8, 3)
        assert torch.equal(result["trajectory"], candidates[:, 0])


# ---------------------------------------------------------------------------
# Cross-Cutting Scorer Tests — Property-Based Tests (Task 5)
# ---------------------------------------------------------------------------


# Feature: bridgesim-to-navsafe-migration, Property 7: Scorer determinism
# **Validates: Requirements 7.7**
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    num_candidates=st.integers(min_value=1, max_value=8),
    timesteps=st.integers(min_value=2, max_value=8),
    data=st.data(),
)
@settings(max_examples=100, deadline=None)
def test_scorer_determinism_property(batch_size, num_candidates, timesteps, data, tmp_path_factory):
    """Property 7: Scorer determinism.

    For any scorer (CLS, GT, TTA, and Learned in eval mode) and any valid
    input, calling select_best twice with identical inputs shall produce
    identical outputs (same best_idx, same scores, same trajectory values).
    """
    B, N, T = batch_size, num_candidates, timesteps

    candidates = torch.randn(B, N, T, 3)
    confidence_scores = torch.randn(B, N)

    # --- CLS Scorer ---
    cls_scorer = ClsScorer()
    cls_input = {
        "confidence_scores": confidence_scores.clone(),
        "all_candidates": candidates.clone(),
    }
    cls_result1 = cls_scorer.select_best(
        {"confidence_scores": confidence_scores.clone(), "all_candidates": candidates.clone()}
    )
    cls_result2 = cls_scorer.select_best(
        {"confidence_scores": confidence_scores.clone(), "all_candidates": candidates.clone()}
    )
    assert torch.equal(cls_result1["best_idx"], cls_result2["best_idx"]), (
        "CLS: best_idx not deterministic"
    )
    assert torch.equal(cls_result1["scores"], cls_result2["scores"]), (
        "CLS: scores not deterministic"
    )
    assert torch.equal(cls_result1["trajectory"], cls_result2["trajectory"]), (
        "CLS: trajectory not deterministic"
    )

    # --- GT Scorer (mocked EPDMS) ---
    # Generate fixed EPDMS results for both calls
    epdms_values = []
    for _b in range(B):
        for _n in range(N):
            score_val = data.draw(st.floats(min_value=0.0, max_value=1.0,
                                            allow_nan=False, allow_infinity=False,
                                            allow_subnormal=False))
            epdms_values.append(score_val)

    def _make_gt_scorer_with_mock(values):
        # values is a flat per-candidate list in b-major order; regroup into
        # one (scores, metrics) return per batch element.
        scorer = GtScorer()
        batch_returns = []
        for b in range(B):
            chunk = values[b * N:(b + 1) * N]
            metrics = [{key: s for key in _METRIC_KEY_MAP} for s in chunk]
            batch_returns.append((np.asarray(chunk, dtype=np.float64), metrics))
        scorer.epdms = MagicMock()
        scorer.epdms.score_candidates = MagicMock(side_effect=batch_returns)
        return scorer

    gt_scorer1 = _make_gt_scorer_with_mock(list(epdms_values))
    gt_result1 = gt_scorer1.select_best(
        {"all_candidates": candidates.clone()}, ego_state=_EGO_STATE, frame_idx=0
    )
    gt_scorer2 = _make_gt_scorer_with_mock(list(epdms_values))
    gt_result2 = gt_scorer2.select_best(
        {"all_candidates": candidates.clone()}, ego_state=_EGO_STATE, frame_idx=0
    )
    assert torch.equal(gt_result1["best_idx"], gt_result2["best_idx"]), (
        "GT: best_idx not deterministic"
    )
    assert torch.allclose(gt_result1["scores"], gt_result2["scores"], atol=1e-6), (
        "GT: scores not deterministic"
    )
    assert torch.equal(gt_result1["trajectory"], gt_result2["trajectory"]), (
        "GT: trajectory not deterministic"
    )

    # --- TTA Scorer (mocked EPDMS, with prev_plan) ---
    prev_plan = torch.randn(T, 3)

    def _make_tta_scorer_with_mock(values, prev):
        scorer = TtaScorer(continuity_weight=1.0, collision_weight=0.0)
        scorer.prev_plan = prev.clone()
        batch_returns = [
            (np.asarray(values[b * N:(b + 1) * N], dtype=np.float64), [])
            for b in range(B)
        ]
        scorer.epdms = MagicMock()
        scorer.epdms.score_candidates = MagicMock(side_effect=batch_returns)
        return scorer

    tta_scorer1 = _make_tta_scorer_with_mock(list(epdms_values), prev_plan)
    tta_result1 = tta_scorer1.select_best(
        {"all_candidates": candidates.clone()}, ego_state=_EGO_STATE, frame_idx=0
    )
    tta_scorer2 = _make_tta_scorer_with_mock(list(epdms_values), prev_plan)
    tta_result2 = tta_scorer2.select_best(
        {"all_candidates": candidates.clone()}, ego_state=_EGO_STATE, frame_idx=0
    )
    assert torch.equal(tta_result1["best_idx"], tta_result2["best_idx"]), (
        "TTA: best_idx not deterministic"
    )
    assert torch.allclose(tta_result1["scores"], tta_result2["scores"], atol=1e-6), (
        "TTA: scores not deterministic"
    )
    assert torch.equal(tta_result1["trajectory"], tta_result2["trajectory"]), (
        "TTA: trajectory not deterministic"
    )

    # --- Learned Scorer (eval mode) ---
    ckpt_path = tmp_path_factory.mktemp("det_ckpt") / "mock.ckpt"
    _create_mock_checkpoint(ckpt_path, d_model=16, nhead=2, num_metrics=3)

    learned_scorer = LearnedScorer(str(ckpt_path), top_k=min(3, N), device="cpu")
    # Ensure eval mode
    learned_scorer.coarse_decoder.eval()
    learned_scorer.fine_heads.eval()

    learned_result1 = learned_scorer.select_best(
        {"all_candidates": candidates.clone()}
    )
    learned_result2 = learned_scorer.select_best(
        {"all_candidates": candidates.clone()}
    )
    assert torch.equal(learned_result1["best_idx"], learned_result2["best_idx"]), (
        "Learned: best_idx not deterministic"
    )
    assert torch.allclose(learned_result1["scores"], learned_result2["scores"], atol=1e-6), (
        "Learned: scores not deterministic"
    )
    assert torch.equal(learned_result1["trajectory"], learned_result2["trajectory"]), (
        "Learned: trajectory not deterministic"
    )
