# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The human tier of selection: reading a reviewer's sheet, and what it decides.

The point of these is the SEPARATION. A reviewer's verdict and a geometry
predicate answer different questions, so the row where they disagree has to
survive as a disagreement rather than being collapsed into one status.
"""

from __future__ import annotations

import pytest

from navsafe.benchmark.mining.mine import MineRow
from navsafe.benchmark.mining.review import ReviewError, read_review, review_for

HEADER = ("Scenario ID (token),Taxonomy Leaf,Scenario Type(s),Bundle Status,"
          "Keep? (Y/N),For Editing 90,Reviewer Notes\n")


def _sheet(tmp_path, *rows, header=HEADER):
    path = tmp_path / "review.csv"
    path.write_text(header + "".join(rows))
    return path


class TestReadReview:
    def test_only_assigned_rows_are_read(self, tmp_path):
        # The sheet lists the whole corpus; most of it has no leaf, and that is
        # not a finding.
        path = _sheet(
            tmp_path,
            "aaa,C-5,x,Published,Y,C-7,\n",
            "bbb,C-5,x,Published,Y,,\n",
        )
        out = read_review(path)
        assert list(out) == ["C-7"] and [r.token for r in out["C-7"]] == ["aaa"]

    def test_one_cell_can_name_two_leaves(self, tmp_path):
        # R-3 and R-4 are the same template with a different asset family, so
        # "something crosses the road here" is a verdict about both.
        out = read_review(_sheet(tmp_path, "aaa,R-1,x,Published,Y,R-3/R-4,no crosswalk\n"))
        assert sorted(out) == ["R-3", "R-4"]
        assert out["R-3"][0].note == "no crosswalk"
        assert out["R-4"][0].token == "aaa"

    def test_maybe_counts_as_selected_and_keeps_its_reason(self, tmp_path):
        out = read_review(_sheet(tmp_path, "aaa,x,x,Not ready,Maybe,V-10,quality\n"))
        row = out["V-10"][0]
        assert row.selected and not row.rejected
        assert row.keep == "Maybe" and row.note == "quality"

    def test_n_is_a_rejection(self, tmp_path):
        row = read_review(_sheet(tmp_path, "aaa,x,x,Published,N,C-10,\n"))["C-10"][0]
        assert row.rejected and not row.selected

    def test_a_renamed_column_refuses_rather_than_reviewing_nothing(self, tmp_path):
        # Silently reading zero verdicts looks exactly like an unreviewed
        # corpus, which is the failure this refuses to have.
        path = _sheet(tmp_path, "aaa,x,x,Published,Y,C-7,\n",
                      header="token,leaf,keep\n")
        with pytest.raises(ReviewError, match="missing column"):
            read_review(path)

    def test_an_unknown_leaf_is_skipped_not_invented(self, tmp_path):
        out = read_review(_sheet(tmp_path, "aaa,x,x,Published,Y,Q-99,\n"))
        assert out == {}

    def test_review_for_keys_by_token(self, tmp_path):
        picks = review_for(
            _sheet(tmp_path, "aaa,x,x,Published,Y,R-2,\n", "bbb,x,x,Published,N,R-2,\n"),
            "R-2",
        )
        assert sorted(picks) == ["aaa", "bbb"]
        assert picks["aaa"].selected and picks["bbb"].rejected


class TestTheTwoAxesStaySeparate:
    @staticmethod
    def _row(status, keep):
        return MineRow(leaf="R-2", scenario_id="s", token="t", status=status,
                       human={"keep": keep, "selected": keep in ("Y", "Maybe"),
                              "rejected": keep == "N"})

    def test_agreement_is_not_a_dispute(self):
        assert not self._row("qualified", "Y").disputed
        assert not self._row("rejected_host", "N").disputed

    def test_reviewer_yes_geometry_no_is_a_dispute(self):
        # The interesting row: a person saw the scenario, and the host has no
        # lane a cyclist belongs in. One of the two is wrong and neither can
        # say which.
        row = self._row("rejected_host", "Y")
        assert row.picked and not row.ok and row.disputed
        assert "reviewer:Y!" in row.describe()

    def test_reviewer_no_geometry_yes_is_a_dispute(self):
        row = self._row("qualified", "N")
        assert row.ok and not row.picked and row.disputed

    def test_an_unreviewed_row_is_never_disputed(self):
        assert not MineRow(leaf="R-2", scenario_id="s", token="t",
                           status="qualified").disputed

    def test_the_verdict_survives_a_round_trip(self, tmp_path):
        from navsafe.benchmark.mining.mine import load_rows, write_rows

        path = write_rows([self._row("qualified", "Maybe")], tmp_path / "R-2.jsonl")
        back = load_rows(path)[0]
        assert back.human["keep"] == "Maybe" and back.picked
