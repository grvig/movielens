"""Tests for the commit-message gate.

Both rules here exist because they were broken on this project and the fix was expensive:
a multi-paragraph body required a history rewrite, and a rewrite duplicated every commit in
GitHub's contribution graph.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_commit_message import check


def test_a_good_message_passes():
    assert check("Added item-kNN model") == []


def test_trailing_newline_is_fine():
    assert check("Fixed temporal split leakage\n") == []


def test_git_comment_lines_are_ignored():
    message = "Added the harness\n\n# Please enter the commit message\n# On branch main\n"
    assert check(message) == []


def test_an_empty_message_is_rejected():
    assert len(check("")) == 1
    assert len(check("\n# only comments\n")) == 1


def test_a_body_is_rejected():
    problems = check("Added the harness\n\nIt does a thing worth explaining.\n")
    assert any("one" in problem for problem in problems)


def test_a_bullet_list_body_is_rejected():
    assert len(check("Added the harness\n\n- one thing\n- another\n")) > 0


def test_an_over_long_subject_is_rejected():
    assert len(check("Added " + "x" * 100)) > 0


def test_an_attribution_trailer_is_rejected():
    message = "Added the harness\n\nCo-Authored-By: Someone <a@b.c>\n"
    problems = check(message)
    assert any("attribution trailer" in problem for problem in problems)


def test_an_unfamiliar_trailer_shape_is_also_rejected():
    """Trailers are matched by shape, so one never seen before still fails."""
    assert len(check("Added it\n\nReviewed-By: Someone\n")) > 0
    assert len(check("Added it\n\nSigned-off-by: Someone\n")) > 0


def test_a_generated_with_line_is_rejected():
    assert len(check("Added it\n\nGenerated with a tool\n")) > 0


def test_an_assistance_credit_is_rejected():
    assert len(check("Added it\n\nAssisted by a tool\n")) > 0


def test_the_check_is_case_insensitive():
    assert len(check("Added it\n\nCO-AUTHORED-BY: x\n")) > 0


def test_every_commit_in_this_repository_would_pass():
    """The convention the repository already follows must satisfy its own gate."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    output = subprocess.run(
        ["git", "log", "--format=%B%x00"],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    messages = [m for m in output.stdout.split("\x00") if m.strip() != ""]
    assert len(messages) > 40
    for message in messages:
        assert check(message) == [], message
