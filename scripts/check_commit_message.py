"""Enforce this project's commit message convention.

Run as a commit-msg hook by .pre-commit-config.yaml, or directly::

    python scripts/check_commit_message.py .git/COMMIT_EDITMSG

Two rules, both of which have cost this project real time when broken.

**One line, no body.** The convention is a single short past-tense line. Rationale belongs
in module docstrings and the README, next to the code it explains, where it survives a
rebase.

**No attribution trailers.** A trailer that reaches a public remote can only be removed by
rewriting history, and rewriting history duplicates every commit in GitHub's contribution
graph. Catching it here costs nothing; catching it after a push cost two repository
recreations. Trailers are matched by shape rather than by name, so the rule keeps working
for ones this project has never seen.
"""

import re
import sys
from pathlib import Path

# Attribution trailers, matched by shape rather than by naming any particular tool: any
# "Something-By:" line, and the phrasings that usually accompany one. Matching the shape
# means the rule keeps working for trailers this project has never seen.
BANNED_PATTERNS = [
    (r"^[a-z][a-z-]*-by:", "an attribution trailer"),
    (r"generated with", "a generated-with line"),
    (r"assisted by", "an assistance credit"),
]
MAX_SUBJECT_LENGTH = 90


def content_lines(message):
    lines = []
    for line in message.split("\n"):
        if line.startswith("#"):
            continue
        if line.strip() == "":
            continue
        lines.append(line.rstrip())
    return lines


def check(message):
    """Return a list of problems; empty means the message is fine."""
    problems = []
    lines = content_lines(message)

    if len(lines) == 0:
        problems.append("the commit message is empty")
        return problems

    if len(lines) > 1:
        problems.append(
            "the message has " + str(len(lines)) + " content lines; this project uses one "
            "short past-tense line with no body. Put the reasoning in a docstring or the "
            "README instead."
        )

    subject = lines[0]
    if len(subject) > MAX_SUBJECT_LENGTH:
        problems.append(
            "the subject is " + str(len(subject)) + " characters; keep it under "
            + str(MAX_SUBJECT_LENGTH)
        )

    lowered = message.lower()
    for pattern, description in BANNED_PATTERNS:
        if re.search(pattern, lowered, flags=re.MULTILINE):
            problems.append(
                "the message looks like it contains " + description
                + "; this project carries no attribution trailers"
            )
    return problems


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/check_commit_message.py <path to message file>")
        raise SystemExit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print("commit message file not found: " + str(path))
        raise SystemExit(2)

    problems = check(path.read_text(encoding="utf-8", errors="replace"))
    if len(problems) == 0:
        raise SystemExit(0)

    print("commit message rejected:")
    for problem in problems:
        print("  - " + problem)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
