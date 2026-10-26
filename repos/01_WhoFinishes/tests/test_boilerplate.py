"""The non-evidential file denylist typeset in Appendix ``app:boilerplate``.

Every rule in that table has to fire at repository root, not only inside a
subdirectory: a root-level ``.gitignore`` or ``.cursor/`` config is exactly the
churn the linker's fourth filter must refuse to count as evidence.
"""

from __future__ import annotations

import pytest

from prfix.rq1_fix_rates.boilerplate import is_boilerplate

BOILERPLATE = [
    # dotted basenames and dotted directory prefixes AT ROOT
    ".gitignore",
    ".gitattributes",
    ".DS_Store",
    ".release-please-manifest.json",
    ".cursor/rules.md",
    ".vscode/settings.json",
    ".idea/workspace.xml",
    ".devcontainer/devcontainer.json",
    # the same paths reached through an explicit "./" prefix
    "./.gitignore",
    "./.vscode/settings.json",
    # and nested, where they already worked
    "sub/dir/.gitignore",
    # lockfiles, changelog/version, snapshots, generated artifacts
    "package-lock.json",
    "backend/poetry.lock",
    "CHANGELOG.md",
    "docs/changelog.rst",
    "VERSION",
    "src/__version__.py",
    "src/__snapshots__/App.test.js.snap",
    "api/service_pb2.py",
    "web/routes.generated.ts",
]

EVIDENTIAL = [
    "src/main.py",
    "README.md",
    ".github/workflows/ci.yml",
    "cursor/rules.md",            # not the .cursor/ config directory
    "gitignore",                  # not the dotfile
    "src/version_utils.py",
    "src/snapshots/App.js",
]


@pytest.mark.parametrize("path", BOILERPLATE)
def test_boilerplate_paths_are_denied(path):
    assert is_boilerplate(path) is True


@pytest.mark.parametrize("path", EVIDENTIAL)
def test_evidential_paths_are_kept(path):
    assert is_boilerplate(path) is False
