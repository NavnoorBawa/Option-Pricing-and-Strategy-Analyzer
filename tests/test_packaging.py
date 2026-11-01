"""Packaging: the installed app keeps the repository's Streamlit settings and the launcher fails cleanly."""

import builtins
from pathlib import Path

import pytest

from options_analyzer.ui import launch

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_streamlit_config_matches_the_repository_config():
    # Streamlit reads .streamlit/config.toml from the working directory or next to the main script; the installed
    # app runs options_analyzer/ui/streamlit_app.py, so it needs its own copy (shipped as package data).
    repo = (ROOT / ".streamlit" / "config.toml").read_bytes()
    packaged = (ROOT / "options_analyzer" / "ui" / ".streamlit" / "config.toml").read_bytes()
    assert packaged == repo


def test_launcher_without_streamlit_explains_the_app_extra(monkeypatch):
    real_import = builtins.__import__

    def no_streamlit(name, *args, **kwargs):
        if name == "streamlit" or name.startswith("streamlit."):
            raise ModuleNotFoundError(f"No module named {name!r}", name="streamlit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_streamlit)
    with pytest.raises(SystemExit) as exc:
        launch.main()
    assert "'app' extra" in str(exc.value.code)


def _git(*args: str) -> list[str]:
    """Lines of a git command run in the checkout; skips where there is no git repository (an unpacked sdist)."""
    import shutil
    import subprocess

    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_no_tracked_file_is_one_that_gitignore_excludes():
    # Four logs of a failed local run (study_*.log, each one line naming a missing script by its local path) were
    # committed at the repository root; *.log is ignored now, and this catches any other ignored file that slips in.
    assert _git("ls-files", "--cached", "--ignored", "--exclude-standard") == []
