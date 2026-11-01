"""Packaging and the repository: the installed app keeps the repository's Streamlit settings, the launcher fails
cleanly, the checkout holds no ignored files, and every library a test needs is installed by some requirements file."""

import ast
import builtins
import re
import shutil
import subprocess
import sys
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
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_no_tracked_file_is_one_that_gitignore_excludes():
    # Four logs of a failed local run (study_*.log, each one line naming a missing script by its local path) were
    # committed at the repository root; *.log is ignored now, and this catches any other ignored file that slips in.
    assert _git("ls-files", "--cached", "--ignored", "--exclude-standard") == []


def _requirements(name: str) -> set[str]:
    """Distribution names a requirements file lists itself (not through ``-r``), lower-cased."""
    names = set()
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0].lower())
    return names


#: Where each library that a test skips without comes from: the requirements file that lists it, or the dependency
#: that brings it. A test that skips on a library no CI job installs is a test that never runs in CI.
_SKIPPED_WITHOUT = {
    "brotli": ("brotli", "requirements-dev.txt"),
    "fontTools.ttLib": ("fonttools", "requirements-dev.txt"),
    "hypothesis": ("hypothesis", "requirements-dev.txt"),
    "mpmath": ("mpmath", "requirements-dev.txt"),
    "QuantLib": ("quantlib", "requirements-oracle.txt"),
    "py_lets_be_rational": ("lets-be-rational", "requirements-oracle.txt"),
    "xlsxwriter": ("xlsxwriter", "requirements.txt"),
    "requests": ("streamlit", "requirements.txt"),  # a dependency of Streamlit's
    "playwright.sync_api": None,  # the browser tests: installed by hand (CONTRIBUTING.md)
    "tomllib": None,  # the standard library from Python 3.11
}


def test_every_library_a_test_skips_without_is_installed_by_a_requirements_file():
    # The five font-metric tests skipped in every CI job because fontTools and brotli were in no requirements file.
    # tests/test_harness.py names missing libraries on purpose, to check the skip guards.
    pattern = re.compile(r"""importorskip\(\s*["']([\w.]+)["']""")
    skipped = {
        name
        for path in sorted((ROOT / "tests").glob("**/*.py"))
        if path.name != "test_harness.py"
        for name in pattern.findall(path.read_text(encoding="utf-8"))
    }
    assert skipped <= set(_SKIPPED_WITHOUT), f"say where {sorted(skipped - set(_SKIPPED_WITHOUT))} comes from"
    for module, source in _SKIPPED_WITHOUT.items():
        if source is not None:
            distribution, requirements = source
            assert distribution in _requirements(requirements), f"{module}: {requirements} lacks {distribution}"


def _ci_job(name: str) -> str:
    """The text of one job in .github/workflows/ci.yml, up to the next job."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    match = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", workflow, flags=re.MULTILINE | re.DOTALL)
    assert match, f"ci.yml has no job {name}"
    return match.group(1)


def test_the_oldest_dependency_job_runs_every_declared_lower_bound_on_the_oldest_python():
    # The job ran on Python 3.11 although the project declares 3.10, so the real floor was never tested.
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    job = _ci_job("oldest-dependencies")
    floor = project["requires-python"].removeprefix(">=")
    assert re.search(rf'python-version: "{re.escape(floor)}"', job), f"the job should run on Python {floor}"
    pinned = dict(re.findall(r'"([\w-]+)==([\d.]+)\.\*"', job))
    for requirement in [*project["dependencies"], *project["optional-dependencies"]["app"]]:
        if ";" in requirement:  # tzdata, on Windows only
            continue
        name, lower = re.match(r"([\w-]+)>=([\d.]+)", requirement).groups()
        assert pinned.get(name) == lower, f"{name}: declared >={lower}, the job installs {pinned.get(name)}"


#: What the numerical core (everything outside options_analyzer/ui) may import besides the standard library: its
#: declared dependencies, and dateutil, which pandas requires (2.1 to 3.x) and so comes with every install.
_CORE_IMPORTS = frozenset({"numpy", "scipy", "pandas", "dateutil", "options_analyzer"})


def test_the_numerical_core_imports_only_its_declared_dependencies():
    # `pip install .` installs NumPy, SciPy and pandas only; one stray `import plotly` in a core module would have
    # shipped unnoticed, because CI installed only the wheel with the app extra. An import inside a function counts
    # too: it fails as soon as the function runs. (CI's package job also imports every core module from the bare
    # wheel and runs the core tests there.)
    stdlib = set(sys.stdlib_module_names)
    package = ROOT / "options_analyzer"
    offences = []
    for path in sorted(package.rglob("*.py")):
        if path.relative_to(package).parts[0] == "ui":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            offences += [
                f"{path.relative_to(ROOT)}:{node.lineno} imports {name}"
                for name in names
                if name.split(".")[0] not in stdlib | _CORE_IMPORTS
            ]
    assert offences == []


def test_the_package_is_marked_typed_and_ships_the_marker():
    # Without py.typed (PEP 561), mypy skipped the installed package ("missing library stubs or py.typed marker")
    # although most of its functions are annotated. CI's wheel check then confirms the marker is in the wheel.
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    assert (ROOT / "options_analyzer" / "py.typed").is_file()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "py.typed" in config["tool"]["setuptools"]["package-data"]["options_analyzer"]
