import json
import tomllib
from pathlib import Path

from policyengine_us_data.utils import policyengine as policyengine_utils


class _FakeDistribution:
    def __init__(self, direct_url: dict | None):
        self.direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        if name != "direct_url.json" or self.direct_url is None:
            return None
        return json.dumps(self.direct_url)


def _patch_distribution(monkeypatch, direct_url: dict | None):
    policyengine_utils.get_policyengine_us_build_info.cache_clear()
    monkeypatch.setattr(
        policyengine_utils.metadata,
        "version",
        lambda package_name: "1.0.0",
    )
    monkeypatch.setattr(
        policyengine_utils.metadata,
        "distribution",
        lambda package_name: _FakeDistribution(direct_url),
    )
    monkeypatch.setattr(
        policyengine_utils,
        "get_locked_dependency_version",
        lambda package_name: "1.0.0",
    )


def test_policyengine_us_build_info_does_not_infer_git_commit_from_site_packages(
    monkeypatch,
):
    _patch_distribution(monkeypatch, direct_url=None)
    monkeypatch.setattr(
        policyengine_utils,
        "_get_git_commit",
        lambda path: (_ for _ in ()).throw(AssertionError("unexpected git lookup")),
    )

    build_info = policyengine_utils.get_policyengine_us_build_info()

    assert build_info.git_commit is None
    assert build_info.source_path.endswith("policyengine_us")


def test_policyengine_us_build_info_reads_vcs_commit_from_direct_url(monkeypatch):
    _patch_distribution(
        monkeypatch,
        direct_url={
            "url": "https://github.com/PolicyEngine/policyengine-us",
            "vcs_info": {
                "vcs": "git",
                "commit_id": "deadbeef",
            },
        },
    )
    monkeypatch.setattr(
        policyengine_utils,
        "_get_git_commit",
        lambda path: (_ for _ in ()).throw(AssertionError("unexpected git lookup")),
    )

    build_info = policyengine_utils.get_policyengine_us_build_info()

    assert build_info.git_commit == "deadbeef"
    assert build_info.source_path == "https://github.com/PolicyEngine/policyengine-us"


def test_policyengine_us_build_info_reads_git_commit_from_local_direct_url(
    monkeypatch,
    tmp_path,
):
    source_path = tmp_path / "policyengine-us"
    source_path.mkdir()
    _patch_distribution(
        monkeypatch,
        direct_url={
            "url": f"file://{source_path}",
            "dir_info": {"editable": True},
        },
    )
    monkeypatch.setattr(
        policyengine_utils,
        "_get_git_commit",
        lambda path: "abc123" if path == source_path else None,
    )

    build_info = policyengine_utils.get_policyengine_us_build_info()

    assert build_info.git_commit == "abc123"
    assert build_info.source_path == str(source_path)


def test_policyengine_us_dependency_floor_matches_lockfile():
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    lock_data = tomllib.loads((repo_root / "uv.lock").read_text())

    dependency = next(
        dep
        for dep in pyproject["project"]["dependencies"]
        if dep.startswith("policyengine-us>=")
    )
    declared_floor = dependency.split(">=", 1)[1]
    locked_version = next(
        package["version"]
        for package in lock_data["package"]
        if package["name"] == "policyengine-us"
    )

    assert declared_floor == locked_version
