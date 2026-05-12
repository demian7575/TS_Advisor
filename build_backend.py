"""Tiny local PEP 517/660 backend for offline editable installs.

It intentionally declares no build requirements so `pip install -e .` works in
restricted environments. Runtime dependencies are documented in requirements.txt.
"""
import base64
import hashlib
import os
import pathlib
import zipfile

NAME = "ts_advisor"
VERSION = "0.1.0"
DIST = f"{NAME}-{VERSION}.dist-info"


def _metadata():
    return "\n".join([
        "Metadata-Version: 2.1",
        "Name: ts-advisor",
        f"Version: {VERSION}",
        "Summary: Behavior-preserving refactor of the TS Advisor notebook.",
        "Requires-Python: >=3.8",
        "",
    ])


def _wheel():
    return "\n".join([
        "Wheel-Version: 1.0",
        "Generator: local-ts-advisor-backend",
        "Root-Is-Purelib: true",
        "Tag: py3-none-any",
        "",
    ])


def _hash(data):
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    dist = pathlib.Path(metadata_directory) / DIST
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(_metadata())
    (dist / "WHEEL").write_text(_wheel())
    (dist / "RECORD").write_text("")
    return DIST


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_pth_wheel(wheel_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_pth_wheel(wheel_directory)


def _build_pth_wheel(wheel_directory):
    root = pathlib.Path(__file__).resolve().parent
    pth_name = f"{NAME}.pth"
    wheel_name = f"{NAME}-{VERSION}-py3-none-any.whl"
    wheel_path = pathlib.Path(wheel_directory) / wheel_name
    files = {
        pth_name: str(root / "src") + os.linesep,
        f"{DIST}/METADATA": _metadata(),
        f"{DIST}/WHEEL": _wheel(),
    }
    record_rows = []
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, text in files.items():
            data = text.encode()
            zf.writestr(path, data)
            record_rows.append(f"{path},{_hash(data)},{len(data)}")
        record_path = f"{DIST}/RECORD"
        record_rows.append(f"{record_path},,")
        zf.writestr(record_path, "\n".join(record_rows) + "\n")
    return wheel_name
