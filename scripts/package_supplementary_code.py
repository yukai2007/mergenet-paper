#!/usr/bin/env python3
"""Package the public implementation for anonymous supplementary submission."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "gnawyymmij", "yukai2007", "lisiyuan", "liziqing",
    "EXPERIMENTS.md", "DATA_HANDOFF.md", "share_packet",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT.parent / "MergeNet")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "mergenet-anonymous-code.zip")
    args = parser.parse_args()
    repo = args.repo.resolve()
    tracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z"]
    ).decode().split("\0")
    files: dict[str, bytes] = {}
    for name in tracked:
        if not name or name == ".gitignore":
            continue
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid archive path: {name}")
        if path.suffix in {".pth", ".pt", ".ckpt", ".pyc"}:
            raise ValueError(f"Unexpected artifact: {name}")
        content = (repo / path).read_bytes()
        text = content.decode("utf-8")
        for token in FORBIDDEN:
            if token.lower() in text.lower():
                raise ValueError(f"Identifying or internal reference in {name}: {token}")
        if path.suffix == ".py":
            compile(text, name, "exec")
        if name == "README.md":
            text = text.replace("# MergeNet\n", "# MergeNet — Anonymous Supplementary Code\n", 1)
            text += (
                "\n## Supplementary package\n\n"
                "This package contains the implementation, ImageNet-1K training "
                "configuration, launcher, and implementation checks. It does not "
                "contain datasets or pretrained checkpoints. The inference example "
                "uses random weights.\n\n"
                "Git history and repository-owner information are excluded. "
                "Third-party license notices and dependency URLs are retained; "
                "they identify upstream software, not the submission authors. "
                "See NOTICE.md. SHA256SUMS records the included files.\n"
            )
            content = text.encode("utf-8")
        files[name] = content

    files["NOTICE.md"] = (
        "# Third-party attribution\n\n"
        "This implementation builds on OpenToMe, ToMe, DTEM, timm, and "
        "FlashAttention. Original third-party copyright notices and references "
        "are retained in the source and LICENSE. These attributions describe "
        "upstream software and do not identify this submission's authors.\n"
    ).encode()
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(files.items())
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            item = zipfile.ZipInfo(f"mergenet_code/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.create_system = 3
            item.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(item, content)
    print(f"Wrote {args.output}: {len(files)} files, {args.output.stat().st_size:,} bytes")
    print("Python sources parse; screened owner handles and internal references are absent.")


if __name__ == "__main__":
    main()
