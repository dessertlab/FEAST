import re
from pathlib import Path

from ingestion.schema import FunctionSample

_CWE_DIR_RE = re.compile(r"CWE[-_]?(\d+)", re.IGNORECASE)


def _cwe_from_path(path: Path) -> str | None:
    for part in path.parts:
        m = _CWE_DIR_RE.search(part)
        if m:
            return f"CWE-{int(m.group(1))}"
    return None


def extract_llmseceval(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from LLMSecEval directory structure.

    Sources:
      - Vulnerable: Zenodo record 5225651 (copilot-cwe-scenarios-dataset)
      - Safe: GitHub tuhh-softsec/LLMSecEval (Dataset/Secure Code Samples)

    Structure (after download notebook reorganisation):
      data_path/
        CWE-NNN/
          <vulnerable_file>.py   -- label=1
          Secure/
            <safe_file>.py       -- label=0

    Python only (.py files).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"LLMSecEval directory not found: {data_path}")

    samples: list[FunctionSample] = []

    for cwe_dir in data_path.iterdir():
        if not cwe_dir.is_dir():
            continue
        cwe = _cwe_from_path(cwe_dir)
        if not cwe:
            continue

        secure_dir = cwe_dir / "Secure"

        for fpath in cwe_dir.iterdir():
            if fpath.is_dir():
                continue
            if fpath.suffix.lower() != ".py":
                continue
            code = fpath.read_text(encoding="utf-8", errors="replace").strip()
            if not code:
                continue
            samples.append(FunctionSample(
                code=code, cwes=[cwe], label=1,
                branch="ai", language="Python",
            ))

        if secure_dir.exists():
            for fpath in secure_dir.iterdir():
                if fpath.suffix.lower() != ".py":
                    continue
                code = fpath.read_text(encoding="utf-8", errors="replace").strip()
                if not code:
                    continue
                samples.append(FunctionSample(
                    code=code, cwes=[], label=0,
                    branch="ai", language="Python",
                ))

    return samples
