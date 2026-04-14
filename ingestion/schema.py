from dataclasses import dataclass


@dataclass
class FunctionSample:
    code: str
    cwes: list[str]   # CWE IDs attributed to this function; empty for label=0
    label: int        # 1 = vulnerable, 0 = safe
