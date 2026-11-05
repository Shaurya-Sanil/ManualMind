"""
ManualMind Pattern Extraction Module (M3)
Implements precise regular expressions to extract domain patterns:
- Error Codes (e.g. E-101, E-204)
- Model & Serial Numbers (e.g. AX3000-PRO, SN: AUR2026X991)
- Firmware Versions (e.g. v1.4.0, v3.2.0)
- URLs & IP Addresses (e.g. http://192.168.1.1, https://aurora-tech.com/support)
- Email Addresses (e.g. support@aurora-tech.com)
- Phone Numbers (e.g. +1-800-555-0199)
- Dates (e.g. November 2026, 2026-11-05)
- Measurements with Units (e.g. 5 GHz, 12 V, 2402 Mbps, 85°C)
- GUI Navigation Menu Paths (e.g. Settings > Wireless > Guest Network)
- MAC Addresses (e.g. 00:1A:2B:3C:4D:5E)

These extracted patterns boost exact retrieval and populate the entity layer.
"""

import re
from typing import Dict, List, Pattern, Any


# ---------------------------------------------------------------------------
# Compiled Regular Expression Patterns
# ---------------------------------------------------------------------------

PATTERNS: Dict[str, Pattern] = {
    "ERROR_CODE": re.compile(
        r"\b(?:E|ERR)[-_]\d{3,4}\b",
        re.IGNORECASE
    ),
    "MODEL_NUMBER": re.compile(
        r"\b(?:AX3000(?:-PRO)?|AX[0-9]{4}(?:-[A-Z0-9]+)?)\b"
    ),
    "SERIAL_NUMBER": re.compile(
        r"\b(?:SN[:\s-]*[A-Z0-9]{8,16})\b",
        re.IGNORECASE
    ),
    "FIRMWARE_VERSION": re.compile(
        r"\bv?\d+\.\d+(?:\.\d+)?(?:[-_]build\d+)?\b",
        re.IGNORECASE
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    "URL": re.compile(
        r"https?://(?:[a-zA-Z0-9_\-\.]+(?::[0-9]+)?)(?:/[a-zA-Z0-9_\-\./\?%&=#]*)?",
        re.IGNORECASE
    ),
    "EMAIL": re.compile(
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "DATE": re.compile(
        r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",
        re.IGNORECASE
    ),
    "MEASUREMENT": re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:GHz|MHz|Mbps|Gbps|Kbps|V|A|mA|W|mW|dBm|dBi|GB|MB|KB|kg|g|lbs|cm|inches|°C|°F|ms|seconds|minutes)\b",
        re.IGNORECASE
    ),
    "MENU_PATH": re.compile(
        r"\b[A-Za-z0-9\s]+\s*(?:>|->|→)\s*[A-Za-z0-9\s]+(?:\s*(?:>|->|→)\s*[A-Za-z0-9\s]+)*"
    ),
    "MAC_ADDRESS": re.compile(
        r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b"
    )
}


def extract_all_patterns(text: str) -> Dict[str, List[str]]:
    """
    Extracts all recognized technical entities and patterns from the text.
    Returns a dictionary mapping pattern names to lists of matched strings.
    """
    matches: Dict[str, List[str]] = {}
    for name, pattern in PATTERNS.items():
        found = pattern.findall(text)
        # Clean menu path false matches
        if name == "MENU_PATH":
            clean_paths = []
            for m in found:
                cleaned = m.strip()
                if ">" in cleaned or "->" in cleaned or "→" in cleaned:
                    parts = [p.strip() for p in re.split(r">|->|→", cleaned) if p.strip()]
                    if len(parts) >= 2:
                        clean_paths.append(" > ".join(parts))
            found = clean_paths
            
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for item in found:
            norm_item = item.strip()
            if norm_item and norm_item not in seen:
                seen.add(norm_item)
                deduped.append(norm_item)
        matches[name] = deduped

    return matches


def find_error_codes(text: str) -> List[str]:
    """Convenience helper to extract error codes specifically."""
    return [c.upper().replace("_", "-") for c in PATTERNS["ERROR_CODE"].findall(text)]
