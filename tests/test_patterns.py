"""
Tests for Module M3: Regular Expression Pattern Extraction
Includes comprehensive pytest parameterized test tables of positive and negative examples.
"""

import pytest
from nlp.patterns import PATTERNS, extract_all_patterns, find_error_codes


# Parameterized table for ERROR_CODE
@pytest.mark.parametrize("text,expected", [
    ("Encountered error code E-101 on WAN port", ["E-101"]),
    ("Diagnostic code E-204 indicates DNS failure", ["E-204"]),
    ("Check ERR-501 in the log", ["ERR-501"]),
    ("Normal operating mode without errors", []),
    ("Letter E followed by dash without digits E-", []),
    ("Just numbers 101 or 204", []),
])
def test_error_code_regex(text, expected):
    matches = PATTERNS["ERROR_CODE"].findall(text)
    assert matches == expected


# Parameterized table for MODEL_NUMBER
@pytest.mark.parametrize("text,expected", [
    ("The Aurora AX3000-PRO router", ["AX3000-PRO"]),
    ("Supports AX3000 standard", ["AX3000"]),
    ("Device model AX5400-V2", ["AX5400-V2"]),
    ("Generic wireless router model", []),
])
def test_model_number_regex(text, expected):
    matches = PATTERNS["MODEL_NUMBER"].findall(text)
    assert matches == expected


# Parameterized table for IP_ADDRESS
@pytest.mark.parametrize("text,expected", [
    ("Default gateway is 192.168.1.1", ["192.168.1.1"]),
    ("DNS server at 8.8.8.8 and 1.1.1.1", ["8.8.8.8", "1.1.1.1"]),
    ("Local router IP is 10.0.0.1", ["10.0.0.1"]),
    ("Invalid IP 999.999.1.1 should not match", []),
    ("Version string v1.4.0 is not an IP", []),
])
def test_ip_address_regex(text, expected):
    matches = PATTERNS["IP_ADDRESS"].findall(text)
    assert matches == expected


# Parameterized table for MEASUREMENTS
@pytest.mark.parametrize("text,expected", [
    ("Operates on 5 GHz and 2.4 GHz bands", ["5 GHz", "2.4 GHz"]),
    ("Power supply requires 12 V and 2.0 A", ["12 V", "2.0 A"]),
    ("Max speed is 2402 Mbps on 160 MHz width", ["2402 Mbps", "160 MHz"]),
    ("Thermal trip threshold is 85°C", ["85°C"]),
    ("Flash memory is 256 MB", ["256 MB"]),
    ("Non-measurement text five gigahertz", []),
])
def test_measurements_regex(text, expected):
    matches = PATTERNS["MEASUREMENT"].findall(text)
    assert matches == expected


# Parameterized table for MENU_PATH
@pytest.mark.parametrize("text,expected_part", [
    ("Navigate to Settings > Wireless > Guest Network", "Settings > Wireless > Guest Network"),
    ("Go to Settings > System Tools > Administration", "Settings > System Tools > Administration"),
    ("Open Settings -> Network -> WAN", "Settings -> Network -> WAN"),
])
def test_menu_path_regex(text, expected_part):
    res = extract_all_patterns(text)
    paths = res["MENU_PATH"]
    assert len(paths) >= 1
    assert any("Settings" in p and ("Wireless" in p or "System Tools" in p or "Network" in p) for p in paths)


def test_contact_patterns():
    text = "Call +1-800-555-0199 or email support@aurora-tech.com for warranty claims"
    res = extract_all_patterns(text)
    assert "+1-800-555-0199" in res["PHONE_NUMBER"]
    assert "support@aurora-tech.com" in res["EMAIL"]


def test_find_error_codes_helper():
    assert find_error_codes("Error E-101 and E_204 detected") == ["E-101", "E-204"]
    assert find_error_codes("No error codes present") == []

