#!/usr/bin/env python3
"""Reject missing, duplicate, failed, unexpectedly skipped or unlisted XCTest cases."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("listing", type=Path)
parser.add_argument("report", type=Path)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
expected = {line.strip() for line in args.listing.read_text().splitlines()
            if line.startswith("WorkbenchTests.") and "Hardware" not in line and "/test" in line}
metal = "WorkbenchTests.TransformerLifetimeTests/testCompiledWeightsAreReleasedAcrossPromptShapes"
assert metal in expected, "Metal regression missing from discovery"
if args.cpu: expected.remove(metal)
assert expected, "Test listing is empty"
root = ET.parse(args.report).getroot()
cases = root.findall(".//testcase")
actual = {case.attrib["classname"] + "/" + case.attrib["name"] for case in cases}
assert len(cases) == len(actual), "Duplicate cases"
assert actual == expected, f"Coverage mismatch: missing={expected-actual}, extra={actual-expected}"
assert not root.findall(".//failure") and not root.findall(".//error"), "Test failures or errors"
skipped = {case.attrib["classname"] + "/" + case.attrib["name"] for case in cases if case.find("skipped") is not None}
allowed = set()
assert skipped == allowed, f"Unexpected skips: {skipped ^ allowed}"
print(json.dumps({"discovered": len(expected), "passed": len(cases)-len(skipped), "skipped": sorted(skipped), "notCovered": [metal] if args.cpu else []}))
