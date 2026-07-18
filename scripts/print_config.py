#!/usr/bin/env python
"""Print a fully-resolved config (resolves `extends:`). Used by Makefile before
any scientific run so nothing launches without showing its configuration."""
import sys
from _common import print_resolved_config  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: print_config.py <config.yaml>")
    print_resolved_config(sys.argv[1])
