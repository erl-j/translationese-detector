#!/usr/bin/env python3
"""Stream-extract only txt/<lang>/*.txt from the Europarl v7 source tarball.

The tarball is untrusted input: we never use tarfile.extract(); we only read
regular-file members whose names match txt/<lang>/<safe-name>.txt and write
their bytes to <out>/<lang>/<safe-name>.txt ourselves.

Usage (run with -I; the build script does this for you):
    python3 -I data/extract_europarl.py data/raw/europarl/europarl.tgz \
        data/raw/europarl/txt sv en
"""
import os
import re
import sys
import tarfile

SAFE = re.compile(r"^txt/([a-z]{2})/(ep-[0-9A-Za-z_.-]+\.txt)$")


def main(argv):
    if len(argv) < 4:
        sys.exit(__doc__)
    tgz, out = argv[1], argv[2]
    langs = set(argv[3:])
    counts = {l: 0 for l in langs}
    for l in langs:
        os.makedirs(os.path.join(out, l), exist_ok=True)
    with tarfile.open(tgz, mode="r|gz") as tf:  # streaming, no random access
        for m in tf:
            if not m.isreg():
                continue
            mm = SAFE.match(m.name)
            if not mm or mm.group(1) not in langs:
                continue
            f = tf.extractfile(m)
            data = f.read()
            dst = os.path.join(out, mm.group(1), mm.group(2))
            with open(dst, "wb") as g:
                g.write(data)
            counts[mm.group(1)] += 1
    for l in sorted(counts):
        print("extracted %s: %d files" % (l, counts[l]))


if __name__ == "__main__":
    main(sys.argv)
