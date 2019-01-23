
import subprocess
import sys

nose_run = [
    "nosetests",
    "test",
    "-vv",
    "--with-coverage",
    "--cover-inclusive",
    "--cover-package=thermotools",
    "--with-doctest",
    "--doctest-options=+NORMALIZE_WHITESPACE,+ELLIPSIS"]

res = subprocess.call(nose_run)

sys.exit(res)
