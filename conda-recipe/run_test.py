
import subprocess
import os
import sys
import shutil
import re

src_dir = os.getenv('SRC_DIR')

def coverage_report():
    fn = '.coverage'
    assert os.path.exists(fn)
    build_dir = os.getenv('TRAVIS_BUILD_DIR')
    dest = os.path.join(build_dir, fn)
    print( "copying coverage report to", dest)
    shutil.copy(fn, dest)
    assert os.path.exists(dest)

    # fix paths in .coverage file
    with open(dest, 'r') as fh:
        data = fh.read()
    match= '"/home/travis/miniconda/envs/_test/lib/python.+?/site-packages/.+?/(thermotools/.+?)"'
    repl = '"%s/\\1"' % build_dir
    data = re.sub(match, repl, data)
    os.unlink(dest)
    with open(dest, 'w+') as fh:
       fh.write(data)

nose_run = [
    "nosetests",
    os.path.join(src_dir, "test"),
    "-vv",
    "--with-coverage",
    "--cover-inclusive",
    "--cover-package=thermotools",
    "--with-doctest",
    "--doctest-options=+NORMALIZE_WHITESPACE,+ELLIPSIS"]

res = subprocess.call(nose_run)

# move .coverage file to git clone on Travis CI
if os.getenv('TRAVIS', False):
    coverage_report()

sys.exit(res)
