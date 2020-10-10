"""Usage:
  spm12 [options]

Options:
  -c DIR, --cache DIR  : directory to use for installation [default: ~/.spm12].
  -s VER, --spm-version  : version [default: 12].
"""
import logging

from argopt import argopt

from .utils import ensure_spm


def main(argv=None):
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)s:%(funcName)s:%(message)s"
    )
    args = argopt(__doc__).parse_args(argv)
    ensure_spm(cache=args.cache, version=args.spm_version)
    print("SPM{v} is successfully installed".format(v=args.spm_version))
    return 0
