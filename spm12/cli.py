"""Usage:
  spm12 [options] [<command>...]

Options:
  -c DIR, --cache DIR  : directory to use for installation [default: ~/.spm12].
  -s VER, --spm-version  : version [default: 12].
  -r, --runtime  : use runtime (not full MATLAB).

Arguments:
  <command>  : Runtime command [default: quit]|gui|help|...
"""
import logging

from argopt import argopt

from .utils import ensure_spm, mcr_run

log = logging.getLogger(__name__)


def main(argv=None):
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(funcName)s:%(message)s")
    args = argopt(__doc__).parse_args(argv)
    log.info(args)
    if isinstance(args.command, str):
        args.command = [args.command]
    if len(args.command) == 1 and args.command[0] == "gui":
        args.command = []
    if args.runtime:
        log.debug(mcr_run(*args.command, cache=args.cache, version=args.spm_version))
    else:
        ensure_spm(cache=args.cache, version=args.spm_version)
    print("SPM{v} is successfully installed".format(v=args.spm_version))
    return 0
