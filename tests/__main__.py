import logging

from miutil.fdio import extractall
from miutil.web import urlopen_cached

from .conftest import HOME

log = logging.getLogger(__name__)
DATA_URL = (
    "https://zenodo.org/record/3877529/files/amyloidPET_FBP_TP0_extra.zip?download=1"
)


def main():
    log.info(f"Downloading {DATA_URL}\nto ${{DATA_ROOT:-~}} ({HOME})")
    with urlopen_cached(DATA_URL, HOME) as fd:
        extractall(fd, HOME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
