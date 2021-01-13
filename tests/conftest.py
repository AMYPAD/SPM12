from os import getenv
from textwrap import dedent

from miutil.fdio import Path
from pytest import fixture, skip

HOME = Path(getenv("DATA_ROOT", "~")).expanduser()


@fixture(scope="session")
def folder_in():
    Ab_PET_mMR_test = HOME / "Ab_PET_mMR_test"
    if not Ab_PET_mMR_test.is_dir():
        skip(
            dedent(
                """\
            Cannot find Ab_PET_mMR_test in ${DATA_ROOT:-~} (%s).
            Try running `python -m tests` to download it.
            """
            )
            % HOME
        )
    return Ab_PET_mMR_test


@fixture(scope="session")
def folder_ref(folder_in):
    Ab_PET_mMR_ref = folder_in / "testing_reference" / "Ab_PET_mMR_ref"
    if not Ab_PET_mMR_ref.is_dir():
        skip(
            dedent(
                """\
            Cannot find Ab_PET_mMR_ref in
            ${DATA_ROOT:-~}/testing_reference (%s/testing_reference).
            Try running `python -m tests` to download it.
            """
            )
            % HOME
        )
    return Ab_PET_mMR_ref


@fixture
def PET(folder_ref):
    res = folder_ref / "basic" / "17598013_t-3000-3600sec_itr-4_suvr.nii.gz"
    assert res.is_file()
    return res


@fixture
def MRI(folder_in):
    res = folder_in / "T1w_N4" / "t1_S00113_17598013_N4bias_cut.nii.gz"
    assert res.is_file()
    return res
