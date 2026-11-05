from importlib.metadata import version

import nd_arrow_array as nda


def test_import_exposes_error():
    assert issubclass(nda.NdFormatError, ValueError)
    assert nda.__version__ == version("nd-arrow-array")
