from .errorhandlers import ImportError


class DaskImportError(ImportError):
    """
    Exception raised when dask was not able to be imported.
    """

    def __init__(self):
        super().__init__('dask')


_have_dask = True
# noinspection PyBroadException
try:
    import dask
except:
    _have_dask = False


def have_dask() -> bool:
    global _have_dask
    return _have_dask


def _ensure_dask():
    if not have_dask():
        raise DaskImportError()
