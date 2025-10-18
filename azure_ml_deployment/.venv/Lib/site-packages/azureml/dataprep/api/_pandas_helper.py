from typing import Tuple, List
from .errorhandlers import NumpyImportError, PandasImportError


_import_tried = False
_have_pandas = True
_have_numpy = True
_have_pyarrow = True
_pyarrow_supports_cdata = True
_pyarrow_supports_promote_options = True # available with latest stable version (14.0.0)

def _try_import():
    global _import_tried
    global _have_pandas
    global _have_numpy
    global _have_pyarrow
    global _pyarrow_supports_cdata
    global _pyarrow_supports_promote_options

    if _import_tried:
        return

    try:
        import pandas
    except:
        _have_pandas = False

    try:
        import numpy
    except:
        _have_numpy = False

    try:
        import pyarrow
        pyarrow_version_components = pyarrow.__version__.split('.')
        pyarrow_major = int(pyarrow_version_components[0])
        pyarrow_minor = int(pyarrow_version_components[1])

        _pyarrow_supports_cdata = hasattr(pyarrow.RecordBatch, '_import_from_c')

        if pyarrow_major <= 0 and pyarrow_minor < 16:
            _have_pyarrow = False

        if pyarrow_major < 14:
            _pyarrow_supports_promote_options = False

    except:
        _have_pyarrow = False
        _pyarrow_supports_cdata = False

    _import_tried = True

def have_numpy() -> bool:
    _try_import()
    global _have_numpy
    return _have_numpy

def have_pandas() -> bool:
    _try_import()
    global _have_pandas
    return _have_pandas

def have_pyarrow() -> bool:
    _try_import()
    global _have_pyarrow
    return _have_pyarrow

def pyarrow_supports_cdata() -> bool:
    _try_import()
    global _pyarrow_supports_cdata
    return _pyarrow_supports_cdata

def pyarrow_supports_promote_options() -> bool:
    _try_import()
    global _pyarrow_supports_promote_options
    return _pyarrow_supports_promote_options

def _ensure_numpy_pandas():
    _try_import()
    if not have_pandas():
        raise PandasImportError()
    if not have_numpy():
        raise NumpyImportError()

def _sanitize_df_for_native(df):
    import pandas as pd
    import numpy as np
    # Ensure column schema is one dimensional and strings.
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex by getting higherarchy as tuples then joining each tuple to a _ seperated string.
        df.columns = ['_'.join(t) for t in df.columns.values]
    else:
        # Cast any non-string index values to be strings.
        df.columns = df.columns.astype(str)
    # Handle Categorical typed columns. Categorical is a pandas type not a numpy type and azureml-dataprep-native can't
    # handle it. This is temporary pending improvments to native that can handle Categoricals, vso: 246011
    new_schema = df.columns.tolist()
    new_values = []
    for column_name in new_schema:
        if pd.api.types.is_categorical_dtype(df[column_name]):
            new_values.append(np.asarray(df[column_name]))
        else:
            new_values.append(df[column_name].values)
    return (new_schema, new_values)


def ensure_df_native_compat(df: 'pandas.DataFrame') -> Tuple[List[str], List]:
    _ensure_numpy_pandas()
    return _sanitize_df_for_native(df)
