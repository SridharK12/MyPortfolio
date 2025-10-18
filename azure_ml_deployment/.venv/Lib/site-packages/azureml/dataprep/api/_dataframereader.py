import os
import glob
import warnings
from shutil import rmtree
from threading import RLock
from typing import List, Tuple
from uuid import uuid4
from ._pandas_helper import (
    have_pandas,
    have_pyarrow,
    pyarrow_supports_cdata,
    pyarrow_supports_promote_options
)
from .errorhandlers import (
    DataPrepException,
    UnexpectedError,
    StorageAccountLimit,
    LocalDiskFull,
    PandasImportError
)
from ._rslex_executor import get_rslex_executor
from ._loggerfactory import _LoggerFactory, trace
from azureml.dataprep.rslex import PyRsDataflow
from ._loggerfactory import _log_dataflow_execution_activity as _og_log_dataflow_execution_activity

_LOGGER = _LoggerFactory.get_logger("DataframeReader")
_TRACER = trace.get_tracer(__name__)


# 20,000 rows gives a good balance between memory requirement and throughput by requiring that only
# (20000 * CPU_CORES) rows are materialized at once while giving each core a sufficient amount of
# work.
PARTITION_SIZE = 20000


class _InconsistentSchemaError(Exception):
    def __init__(self, reason: str):
        super().__init__(
            "Inconsistent or mixed schemas detected across partitions: " + reason
        )


def _log_dataflow_execution_activity(activity, rslex_failed, rslex_error, execution_succeeded, preppy):
    _og_log_dataflow_execution_activity(_LOGGER, activity, rslex_failed, rslex_error, execution_succeeded, preppy)


def _collect_preppy_files(preppy_dirc):
    return list(sorted(map(str, glob.glob(os.path.join(preppy_dirc, "part-*")))))


def _preppy_files_to_pandas_dataframe(preppy_dirc, extended_types, nulls_as_nan):
    import pandas
    from azureml.dataprep.native import preppy_to_ndarrays
    from collections import OrderedDict

    intermediate_files = _collect_preppy_files(preppy_dirc)
    dataset = preppy_to_ndarrays(intermediate_files, extended_types, nulls_as_nan)
    return pandas.DataFrame.from_dict(OrderedDict(dataset))


def _preppy_files_to_pyrecords(preppy_dirc):
    from azureml.dataprep.native import preppy_to_pyrecords
    intermediate_files = _collect_preppy_files(preppy_dirc)
    dataset = preppy_to_pyrecords(intermediate_files)
    return [{k: pyrec[k] for k in pyrec} for pyrec in dataset]


def _write_preppy(
    activity,
    dataflow,
    span_context=None,
    telemetry_dict=None,
):
    import tempfile
    from pathlib import Path
    import os

    subfolder = f"{uuid4()}_{os.getpid()}"
    intermediate_path = Path(os.path.join(tempfile.gettempdir(), subfolder))

    if isinstance(dataflow, str):
        rs_dataflow = PyRsDataflow(dataflow)
        convert = False
    else:
        rs_dataflow = dataflow
        convert = True

    rs_dataflow = rs_dataflow.add_transformation(
            "write_files",
            {
                "writer": "preppy",
                "destination": {
                    "directory": str(intermediate_path),
                    "handler": "Local",
                },
                "writer_arguments": {
                    "profiling_fields": ["Kinds", "MissingAndEmpty"]
                },
                "existing_file_handling": "replace",
            },
        )

    dataflow = str(rs_dataflow.to_yaml_string()) if convert else rs_dataflow

    def cleanup():
        try:
            rmtree(intermediate_path, ignore_errors=True)
        except BaseException:
            pass  # ignore exception

    try:
        _execute(
            activity,
            dataflow,
            span_context=span_context,
            cleanup=cleanup,
            telemetry_dict=telemetry_dict,
        )
    except Exception as e:
        cleanup()
        raise e

    if not (intermediate_path / '_SUCCESS').exists():
        error = "Missing _SUCCESS sentinel in preppy folder."
        _LoggerFactory.trace_error(_LOGGER, error)
        raise UnexpectedError(error)

    return str(intermediate_path)


def to_pyrecords_with_preppy(activity, dataflow):
    return _execute(activity=activity,
                    dataflow=dataflow,
                    force_preppy=True,
                    convert_preppy_to_pyrecords=True)


def _execute(
    activity,
    dataflow,
    is_to_pandas_dataframe=False,
    force_clex=False,
    allow_fallback_to_clex=False,
    force_preppy=False,
    collect_results=False,
    fail_on_error=False,
    fail_on_mixed_types=False,
    fail_on_out_of_range_datetime=False,
    partition_ids=None,
    traceparent="",
    span_context=None,
    cleanup=None,
    extended_types: bool = False,
    nulls_as_nan: bool = True,
    telemetry_dict=None,
    convert_preppy_to_pyrecords=False,
):
    if force_clex:
        print('force_clex=True passed into _execute. Clex engine has been deprecated, defaulting to RSlex engine')
        _LoggerFactory.trace_warn(_LOGGER, 'force_clex=True passed into _execute')

    if allow_fallback_to_clex:
        print('allow_fallback_to_clex=True passed into _execute. Clex engine has been deprecated, defaulting to RSlex engine')
        _LoggerFactory.trace_warn(_LOGGER, 'allow_fallback_to_clex=True passed into _execute')

    execution_succeeded = False,
    rslex_failed = None
    rslex_error = None
    preppy_telemetry = None
    dataframe_reader = get_dataframe_reader()

    def rslex_execute():
        nonlocal execution_succeeded
        nonlocal rslex_failed

        executor = get_rslex_executor()
        (batches, num_partitions, stream_columns) = executor.execute_dataflow(dataflow,
                                                                              collect_results,
                                                                              fail_on_error,
                                                                              fail_on_mixed_types,
                                                                              fail_on_out_of_range_datetime,
                                                                              traceparent,
                                                                              partition_ids)

        if is_to_pandas_dataframe:
            result = dataframe_reader.process_rslex_batches(batches, stream_columns)
        else:
            result = (batches, num_partitions, stream_columns)

        rslex_failed = False
        execution_succeeded = True
        return result

    def preppy_execution():
        if not have_pyarrow():
            warnings.warn(
                "Please install pyarrow>=0.16.0 for improved performance of to_pandas_dataframe. "
                "You can ensure the correct version is installed by running: pip install "
                "pyarrow>=0.16.0 --upgrade")
        inner_telemetry_dict = {}
        try:
            # force rslex execution
            preppy_dirc = _write_preppy(
                activity="_DataframeReader.preppy_execution",
                dataflow=dataflow,
                span_context=span_context,
                telemetry_dict=inner_telemetry_dict,
            )

            if convert_preppy_to_pyrecords:
                records = _preppy_files_to_pyrecords(preppy_dirc)
                inner_telemetry_dict["to_pyrecords_failed"] = False
            else:
                records = _preppy_files_to_pandas_dataframe(preppy_dirc, extended_types, nulls_as_nan)
                inner_telemetry_dict["to_ndarray_failed"] = False
            return records, inner_telemetry_dict

        except Exception as ex:
            error = "Error from preppy_execution: {}".format(repr(ex))
            _LoggerFactory.trace_error(_LOGGER, error)

            if convert_preppy_to_pyrecords:
                inner_telemetry_dict["to_pyrecords_failed"] = True
                inner_telemetry_dict["to_pyrecords_error"] = repr(ex)
            elif "rslex_error" not in inner_telemetry_dict:
                # if inner_telemetry_dict does not have rslex_error, we failed while converting to ndarrays
                inner_telemetry_dict["to_ndarray_failed"] = True
                inner_telemetry_dict["to_ndarray_error"] = repr(ex)

            if "DestinationFull" in str(ex):
                return ex if isinstance(ex, DataPrepException) else LocalDiskFull(str(ex)), inner_telemetry_dict

            return ex if isinstance(ex, DataPrepException) else UnexpectedError(ex), inner_telemetry_dict

    try:
        if force_preppy:
            df_or_err, preppy_telemetry = preppy_execution()
            if isinstance(df_or_err, Exception):
                execution_succeeded = False
                raise df_or_err
            else:
                execution_succeeded = True
                return df_or_err

        try:
            return rslex_execute()
        except _InconsistentSchemaError as e:
            rslex_failed = True
            rslex_error = e
            reason = e.args[0]
            warnings.warn("Using alternate reader. " + reason)
            df_or_err, preppy_telemetry = preppy_execution()
            if isinstance(df_or_err, Exception):
                execution_succeeded = False
                raise df_or_err
            else:
                execution_succeeded = True
                return df_or_err
        except DataPrepException as e:
            if "is over the account limit" in str(rslex_error):
                execution_succeeded = False
                raise StorageAccountLimit(str(rslex_error))

            execution_succeeded = False
            _LoggerFactory.trace_warn(_LOGGER, "rslex failed")
            raise e
        except Exception as e:
            _LoggerFactory.trace_warn(_LOGGER, "rslex failed")
            execution_succeeded = False
            raise UnexpectedError(e)

    finally:
        if telemetry_dict is not None:
            if rslex_failed is not None:
                telemetry_dict["rslex_failed"] = rslex_failed
            if rslex_error is not None:
                telemetry_dict["rslex_error"] = repr(rslex_error)
            if execution_succeeded is not None:
                telemetry_dict["execution_succeeded"] = execution_succeeded
        else:
            _log_dataflow_execution_activity(activity,
                                             rslex_failed,
                                             rslex_error,
                                             execution_succeeded,
                                             preppy_telemetry)

    return None


def get_partition_count_with_rslex(dataflow, span_context=None):
    _, partition_count, _ = _execute('_DataframeReader.get_partition_count_with_rslex', dataflow,
                                     span_context=span_context, collect_results=False)
    return partition_count


def get_partition_info_with_fallback(dataflow, span_context=None) -> Tuple[int, List[Tuple['StreamInfo', int]]]:
    execution_succeeded = False
    rslex_failed = None
    rslex_error = None

    def rslex_execute():
        nonlocal rslex_failed
        nonlocal rslex_error

        try:
            (num_partitions, partitions_streams_and_counts) = get_rslex_executor().get_partition_info(dataflow, '')
            return (num_partitions, partitions_streams_and_counts)
        except BaseException as ex:
            rslex_failed = True
            rslex_error = ex
            raise ex

    try:
        try:
            return rslex_execute()
        except Exception as e:
            execution_succeeded = False

            if "is over the account limit" in str(rslex_error):
                raise StorageAccountLimit(str(rslex_error))

            _LoggerFactory.trace_warn(_LOGGER, "rslex failed")
            rslex_error = e
            raise e

    finally:
        _log_dataflow_execution_activity(activity='_DataframeReader.get_partition_info',
                                         rslex_failed=rslex_failed,
                                         rslex_error=rslex_error,
                                         execution_succeeded=execution_succeeded,
                                         preppy=None)


# noinspection PyProtectedMember,PyPackageRequirements
class _DataFrameReader:
    def __init__(self):
        self._outgoing_dataframes = {}
        self._incoming_dataframes = {}
        self._iterators = {}
        _LoggerFactory.trace(_LOGGER, "DataframeReader_create")

    def to_pandas_dataframe(
        self,
        dataflow,
        extended_types: bool = False,
        nulls_as_nan: bool = True,
        on_error: str = "null",
        out_of_range_datetime: str = "null",
        span_context: "DPrepSpanContext" = None,
    ) -> "pandas.DataFrame":
        if not have_pandas():
            raise PandasImportError()

        force_preppy = not have_pyarrow() or extended_types
        return _execute(
            '_DataframeReader.to_pandas_dataframe',
            dataflow=dataflow,
            is_to_pandas_dataframe=True,
            force_preppy=force_preppy,
            collect_results=True,
            fail_on_error=on_error != "null",
            fail_on_mixed_types=on_error != "null",
            fail_on_out_of_range_datetime=out_of_range_datetime != "null",
            traceparent=span_context.span_id if span_context is not None else "",
            span_context=span_context,
            cleanup=None,
            extended_types=extended_types,
            nulls_as_nan=nulls_as_nan,
        )

    def _rslex_to_pandas_with_fallback(
        self,
        dataflow
    ):
        if not have_pandas():
            raise PandasImportError()

        if not pyarrow_supports_cdata():
            raise UnexpectedError("pyarrow does not support cdata")

        if not have_pyarrow():
            force_preppy = True

        return _execute(
            '_DataframeReader._rslex_to_pandas_with_fallback',
            dataflow=dataflow,
            is_to_pandas_dataframe=True,
            force_preppy=force_preppy,
            collect_results=True
        )

    def process_rslex_batches(self, batches, stream_columns):
        if batches is None:
            raise RuntimeError("Got no record batches from rslex execution.")

        # process batches before logging + processing with reader
        import pyarrow
        random_id = str(uuid4())
        self.register_incoming_dataframe(random_id)
        self._incoming_dataframes[random_id] \
            = {i: pyarrow.Table.from_batches([batch]) for i, batch in enumerate(batches)}
        return self.complete_incoming_dataframe(random_id, partition_stream_columns=stream_columns)

    def register_incoming_dataframe(self, dataframe_id: str):
        _LoggerFactory.trace(
            _LOGGER, "register_incoming_dataframes", {
                "dataframe_id": dataframe_id})
        self._incoming_dataframes[dataframe_id] = {}

    def complete_incoming_dataframe(
        self, dataframe_id: str, partition_stream_columns=None
    ) -> "pandas.DataFrame":
        import pyarrow
        import pandas as pd

        partitions_dfs = self._incoming_dataframes[dataframe_id]
        if any(isinstance(partitions_dfs[key], pd.DataFrame) for key in partitions_dfs):
            raise _InconsistentSchemaError("A partition has no columns.")

        partitions_dfs = [
            partitions_dfs[key]
            for key in sorted(partitions_dfs.keys())
            if partitions_dfs[key].num_rows > 0
        ]
        _LoggerFactory.trace(
            _LOGGER,
            "complete_incoming_dataframes",
            {"dataframe_id": dataframe_id, "count": len(partitions_dfs)},
        )
        self._incoming_dataframes.pop(dataframe_id)

        if len(partitions_dfs) == 0:
            return pd.DataFrame({})

        def get_column_names(partition: pyarrow.Table) -> List[str]:
            return partition.schema.names

        def verify_column_names():
            def make_schema_error(prefix, p1_cols, p2_cols):
                return _InconsistentSchemaError(
                    "{0} The first partition has {1} columns. Found partition has {2} columns.\n".format(
                        prefix,
                        len(p1_cols),
                        len(p2_cols)) +
                    "First partition columns (ordered): {0}\n".format(p1_cols) +
                    "Found Partition has columns (ordered): {0}".format(p2_cols))

            expected_names = get_column_names(partitions_dfs[0])
            expected_count = partitions_dfs[0].num_columns
            row_count = 0
            size = 0
            for partition in partitions_dfs:
                row_count += partition.num_rows
                size += partition.nbytes
                found_names = get_column_names(partition)
                if partition.num_columns != expected_count:
                    _LoggerFactory.trace(
                        _LOGGER,
                        "complete_incoming_dataframes.column_count_mismatch",
                        {"dataframe_id": dataframe_id},
                    )
                    raise make_schema_error(
                        "partition had different number of columns.",
                        expected_names,
                        found_names,
                    )
                for (a, b) in zip(expected_names, found_names):
                    if a != b:
                        _LoggerFactory.trace(
                            _LOGGER, "complete_incoming_dataframes.column_names_mismatch", {
                                "dataframe_id": dataframe_id}, )
                        raise make_schema_error(
                            "partition column had different name than expected.",
                            expected_names,
                            found_names,
                        )

            _LoggerFactory.trace(
                _LOGGER,
                "complete_incoming_dataframes.info",
                {
                    "dataframe_id": dataframe_id,
                    "count": len(partitions_dfs),
                    "row_count": row_count,
                    "size_bytes": size,
                },
            )

        def determine_column_type(index: int) -> pyarrow.DataType:
            for partition in partitions_dfs:
                column = partition.column(index)
                if (
                    column.type != pyarrow.bool_()
                    or column.null_count != column.length()
                ):
                    return column.type
            return pyarrow.bool_()

        def apply_column_types(fields: List[pyarrow.Field]):
            for i in range(0, len(partitions_dfs)):
                partition = partitions_dfs[i]
                column_types = partition.schema.types
                for j in range(0, len(fields)):
                    column_type = column_types[j]
                    if column_type != fields[j].type:
                        if column_type == pyarrow.bool_():
                            column = partition.column(j)
                            import numpy as np

                            def gen_n_of_x(n, x):
                                k = 0
                                while k < n:
                                    yield x
                                    k = k + 1

                            if isinstance(column, pyarrow.ChunkedArray):
                                typed_chunks = []
                                for chunk in column.chunks:
                                    typed_chunks.append(
                                        pyarrow.array(
                                            gen_n_of_x(
                                                chunk.null_count, None), fields[j].type, mask=np.full(
                                                chunk.null_count, True), ))

                                partition = partition.remove_column(j)
                                try:
                                    partition = partition.add_column(
                                        j,
                                        fields[j],
                                        pyarrow.chunked_array(typed_chunks),
                                    )
                                except Exception as e:
                                    message = "Failed to add column to partition. Target type: {}, actual type: bool, partition id: {}, column idx: {}, error: {}, ".format(
                                        fields[j].type, i, j, e)
                                    _LoggerFactory.trace_error(_LOGGER, message)
                                    raise _InconsistentSchemaError(
                                        "A partition has a column with a different type than expected during append.\nThe type of column "
                                        "'{0}' in the first partition is {1}. In partition '{2}' found type is {3}.".format(
                                            partition.schema.names[j], str(
                                                fields[j].type), i, str(column_type), ))
                            else:
                                new_col = pyarrow.column(
                                    fields[j],
                                    pyarrow.array(
                                        gen_n_of_x(column.null_count, None),
                                        fields[j].type,
                                        mask=np.full(column.null_count, True),
                                    ),
                                )
                                partition = partition.remove_column(j)
                                partition = partition.add_column(j, new_col)
                            partitions_dfs[i] = partition
                        elif column_type != pyarrow.null():
                            if fields[j].type == pyarrow.null():
                                fields[j] = pyarrow.field(
                                    fields[j].name, column_type)
                            else:
                                _LoggerFactory.trace(
                                    _LOGGER, "complete_incoming_dataframes.column_type_mismatch", {
                                        "dataframe_id": dataframe_id}, )
                                raise _InconsistentSchemaError(
                                    "A partition has a column with a different type than expected.\nThe type of column "
                                    "'{0}' in the first partition is {1}. In partition '{2}' found type is {3}.".format(
                                        partition.schema.names[j], str(
                                            fields[j].type), i, str(column_type), ))

        def get_concatenated_stream_columns():
            first_partition_stream_columns = partition_stream_columns[0]
            stream_columns = {}
            # initialize dictionary
            for (paths, values) in first_partition_stream_columns:
                if len(paths) == 1:
                    stream_columns[paths[0]] = values
                else:
                    _LoggerFactory.trace(
                        _LOGGER,
                        "get_concatenated_stream_columns.failure_path_count",
                        {"path_count": len(paths), "partition": 0},
                    )
                    return None
            stream_column_count = len(stream_columns.keys())
            for i in range(1, len(partition_stream_columns)):
                if len(partition_stream_columns[i]) != stream_column_count:
                    # found different count of stream columns as compared to
                    # first partition
                    _LoggerFactory.trace(
                        _LOGGER,
                        "get_concatenated_stream_columns.failure_stream_count",
                        {
                            "stream_count": len(partition_stream_columns[i]),
                            "partition": i,
                            "first_partition_count": stream_column_count,
                        },
                    )
                    return None
                for (paths, values) in partition_stream_columns[i]:
                    if len(
                            paths) != 1 or paths[0] not in stream_columns.keys():
                        _LoggerFactory.trace(
                            _LOGGER, "get_concatenated_stream_columns.failure_column_mismatch", {
                                "path_count": len(paths), "partition": i}, )
                        return None
                    stream_columns[paths[0]].extend(values)

            return stream_columns

        def set_stream_columns(df, stream_columns):
            if stream_columns is not None:
                value_count = 0
                for (column, values) in stream_columns.items():
                    df[column] = values
                    value_count = len(values)
                _LoggerFactory.trace(
                    _LOGGER,
                    "set_stream_columns.success",
                    {
                        "shape": "({},{})".format(
                            len(stream_columns.keys()), value_count
                        )
                    },
                )

        with _TRACER.start_as_current_span(
                "_DataFrameReader.complete_incoming_dataframe", trace.get_current_span()
        ):
            verify_column_names()
            first_partition = partitions_dfs[0]
            column_fields = []
            names = first_partition.schema.names
            for i in range(0, first_partition.num_columns):
                f = pyarrow.field(names[i], determine_column_type(i))
                column_fields.append(f)
            apply_column_types(column_fields)

            import pyarrow

            if pyarrow_supports_promote_options():
                df = pyarrow.concat_tables(partitions_dfs, promote_options='default').to_pandas(
                    use_threads=True)
            else:
                # fallback to pre v0.14.0 way of invoking concat_tables
                df = pyarrow.concat_tables(partitions_dfs, promote=True).to_pandas(
                    use_threads=True)
            if partition_stream_columns:
                stream_columns = get_concatenated_stream_columns()
                set_stream_columns(df, stream_columns)
            _LoggerFactory.trace(
                _LOGGER,
                "complete_incoming_dataframes.success",
                {"dataframe_id": dataframe_id, "shape": str(df.shape)},
            )
            return df


_dataframe_reader = None
_dataframe_reader_lock = RLock()


def get_dataframe_reader():
    global _dataframe_reader
    if _dataframe_reader is None:
        with _dataframe_reader_lock:
            if _dataframe_reader is None:
                _dataframe_reader = _DataFrameReader()

    return _dataframe_reader
