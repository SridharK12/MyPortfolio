# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from typing import List, Dict, Any, Union, Optional, TypeVar, Tuple
import warnings
import random
import os
try:
    # Importing ABCs from collections is removed in PY3.10
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import yaml

from azureml.dataprep.rslex import PyRsDataflow, StreamInfo
from .engineapi.typedefinitions import (FieldType, FileEncoding, InvalidLineHandling, IfDestinationExists)
from .typeconversions import TypeConverter, FloatConverter, DateTimeConverter, StreamInfoConverter
from .dataprofile import DataProfile
from ._dataframereader import _execute, get_dataframe_reader
from ._rslex_executor import get_rslex_executor
from .dataflow import Dataflow, DataflowValidationError, FilePath, PromoteHeadersMode, DecimalMark, MismatchAsOption, SummaryColumnsValue
from .datasources import process_uris, FileDataSource, FileOutput, DataSource, DataDestination, MSSQLDataSource, PostgreSQLDataSource
from ._datastore_helper import (file_datastores_to_uris, _is_datapath, _is_datapaths)
from ._loggerfactory import _LoggerFactory, track, trace, _log_dataflow_execution_activity
from .step import (MultiColumnSelection, ColumnSelector)
from .tracing._open_telemetry_adapter import to_dprep_span_context
from .engineless_builders import ColumnTypesBuilder
from .expressions import Expression, col, _ensure_expression, FunctionExpression
from copy import copy

logger = None
tracer = trace.get_tracer(__name__)
_APP_NAME = 'EnginelessDataflow'
_METADATA_KEY = 'metadata'
_DEFAULT_STREAM_INFO_COLUMN = 'Path'
_PORTABLE_PATH_COLUMN_NAME = 'PortablePath'
_EMPTY_DATAFLOW_STRING = '{}'  # string representation of a empty PyRsDataflow, just an empty Python dictionary as a string


def get_logger():
    global logger
    logger = logger or _LoggerFactory.get_logger("EnginelessDataflow")
    return logger

def _log_not_supported_api_usage_and_raise(api_name: str, suggestion: str = None):
    _LoggerFactory.trace_error(get_logger(),
                              f'[NOT_SUPPORTED_API_USE_ATTEMPT] The [{api_name}] API has been deprecated and is no longer supported',
                              custom_dimensions={'api_name': api_name})
    raise NotImplementedError(f"{api_name} is no longer supported. {suggestion}.")


def _warn_if_not_default(api_name: str, parameter_name: str, value: Any, default_value: Any):
    if value is None and default_value is None:
        return
    if value != default_value:
        _LoggerFactory.trace_warn(get_logger(),
                                  '[DEPRECATED_API_USE_ATTEMPT] The parameter {} is deprecated and will be ignored. The default value of {} will be used instead.'.format(parameter_name, default_value),
                                  custom_dimensions={'api_name': api_name, 'parameter_name': parameter_name, 'default_value': default_value, 'value': value})
        warnings.warn('The parameter {} in {} is deprecated and will be ignored. The default value of {} will be used instead.'.format(parameter_name, api_name, default_value), category = DeprecationWarning, stacklevel = 3)


class EnginelessBuilders:
    def __init__(self, dataflow: 'EnginelessDataflow'):
        self._dataflow = dataflow

    def set_column_types(self) -> ColumnTypesBuilder:
        """
        Constructs an instance of :class:`ColumnTypesBuilder`.
        """
        return ColumnTypesBuilder(self._dataflow)


class EnginelessDataflowMetadata:

    def __init__(self, engineless_dataflow):
        # need to keep reference to parent EnginelessDataflow to set its PyRsDataflow whenever metadata state is updated
        self._engineless_dataflow = engineless_dataflow
        self._set_meta_dict()

    @property
    def _py_rs_dataflow(self):
        return self._engineless_dataflow._py_rs_dataflow

    def _set_meta_dict(self):
        self._dict = self._engineless_dataflow._to_yaml_dict().get('metadata', {})

    def __getitem__(self, key):
        if not self.__contains__(key):
            raise KeyError(key)
        return self._py_rs_dataflow.get_schema_property(_METADATA_KEY, key)

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        self._engineless_dataflow._set_py_rs_dataflow(self._py_rs_dataflow.set_schema_property(_METADATA_KEY, key, value))
        self._set_meta_dict()

    def __str__(self) -> str:
        return str(self._dict)

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, key) -> str:
        return self._py_rs_dataflow.has_schema_property(_METADATA_KEY, key)

    def get(self, key, default=None):
        return self[key] if self.__contains__(key) else default

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def __eq__(self, other):
        if not isinstance(other, (EnginelessDataflowMetadata, dict)):
            raise ValueError('EnginelessDataflowMetadata equality expects a dictionary or a EnginelessDataflowMetadata')
        other = other._dict if isinstance(other, EnginelessDataflowMetadata) else other
        return other == self._dict


class EnginelessDataflow(Dataflow):
    """
    v1 Dataflow wrapper around a RSlex Dataflow YAML.
    """

    def __init__(self, py_rs_dataflow=_EMPTY_DATAFLOW_STRING, dataflow_json=None):
        Dataflow.__init__(self, engine_api=None)
        self._setup(py_rs_dataflow, dataflow_json, self._rs_dataflow_yaml, self._engine_api, self._steps)

    def _setup(self, py_rs_dataflow, dataflow_json=None, _rs_dataflow_yaml=None, _engine_api=None, _steps=None):
        # v1 Dataflow attributes that are kept for consistency, should always be none
        self._engine_api = _engine_api
        self._steps = _steps
        self._rs_dataflow_yaml = _rs_dataflow_yaml

        self._dataflow_json = dataflow_json
        self._set_py_rs_dataflow(py_rs_dataflow)
        self.builders = EnginelessBuilders(self)
        self._meta = EnginelessDataflowMetadata(self)

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        self._set_py_rs_dataflow(self._py_rs_dataflow.set_schema(_METADATA_KEY, value))
        self._meta = EnginelessDataflowMetadata(self)

    @staticmethod
    def open(file_path: str) -> 'Dataflow':
        """
        Opens a Dataflow with specified name from the package file.

        :param file_path: Path to the package containing the Dataflow.
        :return: The Dataflow.
        """
        with open(file_path, 'r') as f:
            return EnginelessDataflow.from_json(f.read())

    def save(self, file_path: str):
        """
        Saves the Dataflow to the specified file

        :param file_path: The path of the file.
        """
        with open(file_path, 'w') as f:
            f.write(self.to_json())

    def _set_py_rs_dataflow(self, py_rs_dataflow):
        py_rs_dataflow = py_rs_dataflow or _EMPTY_DATAFLOW_STRING

        if isinstance(py_rs_dataflow, dict):
            py_rs_dataflow = yaml.safe_dump(py_rs_dataflow)

        if isinstance(py_rs_dataflow, str):
            from azureml.dataprep.api._rslex_executor import ensure_rslex_environment
            ensure_rslex_environment()
            # covers validation
            py_rs_dataflow = PyRsDataflow(py_rs_dataflow)
        elif isinstance(py_rs_dataflow, PyRsDataflow):
            self._py_rs_dataflow = py_rs_dataflow
        else:
            raise ValueError('Expect RSlex Dataflow YAML string or RSlex PyRsDataflow')

        self._py_rs_dataflow = py_rs_dataflow.set_schema_property(_METADATA_KEY, 'infer_column_types', 'False')

    def __repr__(self) -> str:
        return 'EnginelessDataflow:\n' + self._to_yaml_string()

    def __setstate__(self, newstate):
        self._setup(newstate)

    def __getstate__(self):
        return self._to_yaml_string()

    def __getitem__(self, key):
        if isinstance(key, (slice, int)):
            self_yaml_dict = self._to_yaml_dict()

            def get_loader():
                if 'paths' in self_yaml_dict:
                    return {'paths': self_yaml_dict['paths']}
                if 'query_source' in self_yaml_dict:
                    return {'query_source': self_yaml_dict['query_source']}
                raise RuntimeError('missing loader')

            # only return loader, should be either paths or query_source
            if (isinstance(key, int) and key == 0) or key.start == 0 and key.stop == 1:
                return EnginelessDataflow(get_loader())

            # always a slice now
            if isinstance(key, int):
                key = slice(key, key + 1)

            if key.start == 0:
                key = slice(key.start, key.stop - 1, key.step)
                new_yaml_dirc = get_loader()
            else:
                key = slice(key.start - 1, None if key.stop is None else key.stop - 1, key.step)
                new_yaml_dirc = {}

            new_yaml_dirc['transformations'] = self_yaml_dict['transformations'][key]
            return EnginelessDataflow(new_yaml_dirc)

        elif isinstance(key, str):  # TODO leaving expressions as in bc used as input in dataflow
            return col(key)
        elif isinstance(key, Iterable):
            return col(list(key))
        else:
            raise TypeError("Invalid argument type.")

    # Will fold the right EnginelessDataflow into the left by appending the rights steps to the lefts.
    def __add__(self, other):
        if not isinstance(other, EnginelessDataflow):
            raise TypeError("Can only add two EnginelessDataflow objects together. Was given: " + str(type(other)))
        return EnginelessDataflow(self._py_rs_dataflow.append_transformations_from_dataflow(other._py_rs_dataflow))

    def __deepcopy__(self, memo=None):
        return EnginelessDataflow(self._to_yaml_string())

    def _to_yaml_string(self) -> str:
        return self._py_rs_dataflow.to_yaml_string()

    def _to_yaml_dict(self, py_rs_dataflow=None) -> dict:
        py_rs_dataflow = py_rs_dataflow or self._py_rs_dataflow
        return yaml.safe_load(py_rs_dataflow.to_yaml_string())

    def _copy_and_update_metadata(self,
                                  action: str,
                                  source: str,
                                  **kwargs) -> 'EnginelessDataflow':
        # EnginelessDataflow is immutable so even if no changes occur & same instance is passed, nothing bad should happen
        def set_if_not_set(dataflow, key, value):
            if key not in dataflow.meta:
                dataflow.meta[key] = value
                print(dataflow.meta)

        dataflow = EnginelessDataflow(self._to_yaml_string())
        set_if_not_set(dataflow, 'activity', action)
        set_if_not_set(dataflow, 'activityApp', source)

        run_id = os.environ.get("AZUREML_RUN_ID", None)
        if run_id is not None:
            # keep this here so not to break existing reporting
            set_if_not_set(dataflow, 'runId', run_id)
            set_if_not_set(dataflow, 'run_id', run_id)

        for (k, v) in kwargs.items():
            set_if_not_set(dataflow, k, v)

        return dataflow

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def read_parquet_file(self, include_path: bool = False) -> 'EnginelessDataflow':
        """
        Reads the Parquet files in the dataset.

        :param include_path_column: Indicates whether to include the path column in the output.
        :return: The modified Dataflow.
        """
        return self._add_transformation('read_parquet', {"include_path_column": include_path})

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def read_preppy(self, include_path: bool = False) -> 'EnginelessDataflow':
        """
        Reads the Preppy files in the dataset.

        :param include_path_column: Indicates whether to include the path column in the output.
        :return: The modified Dataflow.
        """
        return self._add_transformation('read_files', {"keep_existing_columns": include_path, "reader": "preppy"})

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def parse_delimited(self,
                        separator: str,
                        headers_mode: PromoteHeadersMode,
                        encoding: FileEncoding,
                        quoting: bool,
                        partition_size: Optional[int] = None,
                        empty_as_string: bool = False,
                        inlucde_path: bool = True) -> 'Dataflow':
        """
        Adds step to parse CSV data.

        :param separator: The separator to use to split columns.
        :param headers_mode: How to determine column headers.
        :param encoding: The encoding of the files being read.
        :param quoting: Whether to handle new line characters within quotes. This option will impact performance.
        :param skip_rows: How many rows to skip.
        :param skip_mode: The mode in which rows are skipped.
        :param comment: Character used to indicate a line is a comment instead of data in the files being read.
        :param partition_size: Desired partition size.
        :param empty_as_string: Whether to keep empty field values as empty strings. Default is read them as null.
        :return: A new Dataflow with Parse Delimited Step added.
        """
        self._raise_if_multi_char('separator', separator)
        self._validate_partition_size(partition_size)

        headers_mode_mapped = self._map_headers_mode(headers_mode)

        encoding_mapped = self._map_encoding(encoding)
        args = {'delimiter': separator,
                'header': headers_mode_mapped,
                'support_multi_line': quoting,
                'empty_as_string': empty_as_string,
                'encoding': encoding_mapped,
                'include_path_column': inlucde_path,
                'infer_column_types': False}
        if partition_size is not None:
            args['partition_size'] = partition_size
        return self._add_transformation('read_delimited',
                                        args=args)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def parse_json_lines(self,
                         encoding: FileEncoding,
                         partition_size: Optional[int] = None,
                         invalid_lines: InvalidLineHandling = InvalidLineHandling.ERROR,
                         include_path: Optional[bool] = False) -> 'Dataflow':
        """
        Creates a new Dataflow with the operations required to read JSON lines files.

        :param invalid_lines: How to handle invalid JSON lines.
        :param encoding: The encoding of the files being read.
        :param partition_size: Desired partition size.
        :return: A new Dataflow with Read JSON line Step added.
        """
        invalid_lines = self._map_invalid_lines(invalid_lines)

        args = {"invalid_lines": invalid_lines,
                "encoding": self._map_encoding(encoding),
                "include_path_column": include_path}
        if partition_size is not None:
            args['partition_size'] = partition_size
        return self._add_transformation('read_json_lines', args)

    def _map_invalid_lines(self, invalid_lines):
        invalid_lines_mapped = ''
        if invalid_lines.value == InvalidLineHandling.ERROR.value:
            invalid_lines_mapped = 'error'
        elif invalid_lines.value == InvalidLineHandling.DROP.value:
            invalid_lines_mapped = 'drop'
        else:
            raise ValueError('Unsupported invalid lines handling: ' + str(invalid_lines))
        return invalid_lines_mapped

    def _map_encoding(self, encoding):
        encoding_mapped = ''
        if encoding.value == FileEncoding.UTF8.value:
            encoding_mapped = 'utf8'
        elif encoding.value == FileEncoding.ISO88591.value:
            encoding_mapped = 'iso88591'
        elif encoding.value == FileEncoding.LATIN1.value:
            encoding_mapped = 'latin1'
        elif encoding.value == FileEncoding.ASCII.value:
            encoding_mapped = 'ascii'
        elif encoding.value == FileEncoding.WINDOWS1252.value:
            encoding_mapped = 'windows1252'
        elif encoding.value == FileEncoding.UTF16.value:
            encoding_mapped = 'utf16'
        else:
            raise ValueError('Unsupported encoding: ' + str(encoding))
        return encoding_mapped

    def _map_headers_mode(self, headers_mode):
        if headers_mode.value == PromoteHeadersMode.ALLFILES.value:
            return 'all_files_different_headers'
        if headers_mode.value == PromoteHeadersMode.SAMEALLFILES.value:
            return 'all_files_same_headers'
        if headers_mode.value == PromoteHeadersMode.FIRSTFILE.value:
            return 'from_first_file'
        if headers_mode.value == PromoteHeadersMode.NONE.value:
            return 'no_header'
        raise ValueError('Unsupported headers_mode: ' + str(headers_mode))

    def _add_transformation(self, step, args, index = None):
        return EnginelessDataflow(self._py_rs_dataflow.add_transformation(step, args, index))

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def _add_columns_from_partition_format(self,
                                column: str,
                                partition_format: str,
                                ignore_error: bool) -> 'EnginelessDataflow':
        """
        Add new columns to the dataset based on matching the partition format for provided column.

        :param partition_format: The partition format matching the column to create columns.
        :param ignore_error: Indicate whether or not to fail the execution if there is any error.
        :return: The modified Dataflow.
        """
        args = {'path_column': column, 'partition_format': partition_format, 'ignore_errors': ignore_error}
        # this one is special as it is being added as a first transformation to make sure path is still in the dataset, as future read_* steps might drop it
        return self._add_transformation('extract_columns_from_partition_format', args, index = 0)

    @track(get_logger)
    def assert_value(self,
                     columns,
                     expression,
                     policy = None,
                     error_code: str = 'AssertionFailed') -> 'Dataflow':
        _LoggerFactory.trace_warn(get_logger(),
                                  '[DEPRECATED_API_USE_ATTEMPT] [assert_value] is deprecated and will be ignored.',
                                  custom_dimensions={'api_name': 'assert_value'})
        return self

    def get_missing_secrets(self) -> List[str]:
        _LoggerFactory.trace_warn(get_logger(),
                                  '[DEPRECATED_API_USE_ATTEMPT] [get_missing_secrets] is deprecated and will be ignored.',
                                  custom_dimensions={'api_name': 'get_missing_secrets'})
        return []

    def use_secrets(self, secrets: Dict[str, str]):
        _LoggerFactory.trace_warn(get_logger(),
                                  '[DEPRECATED_API_USE_ATTEMPT] [use_secrets] is deprecated and will be ignored.',
                                  custom_dimensions={'api_name': 'use_secrets'})

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def filter(self, expression: Expression) -> 'Dataflow':
        """
        Filters the data, leaving only the records that match the specified expression.

        .. remarks::

            Expressions are started by indexing the Dataflow with the name of a column. They support a variety of
                functions and operators and can be combined using logical operators. The resulting expression will be
                lazily evaluated for each record when a data pull occurs and not where it is defined.

            .. code-block:: python

                dataflow['myColumn'] > dataflow['columnToCompareAgainst']
                dataflow['myColumn'].starts_with('prefix')

        :param expression: The expression to evaluate.
        :return: The modified Dataflow.
        """
        expression = _ensure_expression(expression)
        # wrap expression text first into function taking in a row and expression and then into a function without arguments
        expression_json = expression._as_json_string()
        expression_text = f'{{"r":["Function",[[],{{"r":[]}},{{"r":["Function",[["row"],{{"r":[]}},{expression_json}]]}}]]}}'
        filter_args = {'language': 'native', 'function_source': expression_text}
        return self._add_transformation('filter', filter_args)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def random_split(self,
                     percentage: float,
                     seed: Optional[int] = None) -> ('Dataflow', 'Dataflow'):
        """
        Returns two Dataflows from the original Dataflow, with records randomly and approximately split by the percentage
            specified (using a seed). If the percentage specified is p (where p is between 0 and 1), the first returned dataflow
            will contain approximately p*100% records from the original dataflow, and the second dataflow will contain all the
            remaining records that were not included in the first. A random seed will be used if none is provided.
            Actual split does not happen when this transformation is applied, it only happens when the dataflow is executed.
            Note: even with a seed actual split depends on the platform dataflow is executed on and could differ across platforms and
            sdk versions.

        :param percentage: The approximate percentage to split the Dataflow by. This must be a number between 0.0 and 1.0.
        :param seed: The seed to use for the random split.
        :return: Two Dataflows containing records randomly split from the original Dataflow. If the percentage specified is p,
            the first dataflow contains approximately p*100% of the records from the original dataflow.

        """
        if percentage < 0.0 or percentage > 1.0:
            raise ValueError("percentage must be a number between 0.0 and 1.0.")

        seed = seed or random.randint(0, 4294967295)
        split_a = self._add_transformation('sample', {"sampler": "random_percent",
                                                           "sampler_arguments": {
                                                               "probability": percentage,
                                                               "probability_lower_bound": 0.0,
                                                               "seed": seed}})
        split_b = self._add_transformation('sample', {"sampler": "random_percent",
                                                           "sampler_arguments": {
                                                               "probability": 1.0,
                                                               "probability_lower_bound": percentage,
                                                               "seed": seed}})
        return split_a, split_b

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def rename_columns(self, column_pairs: Dict[str, str]) -> 'Dataflow':
        """
        Renames the specified columns.

        :param column_pairs: The columns to rename and the desired new names.
        :return: The modified Dataflow.
        """
        return self._add_transformation('rename_columns', column_pairs)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def replace_datasource(self, new_datasource: DataSource) -> 'Dataflow':
        """
        Returns new Dataflow with its DataSource replaced by the given one.

        .. remarks::

            The given 'new_datasource' must match the type of datasource already in the Dataflow.
            For example a MSSQLDataSource cannot be replaced with a FileDataSource.

        :param new_datasource: DataSource to substitute into new Dataflow.
        :return: The modified Dataflow.
        """
        dict = self._to_yaml_dict()
        if isinstance(new_datasource, (MSSQLDataSource, PostgreSQLDataSource)):
            from .readers import read_sql, read_postgresql
            if "query_source" not in dict:
                raise ValueError('Dataflow is doesn\'t have a query source present')
            query = dict["query_source"]["query"]
            new_dataflow = read_sql(new_datasource, query) \
                if isinstance(new_datasource, MSSQLDataSource) else read_postgresql(new_datasource, query)
        elif "query_source" in dict:
            raise ValueError("Can only replace 'Database' Datasource with MSSQLDataSource or PostgreSQLDataSource.")
        else:
            new_dataflow = EnginelessDataflow.from_paths(new_datasource)
        return new_dataflow + self[1:]

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def take(self, count: int) -> 'EnginelessDataflow':
        """
        Takes the specified count of records.

        :param count: The number of records to take.
        :return: The modified Dataflow.
        """
        if not (isinstance(count, int) and count >= 0):
            raise ValueError('count must be a positive integer')
        return self._add_transformation('take', count)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def data_changed_time(self, column: str) -> 'EnginelessDataflow':
        """
        Summarizes the data into a single column that contains the time when the data was last changed.

        :param column: The name of the column to add.
        :return: The modified Dataflow.
        """
        return self[0]._add_transformation('data_changed_time', {'data_changed_time_column': column})

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def take_sample(self,
                    probability: float,
                    seed: Optional[int] = None) -> 'EnginelessDataflow':
        """
        Takes a random sample of the available records.

        :param probability: The probability of a record being included in the sample.
        :param seed: The seed to use for the random generator.
        :return: The modified Dataflow.
        """
        return self._add_transformation('take_random_sample',
                                        {"probability": probability,
                                         "seed": seed or random.randint(0, 4294967295)})

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def drop_columns(self, columns: MultiColumnSelection) -> 'EnginelessDataflow':
        """
        Drops the specified columns.

        :param columns: The columns to drop.
        :return: The modified Dataflow.
        """
        return self._add_transformation('drop_columns', _column_selection_to_py_rs_dataflow_selector(columns))

    @track(get_logger)
    def keep_columns(self, columns: MultiColumnSelection, validate_column_exists: bool = False) -> 'EnginelessDataflow':
        """
        Keeps the specified columns.

        :param columns: The columns to keep.
        :return: The modified Dataflow.
        """

        dataflow = self._add_transformation('keep_columns', _column_selection_to_py_rs_dataflow_selector(columns))
        if validate_column_exists:
            dataflow.verify_has_data(verify_columns=True)
        return dataflow

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def write_to_parquet(self,
                         file_path: Optional[DataDestination] = None,
                         directory_path: Optional[DataDestination] = None,
                         single_file: bool = False,
                         error: str = 'ERROR',
                         row_groups: int = 0,
                         if_destination_exists: str = IfDestinationExists.MERGEWITHOVERWRITE,
                         partition_keys: List[str] = None) -> 'EnginelessDataflow':
        """
        Writes out the data in the Dataflow into Parquet files.

        :param file_path: The path in which to store the output file. Currently unsupported for now, will be ignored.
        :param directory_path: The path in which to store the output files.
        :param single_file: Whether to store the entire Dataflow in a single file. Currently unsupported for now, will be ignored.
        :param error: String to use for error values.
        :param row_groups: Number of rows to use per row group. Currently unsupported for now, will be ignored.
        :param if_destination_exists: Behavior if destination exists.
        :param partition_keys: Optional, list of column names used to write the data by.
            Defaults to be None.
        :return: The modified Dataflow.
        """
        if single_file or file_path:
            warnings.warn("Writing to single parquet file is not supported yet, file_path will be ignored")

        if row_groups != 0:
            warnings.warn("Writing to parquet by row groups is not supported yet")

        if partition_keys is not None:
            if any(map(lambda key: not key, partition_keys)):
                raise ValueError("Please provide valid partition keys")

        # rslex are using a different naming convention from clex
        if_destination_exists_mapping = {
            IfDestinationExists.MERGEWITHOVERWRITE.name: 'merge_with_overwrite',
            IfDestinationExists.APPEND.name: 'append',
            IfDestinationExists.FAIL.name: 'fail',
            IfDestinationExists.REPLACE: 'replace'
        }

        # directory_path is always DataReference from partition_by api
        from ._datastore_helper import get_datastore_value
        datastore_value = get_datastore_value(directory_path)[1]._to_pod()

        # TODO add error handling?
        transformation_arguments = {
            'writer': 'parquet',
            'destination':
                {
                    'directory': datastore_value['path'],
                    'handler': 'AmlDatastore',
                    'arguments': datastore_value
                },
            'existing_file_handling': if_destination_exists_mapping[if_destination_exists.name]
        }

        if partition_keys is not None:
            transformation_arguments['writer_arguments'] = {'partition_keys': partition_keys}

        return self._add_transformation('write_files', transformation_arguments)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def write_streams(self,
                      streams_column: str,
                      base_path: FileOutput,
                      file_names_column: Optional[str] = None,
                      prefix_path: Optional[str] = None) -> 'EnginelessDataflow':
        """
        Writes the streams in the specified column to the destination path. By default, the name of the files written
            will be the resource identifier of the streams. This behavior can be overriden by specifying a column which
            contains the names to use.

        :param streams_column: The column containing the streams to write.
        :param file_names_column: A column containing the file names to use.
        :param base_path: The path under which the files should be written.
        :param prefix_path: The prefix path that needs to be removed from the target paths.
        :return: The modified Dataflow.
        """
        # rsdataflow doesnt accept None for prefix path
        if prefix_path is None:
            prefix_path = ""
        return self._add_transformation('write_streams_to_files',
                                        {
                                            'streams_column': streams_column,
                                            'destination':
                                                {
                                                    'directory': str(base_path.underlying_value.resource_details[0].to_pod()["path"]),
                                                    'handler': 'Local'
                                                },
                                            'file_names_column': file_names_column,
                                            'path_prefix': prefix_path
                                        })

    def _find_path_prefix(self, stream_column=_DEFAULT_STREAM_INFO_COLUMN, avoid_data_pull=False):
        if stream_column != _DEFAULT_STREAM_INFO_COLUMN:
            return (None, False)
        try:
            dataflow_yaml_dict = self._to_yaml_dict()
            if 'transformations' in dataflow_yaml_dict:
                for transformation in dataflow_yaml_dict['transformations']:
                    if 'to_streams' in transformation:
                        return (None, False)
            paths = dataflow_yaml_dict['paths']
            if len(paths) > 1:
                return (None, False)
            path_obj = paths[0]
            path = path_obj.get('pattern', path_obj.get('file', path_obj.get('folder')))
            from azureml.dataprep.api._rslex_executor import ensure_rslex_environment
            ensure_rslex_environment()
            resource_id = PyRsDataflow.resource_id_from_uri(path)
            return self._get_prefix(resource_id, avoid_data_pull)
        except Exception:
            return (None, False)

    def _get_prefix(self, path, avoid_data_pull=False):
        """Determine if there exists a common prefix for all files which may exist under the given path/dataflow.

        :param path: Path extracted from dataflow
        :return: Path which is common prefix of all files under path/dataflow, or None if a common prefix was not found.
        """
        from .errorhandlers import ExecutionError
        from .functions import get_portable_path
        from .expressions import col
        from azureml.dataprep.native import DataPrepError
        from azureml.dataprep.rslex import PyErrorValue
        try:
            from azureml.exceptions import UserErrorException
        except ImportError:
            from .mltable._mltable_helper import UserErrorException

        import re

        if '*' in path:
            return ('/'.join(re.split(r'/|\\', path.split('*')[0])[:-1]), False)

        supported_pattern = re.compile(
            r'^(https?|adl|wasbs?|abfss?)://', re.IGNORECASE)
        if supported_pattern.match(path):
            return (path[:path.rindex('/')], False)

        if avoid_data_pull:
            return (None, False)

        dataflow = self.add_column(get_portable_path(col(_DEFAULT_STREAM_INFO_COLUMN), None), _PORTABLE_PATH_COLUMN_NAME, _DEFAULT_STREAM_INFO_COLUMN)
        paths = []
        try:
            paths = [r[_PORTABLE_PATH_COLUMN_NAME] for r in dataflow.take(1)._to_pyrecords(use_new_flow=True)]
        except ExecutionError as e:
            if 'InvalidPath' in e.error_code or 'NotFound' in e.error_code:
                return (None, False)
            raise e
        if len(paths) == 0:
            return (None, False)
        if isinstance(paths[0], (DataPrepError, PyErrorValue)):
            raise UserErrorException('DataPrepError: {0}, '
                                    'please make sure the URI/Path is accessible'
                                    ' and the AML datastore is registered/accessible.'.format(paths[0]))
        if len(paths) == 1 and paths[0].endswith(path):
            return (path.replace('\\', '/')[:path.rindex('/')], True)
        return (path, True)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def add_column(self, expression: Expression, new_column_name: str, prior_column: str) -> 'EnginelessDataflow':
        # create a dictionary with keys new_column, prior_column, function_source
        # function that takes in a string and changes all single quotes to double quotes
        expression = FunctionExpression([], {}, FunctionExpression(['row'], {}, expression))
        add_col = {'language': 'Native', 'expressions': [{'new_column': new_column_name, 'prior_column': prior_column, 'function_source': expression._as_json_string()}]}
        return self._add_transformation('add_columns', add_col)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def _add_columns_from_record(self,
                                 expression: Expression,
                                 prior_column: str = None,
                                 new_column_prefix: str = None) -> 'EnginelessDataflow':
        expression = FunctionExpression([], {}, FunctionExpression(['row'], {}, expression))
        add_col_from_record = {'language': 'native', 'new_column_prefix': new_column_prefix, 'prior_column': prior_column, 'function_source': expression._as_json_string()}
        return self._add_transformation('extract_columns_from_record', add_col_from_record)

    def _add_portable_path_column(self, stream_column=_DEFAULT_STREAM_INFO_COLUMN):
        (prefix_path, _) = self._find_path_prefix(stream_column=stream_column)
        from .functions import get_portable_path
        from .expressions import col
        return self.add_column(get_portable_path(col(stream_column), prefix_path), _PORTABLE_PATH_COLUMN_NAME, stream_column), _PORTABLE_PATH_COLUMN_NAME

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_csv_streams(self, separator: str = ',', na: str = 'NA', error: str = 'ERROR') -> 'EnginelessDataflow':
        return self._add_transformation('to_streams', {'writer': 'delimited', 'writer_arguments': {'delimiter': separator, 'null_replacement': na, 'error_replacement': error}})

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_parquet_streams(self, error: str = 'ERROR', rows_per_group: int = 5000) -> 'EnginelessDataflow':
        if rows_per_group != 5000:
            _LoggerFactory.trace_warn(get_logger(),
                                      "Non default value of [rows_per_group] was provided and is not supported in this version of SDK.",
                                      custom_dimensions={'rows_per_group': rows_per_group})
        _LoggerFactory.trace(get_logger(), "Using default value of [rows_per_group] was overriden.", custom_dimensions={'rows_per_group': rows_per_group})
        # TODO: fix mismatch between runtime and SDK
        return self._add_transformation('to_streams', {'writer': 'parquet', 'writer_arguments': {'error_replacement': error, 'rows_per_group': rows_per_group}})

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def distinct(self, columns: MultiColumnSelection) -> 'EnginelessDataflow':
        """
        Filters out records that contain duplicate values in the specified columns, leaving only a single instance.

        :param columns: The source columns.
        :return: The modified Dataflow.
        """
        return self._add_transformation('distinct', _column_selection_to_py_rs_dataflow_selector(columns))

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def distinct_rows(self) -> 'EnginelessDataflow':
        """
        Filters out records that contain duplicate values in all columns, leaving only a single instance.
        :return: The modified Dataflow.
        """
        return self._add_transformation('distinct', {'pattern': '^.*$',
                'invert': False,
                'ignore_case': True})

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def skip(self,
             count: int) -> 'EnginelessDataflow':
        """
        Skips the specified number of records.

        :param count: The number of records to skip.
        :return: The modified Dataflow.
        """
        return self._add_transformation('skip', count)

    TypeConversionInfo = TypeVar('TypeConversionInfo',
                                 FieldType,
                                 TypeConverter,
                                 Tuple[FieldType, List[str], Tuple[FieldType, str]])

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def set_column_types(self, type_conversions: Dict[str, TypeConversionInfo]) -> 'Dataflow':
        """
        Converts values in specified columns to the corresponding data types.

        .. remarks::

            The values in the type_conversions dictionary could be of several types:

            * :class:`azureml.dataprep.FieldType`
            * :class:`azureml.dataprep.TypeConverter`
            * Tuple of DATE (:class:`azureml.dataprep.FieldType`) and List of format strings (single format string is also supported)

            .. code-block:: python

                import azureml.dataprep as dprep

                dataflow = dprep.read_csv(path='./some/path')
                dataflow = dataflow.set_column_types({'MyNumericColumn': dprep.FieldType.DECIMAL,
                                                   'MyBoolColumn': dprep.FieldType.BOOLEAN,
                                                   'MyAutodetectDateColumn': dprep.FieldType.DATE,
                                                   'MyDateColumnWithFormat': (dprep.FieldType.DATE, ['%m-%d-%Y']),
                                                   'MyOtherDateColumn': dprep.DateTimeConverter(['%d-%m-%Y'])})

            .. note::

                If you choose to convert a column to DATE and do not provide \
                **format(s)** to use, DataPrep will attempt to infer a format to use by pulling on the data. \
                If a format could be inferred from the data, it will be used to convert values in the corresponding
                column. However, if a format could not be inferred, this step will fail and you will need to either \
                provide the format to use or use the interactive builder \
                :class:`azureml.dataprep.api.builders.ColumnTypesBuilder` by calling \
                'dataflow.builders.set_column_types()'

        :param type_conversions: A dictionary where key is column name and value is either
            :class:`azureml.dataprep.FieldType` or :class:`azureml.dataprep.TypeConverter` or a Tuple of
            DATE (:class:`azureml.dataprep.FieldType`) and List of format strings

        :return: The modified Dataflow.
        """
        return self._set_column_types(list(type_conversions.items()))

    def _optimized_get_pandas_dataframe_from_partition_info(self, source, index_in_source, extended_types, nulls_as_nan, on_error, out_of_range_datetime):
        # replace first block in the source dataflow with a single partition StreamInfo, allowing direct file access without any listing.
        optimized_dataflow = EnginelessDataflow.from_stream_infos([source])
        optimized_dataflow += self[1:]
        return optimized_dataflow.select_partitions([index_in_source]).to_pandas_dataframe(extended_types, nulls_as_nan, on_error, out_of_range_datetime)

    @staticmethod
    def _get_optimal_path(common_path_list, input_path_list, is_datastore, verbose):
        """Merge common path segment list with stored path segment list and returns optimal path.

        :param common_path_list: list containing common path segments
        :type partition_keys: builtin.list[str]
        :param input_path_list: list containing stored path segments
        :type input_path_list: builtin.list[str]
        :param verbose: boolean value to print out the optimizations performed
        :type verbose: bool
        :return: str
        :rtype: str
        """
        import itertools
        import fnmatch
        stored_path_str = '/'.join(input_path_list)

        # UX case - check if the path ends with /**/* -> try to normalize it to /
        try:
            if input_path_list[-2] == "**" and input_path_list[-1] == "*":
                input_path_list = input_path_list[:-2]
            elif input_path_list[-1] == "**":
                input_path_list = input_path_list[:-1]
        except IndexError:
            pass

        optimal_path_list = []
        optimized_count = 0
        used_existing_count = 0

        for common_path_segment, input_path_segment in itertools.zip_longest(common_path_list, input_path_list):
            if common_path_segment and not input_path_segment:
                # case where we were able to optimize using common path
                optimized_count += 1
                optimal_path_list.append(common_path_segment)
            elif not common_path_segment and input_path_segment:
                # case where there is no common path, but we still need to keep segment from input to preserve user defined scoping
                # e.g. inputs: ['/data/2019/10/file.txt', '/data/2020/10/file.txt'], common path: /data/ and input path: '/data/2019/10/file.txt' -> we need to keep '/data/2019/10/file.txt'
                # or we risk picking up files user has never specified in the input
                used_existing_count += 1
                optimal_path_list.append(input_path_segment)
            else:
                # case where both are not None
                # if a common path segment match we continue, if they differ and input is a * then we take comon paht
                # otherwise this input did not produce any of the data we care about, so it should be skipped
                if common_path_segment == input_path_segment:
                    used_existing_count += 1
                    optimal_path_list.append(input_path_segment)
                else:
                    # this handles cases where input path segment has wildcards and so we need to make sure that common path segment satisfies it
                    if fnmatch.fnmatch(common_path_segment, input_path_segment):
                        optimized_count += 1
                        optimal_path_list.append(common_path_segment)
                    else:
                        return (False, None)

        # print how many segments we were able to optimize by showing the two different path values.
        result = '/'.join(optimal_path_list)
        if verbose:
            print("Optimized {} path segments using provided common path.".format(optimized_count))
            print("Stored path: {}".format(stored_path_str))
            print("Optimized path: {}".format(result))

        get_logger().info("Successfully optimized {} path segments and used {} existing "
                            "path segments.".format(optimized_count, used_existing_count))

        if is_datastore:
            result = '/'.join(optimal_path_list[1:])
            if verbose:
                print("Removing datastore name, new optimized path: {}".format(result))

        return True, result

    def _remove_data_read_steps_and_drop_columns(self, remove_drop_column):

        yaml_dict = self._to_yaml_dict()

        def find_read_and_drop_transformation(transformations):
            if transformations:
                read_transformation = None
                drop_columns_transformation = None
                for transformation in transformations:
                    name = transformation if isinstance(transformation, str) else next(transformation.keys().__iter__())
                    if name in ['read_delimited', 'read_json_lines', 'read_parquet', 'read_files', 'read_coco', 'read_delta_lake', 'read_files']:
                        read_transformation = transformation
                    elif name == 'drop_columns':
                        columns = transformation.get('drop_columns')
                        if columns and ((isinstance(columns, list) and 'Path' in columns) or columns == 'Path'):
                            drop_columns_transformation = transformation
            return read_transformation, drop_columns_transformation
        transformations = yaml_dict.get('transformations')
        read_transformation, drop_columns_transformation = find_read_and_drop_transformation(transformations)

        if read_transformation is None and drop_columns_transformation is None:
            return self

        if read_transformation:
            transformations.remove(read_transformation)
        if drop_columns_transformation:
            if isinstance(drop_columns_transformation['drop_columns'], list):
                drop_columns_transformation['drop_columns'].remove('Path')
            else:
                transformations.remove(drop_columns_transformation)

        return EnginelessDataflow(yaml_dict)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def summarize(self,
                  summary_columns: Optional[List[SummaryColumnsValue]] = None,
                  group_by_columns: Optional[List[str]] = None,
                  join_back: bool = False,
                  join_back_columns_prefix: Optional[str] = None) -> 'EnginelessDataflow':
        """
        Summarizes data by running aggregate functions over specific columns.

        .. remarks::

            The aggregate functions are independent and it is possible to aggregate the same column multiple times.
                Unique names have to be provided for the resulting columns. The aggregations can be grouped, in which case
                one record is returned per group; or ungrouped, in which case one record is returned for the whole dataset.
                Additionally, the results of the aggregations can either replace the current dataset or augment it by
                appending the result columns.

            .. code-block:: python

                import azureml.dataprep as dprep

                dataflow = dprep.read_csv(path='./some/path')
                dataflow = dataflow.summarize(
                    summary_columns=[
                        dprep.SummaryColumnsValue(
                            column_id='Column To Summarize',
                            summary_column_name='New Column of Counts',
                            summary_function=dprep.SummaryFunction.COUNT)],
                    group_by_columns=['Column 1', 'Column 2'])

        :param summary_columns: List of SummaryColumnsValue, where each value defines the column to summarize,
            the summary function to use and the name of resulting column to add.
        :param group_by_columns: Columns to group by.
        :param join_back: [DEPRECATED] Whether to append subtotals or replace current data with them.
        :param join_back_columns_prefix: [DEPRECATED] Prefix to use for subtotal columns when appending them to current data.
        :return: The modified Dataflow.
        """
        # handle string as a single column
        group_by_columns = [] if group_by_columns is None else\
            [group_by_columns] if isinstance(group_by_columns, str) else group_by_columns
        if not summary_columns and len(group_by_columns) == 0:
            raise ValueError("Missing required argument. Please provide at least one of the following arguments: "
                             "'summary_columns', 'group_by_columns'.")
        summary_columns = [summary_columns] if isinstance(summary_columns, SummaryColumnsValue) else summary_columns

        _warn_if_not_default('summarize', 'join_back', join_back, False)
        _warn_if_not_default('summarize', 'join_back_columns_prefix', join_back_columns_prefix, None)
        fn_mapping = {
            'MIN': 'min',
            'MAX': 'max',
            'MEAN': 'statisticalMoments',
            'MEDIAN': 'tdigest',
            'VAR': 'statisticalMoments',
            'SD': 'statisticalMoments',
            'COUNT': 'count',
            'SUM': 'sum',
            'SKEWNESS': 'statisticalMoments',
            'KURTOSIS': 'statisticalMoments',
            'TOLIST': 'toList',
            'TOPVALUES': 'topValues',
            'BOTTOMVALUES': 'bottomValues',
            'SINGLE': 'single',
            'COMMONPATH': 'commonPath',
        }

        def map_aggregate(summary_column_value):
            return {
                'source_column': summary_column_value.column_id,
                'aggregate': fn_mapping[summary_column_value.summary_function.name],
                'new_column': summary_column_value.summary_column_name,
            }

        return self._add_transformation('summarize', {
                                        'aggregates': [map_aggregate(p) for p in summary_columns] if summary_columns is not None else None,
                                        'group_by': [p for p in group_by_columns] if group_by_columns is not None else None})

    def _get_partition_format(self) -> str:
        def find_extract_from_partition_format_transformation(transformations):
            if transformations:
                for transformation in transformations:
                    name = transformation if isinstance(transformation, str) else next(transformation.keys().__iter__())
                    if name == 'extract_columns_from_partition_format':
                        return transformation['extract_columns_from_partition_format']
            return None

        partition_format_transformation = find_extract_from_partition_format_transformation(self._to_yaml_dict().get('transformations'))
        if partition_format_transformation:
            # meaning this is created with new EnginelessDataflow, so get it from transformations
            return partition_format_transformation['partition_format']
        elif self._dataflow_json is not None:
            # this could be converted from legacy dataflow, in which case transformations have consumed source partition format,
            # so we need to get it from the source json dataflow
            import json
            dflow_object = json.loads(self._dataflow_json)
            blocks = dflow_object['blocks']
            for block in blocks:
                if block['type'] == 'Microsoft.DPrep.AddColumnsFromPartitionFormatBlock':
                    return block['arguments']['partitionFormat']
        return None

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def _with_optimized_datasource(self, common_path, verbose: bool):
        if not self._py_rs_dataflow.is_file_source:
            get_logger().warning("cannot optimize as step type is not one of "
                                 "GetDatastoreFilesBlock, GetFilesBlock, CreateDatasetBlock, "
                                 "and CreateDatasetFilesBlock")

        # return whether it was datastore or not and resource_id
        def _process_path_stream_info(si):
            return (si.handler == 'AmlDatastore', si, si.resource_id)

        path_infos = self._py_rs_dataflow.get_file_sources()
        # convert from input uris, like azureml:// or wasb:// to paths in remote
        input_paths_details = [_process_path_stream_info(PyRsDataflow.parse_uri(next(path_info.values().__iter__()))) for path_info in path_infos]

        # remove leading forward slash to normalize path segments
        common_path_list = common_path.lstrip('/').split('/')

        new_input_infos = []
        for is_datastore, si, input_path in input_paths_details:
            can_be_optimized, new_resource_id = EnginelessDataflow._get_optimal_path(common_path_list, input_path.lstrip('/').split('/'), is_datastore, verbose)
            if can_be_optimized:
                new_input_infos.append(StreamInfo(si.handler, new_resource_id, si.arguments))

        new_dataflow = EnginelessDataflow.from_stream_infos(new_input_infos, force_search=True)

        return new_dataflow + self[1:]

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def _set_column_types(self, type_conversions: List[Tuple[ColumnSelector, TypeConversionInfo]]) -> 'Dataflow':
        def _get_validity_and_type_arguments(converter: TypeConverter) -> Tuple[bool, Optional[Dict[str, Any]]]:
            if isinstance(converter, DateTimeConverter):
                return (converter.formats is not None and len(converter.formats) > 0,  {'formats': converter.formats})
            if isinstance(converter, TypeConverter) and converter.data_type.value == FieldType.DATE.value:
                return (False, None)
            if isinstance(converter, FloatConverter):
                return (True, None)
            if isinstance(converter, StreamInfoConverter) and converter.workspace is not None:
                return (True, {
                    'subscription': converter.workspace.subscription_id,
                    'resource_group': converter.workspace.resource_group,
                    'workspace_name': converter.workspace.name,
                    'escaped': converter.escaped
                })
            return (True, None)

        normalized_type_conversions : Tuple[ColumnSelector, TypeConverter] = []
        columns_to_tranform_decimal_marks: List[str] = []
        # normalize type_conversion argument
        for col, conv_info in type_conversions:
            if not isinstance(conv_info, TypeConverter):
                # if a 2 value tuple and first value is FieldType.Date
                if isinstance(conv_info, tuple) and conv_info[0] == FieldType.DATE and len(conv_info) == 2:
                    converter = DateTimeConverter(conv_info[1] if isinstance(conv_info[1], List)
                                                              else [conv_info[1]])
                elif isinstance(conv_info, tuple) and conv_info[0] == FieldType.DECIMAL and len(conv_info) == 2:
                    converter = FloatConverter(conv_info[1])
                    if converter.decimal_mark.value == DecimalMark.COMMA.value:
                        columns_to_tranform_decimal_marks.append(col)
                elif isinstance(conv_info, tuple) and conv_info[0] == FieldType.STREAM and len(conv_info) == 2:
                    converter = StreamInfoConverter(conv_info[1])
                elif isinstance(conv_info, FieldType) and conv_info.value < FieldType.UNKNOWN.value:
                    converter = TypeConverter(conv_info)
                else:
                    raise ValueError(f'Unexpected conversion info "{conv_info}" for column "{col}"')
            else:
                if isinstance(conv_info, FloatConverter) and conv_info.decimal_mark == ',':
                    columns_to_tranform_decimal_marks.append(col)
                converter = conv_info

            normalized_type_conversions.append((col, converter))

        # construct transformation arguments
        def fieldType_to_string(field_type: FieldType) -> str:
            if field_type == FieldType.DATE:
                return 'datetime'
            if field_type == FieldType.DECIMAL:
                return 'float'
            if field_type == FieldType.BOOLEAN:
                return 'boolean'
            if field_type == FieldType.INTEGER:
                return 'int'
            if field_type == FieldType.STRING:
                return 'string'
            if field_type == FieldType.STREAM:
                return 'stream_info'

            raise ValueError('Unexpected field type: ' + str(field_type))

        columns_needing_inference = []
        arguments = []
        for col, converter in normalized_type_conversions:
            col_conversion : Dict[str, Any] = {'columns': col}
            is_valid, args = _get_validity_and_type_arguments(converter)
            if not is_valid:
                columns_needing_inference.append(col)
                continue
            type = fieldType_to_string(converter.data_type)
            col_conversion['column_type'] = { type: args } if args is not None else type
            arguments.append(col_conversion)

        if len(columns_needing_inference) > 0:
            just_columns_to_infer = self.keep_columns(columns_needing_inference)
            ex = get_rslex_executor()
            error = None
            inference_result = None
            try:
                with tracer.start_as_current_span('EnginelessDataflow.set_columns_types.infer_missing_date_formats', trace.get_current_span()) as span:
                    inference_result = ex.infer_types(just_columns_to_infer._to_yaml_string(), 200, to_dprep_span_context(span.get_context()).span_id)
            except Exception as e:
                error = e
                raise
            finally:
                _log_dataflow_execution_activity(get_logger(),
                                                 activity='set_columns_types.infer_missing_date_formats',
                                                 rslex_failed=error is not None,
                                                 rslex_error=error,
                                                 execution_succeeded=error is None,
                                                 extra_props={'inference_col_count': len(inference_result)} if error is None else None)

            for col, type_inference in inference_result.items():
                col_conversion : Dict[str, Any] = {'columns': col}
                if type_inference.field_type == 'datetime':
                    if len(type_inference.ambiguous_formats) > 0:
                        raise ValueError(f'Ambiguous date formats for column: "{col}", matching formats: {type_inference.ambiguous_formats}. Please specify the format explicitly.')
                    if len(type_inference.datetime_formats) == 0:
                        # no date formats from inference means that the column was already of date type
                        continue
                    col_conversion['column_type'] = { 'datetime': {'formats': type_inference.datetime_formats}}
                else:
                    raise ValueError(f'Unexpected field type for {col}, provided {normalized_type_conversions[col]} but got {type_inference.field_type} during inference. '
                                     'Make sure the provided type has all required arguments so that inference could be avoided.')
                arguments.append(col_conversion)

        # if columns_to_tranform_decimal_marks > 0 then also arguments > 0
        transformed = self._with_columns_to_transform_decimal_marks(columns_to_tranform_decimal_marks)
        return transformed._add_transformation('convert_column_types', arguments) if len(arguments) > 0 else self

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def _with_columns_to_transform_decimal_marks(self, columns: ColumnSelector):
        if len(columns) > 0:
            return self._add_transformation('transform_columns',
                                            {'language': 'Native',
                                             'transformations': [{
                                                 'columns': columns,
                                                 'function_source': '{"r":["Function",[[],{"r":[]},{"r":["Invoke",[{"r":["Identifier","CleanStringNumberTransform"]},[".",false,false]]]}]]}'}]})
        return self

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def _with_partition_size(self, partition_size: int, skip_validation: bool = False) -> 'EnginelessDataflow':
        if not (isinstance(partition_size, int) and partition_size > 0):
            raise ValueError('expect partition_size to be positive int')
        if not skip_validation:
            self._validate_partition_size(partition_size)

        def find_transformation_for_partition_size(transformations):
            if transformations:
                for transformation in transformations:
                    if 'read_delimited' in transformation:
                        return transformation['read_delimited']
                    if 'read_json_lines' in transformation:
                        return transformation['read_json_lines']
                    if 'read_files' in transformation:
                        read_files = transformation['read_files']
                        if read_files['reader'] == 'textLines':
                            return read_files['reader_arguments']
                        raise ValueError('Can only update partition_size if textLines reader is used, found : ' + read_files['reader'])                         
            raise ValueError('Can only update partition_size if `read_delimited` or `read_json_lines` '
                             'are in the EnglinelessDataflow')

        rs_dataflow_yaml = self._to_yaml_dict()
        try:
            transformation = find_transformation_for_partition_size(rs_dataflow_yaml.get('transformations'))

            transformation['partition_size'] = partition_size
            return EnginelessDataflow(rs_dataflow_yaml)
        except ValueError as ve:
            if skip_validation:
                get_logger().warning(f"Ignoring error while updating partition_size: {ve}, as skip_validation is set to True")
                return self
            else:
                raise ve

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def get_profile(self, include_stype_counts: bool = False, number_of_histogram_bins: int = 10) -> DataProfile:
        return self._get_profile(include_stype_counts, number_of_histogram_bins)

    def _get_profile(self,
                     include_stype_counts: bool = False,
                     number_of_histogram_bins: int = 10,
                     include_average_spaces_count: bool = False,
                     include_string_lengths: bool = False) -> DataProfile:
        _warn_if_not_default('get_profile', 'include_stype_counts', include_stype_counts, False)
        _warn_if_not_default('get_profile', 'number_of_histogram_bins', number_of_histogram_bins, 10)
        _warn_if_not_default('get_profile', 'include_average_spaces_count', include_average_spaces_count, False)
        _warn_if_not_default('get_profile', 'include_string_lengths', include_string_lengths, False)

        dataflow = self._add_transformation('summarize_each', {'aggregates': ['min', 'max', 'statisticalMoments', 'tdigest', 'missingAndEmpty', 'valueCountsLimited', 'valueKinds']})
        dataflow = dataflow._add_transformation('add_columns', {
            'language': 'Native',
            'expressions': [
                {
                    'new_column': 'quantiles',
                    'prior_column': 'tdigest',
                    'function_source': '{"r":["Function",[[],{"r":[]},{"r":["Function",[["row"],{"r":["mapFunc",{"r":["Invoke",[{"r":["Identifier","QuantilesFromTDigestOnValue"]},[[0,0.001,0.01,0.05,0.25,0.5,0.75,0.95,0.99,0.999,1]]]]}]},{"r":["Invoke",[{"r":["Identifier","Map"]},[{"r":["RecordField",[{"r":["Identifier","row"]},"tdigest"]]},{"r":["Identifier","mapFunc"]}]]]}]]}]]}'
                },
                {
                    'new_column': 'histogram',
                    'prior_column': 'valueKinds',
                    'function_source': '{"r":["Function",[[],{"r":[]},{"r":["Function",[["row"],{"r":["mapFunc",{"r":["Invoke",[{"r":["Identifier","HistogramFromTDigestOnValue"]},[10]]]}]},{"r":["Invoke",[{"r":["Identifier","Map"]},[{"r":["RecordField",[{"r":["Identifier","row"]},"tdigest"]]},{"r":["Identifier","mapFunc"]}]]]}]]}]]}'
                },
                {
                    'new_column': 'whiskers',
                    'prior_column': 'histogram',
                    'function_source': '{"r":["Function",[[],{"r":[]},{"r":["Function",[["row"],{"r":["mapFunc",{"r":["Identifier","WhiskersFromTDigestOnValue"]}]},{"r":["Invoke",[{"r":["Identifier","Map"]},[{"r":["RecordField",[{"r":["Identifier","row"]},"tdigest"]]},{"r":["Identifier","mapFunc"]}]]]}]]}]]}'
                }
            ]})

        dataflow = dataflow.drop_columns('tdigest')

        return DataProfile._from_rslex_execution(dataflow._to_yaml_string())

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_path(self, activity, source, stream_column=_DEFAULT_STREAM_INFO_COLUMN):
        dataflow, portable_path = self._add_portable_path_column(stream_column)
        dataflow = dataflow._copy_and_update_metadata(activity, source)
        records = dataflow._to_pyrecords(use_new_flow=True)
        return [r[portable_path] for r in records]

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def download(self, target_path, source, stream_column=_DEFAULT_STREAM_INFO_COLUMN):
        from .datasources import LocalFileOutput
        base_path = LocalFileOutput(target_path)
        dataflow, portable_path = self._add_portable_path_column(stream_column)
        dataflow = dataflow.write_streams(
            streams_column=stream_column,
            base_path=base_path,
            file_names_column=portable_path)

        dataflow = dataflow._copy_and_update_metadata('download', source)
        return dataflow._to_pyrecords(use_new_flow=True)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def head(self, count: int = 10) -> 'pandas.DataFrame':
        """
        Pulls the number of records specified from the top of this Dataflow and returns them as a `Link pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_.

        :param count: The number of records to pull. 10 is default.
        :return: A Pandas `Link pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_.
        """
        return self.take(count).to_pandas_dataframe(extended_types=True)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def get_partition_count(self) -> int:
        ex = get_rslex_executor()
        error = None
        partition_count = None
        try:
            with tracer.start_as_current_span('EnginelessDataflow.get_partition_count', trace.get_current_span()) as span:
                partition_count = ex.get_partition_count(self._to_yaml_string(), to_dprep_span_context(span.get_context()).span_id)
                return partition_count
        except BaseException as ex:
            error = ex
            raise
        finally:
            _log_dataflow_execution_activity(
                get_logger(),
                activity='get_partition_count',
                rslex_failed=error is not None,
                rslex_error=error,
                execution_succeeded=error is None,
                extra_props={'partition_count': partition_count} if error is None else None)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def get_partition_info(self) -> Tuple[int, List[Tuple['StreamInfo', int]]]:
        """
        Calculates the partitions for the current Dataflow and returns total count and a list of partition source and partition count in that source.
        Partitioning is guaranteed to be stable for a specific execution mode.
        Partition source list would be a tuple of (StreamInfo, count) if it can be tied directly to a file or None in other cases.

        :return: The count of partitions.
        """
        ex = get_rslex_executor()
        error = None
        partition_count = None
        try:
            with tracer.start_as_current_span('EnginelessDataflow.get_partition_info', trace.get_current_span()) as span:
                (num_partitions, partitions_streams_and_counts) = ex.get_partition_info(self._to_yaml_string(), to_dprep_span_context(span.get_context()).span_id)
                partition_count = num_partitions
                return (num_partitions, partitions_streams_and_counts)
        except BaseException as ex:
            error = ex
            raise
        finally:
            _log_dataflow_execution_activity(
                get_logger(),
                activity='get_partition_info',
                rslex_failed=error is not None,
                rslex_error=error,
                execution_succeeded=error is None,
                extra_props={'partition_count': partition_count} if error is None else None)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def run_local(self) -> None:
        """
        Runs the current Dataflow using the local execution runtime.
        """
        parent = trace.get_current_span()
        with tracer.start_as_current_span('Dataflow.run_local', parent) as span:
            _execute('Dataflow.run_local',
                     self._to_yaml_string(),
                     span_context=to_dprep_span_context(span.get_context()))

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def verify_has_data(self, verify_columns=False):
        """
        Verifies that this Dataflow would produce records if executed. An exception will be thrown otherwise.
        """
        with tracer.start_as_current_span('EnginelessDataflow.verify_has_data', trace.get_current_span()):
              records = self.take(1)._to_pyrecords(use_new_flow=True)
              if len(records) == 0:
                raise DataflowValidationError("The Dataflow produced no records.")
              if verify_columns:
                if len(records[0]) == 0:
                    raise DataflowValidationError("The Dataflow produced no columns.")

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def _to_pyrecords(self, use_new_flow=True):
        if use_new_flow:
            ex = get_rslex_executor()
            error = None
            record_count = None
            try:
                with tracer.start_as_current_span('EnginelessDataflow._to_pyrecords', trace.get_current_span()) as span:
                    records = ex.to_pyrecords(self._to_yaml_string(), to_dprep_span_context(span.get_context()).span_id)
                    record_count = len(records)
                    return records
            except Exception as e:
                error = e
                raise
            finally:
                _log_dataflow_execution_activity(
                    get_logger(),
                    activity='_to_pyrecords',
                    rslex_failed=error is not None,
                    rslex_error=error,
                    execution_succeeded=error is None,
                    extra_props={'record_count': record_count} if error is None else None)
        else:
            return _execute('EnginelessDataflow._to_pyrecords',
                            self._to_yaml_string(),
                            force_preppy=True,
                            convert_preppy_to_pyrecords=True)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def select_partitions(self, partition_indices: List[int]) -> 'EnginelessDataflow':
        """
        Selects specific partitions from the data, dropping the rest.

        :return: The modified Dataflow.
        """
        return self._add_transformation('select_partitions', partition_indices)

    def _partition_to_pandas_dataframe(self,
                                       i: int,
                                       extended_types: bool,
                                       nulls_as_nan: bool,
                                       on_error: str,
                                       out_of_range_datetime: str) -> 'pandas.DataFrame':
        return self.select_partitions([i]).to_pandas_dataframe(extended_types=extended_types,
                                                               nulls_as_nan=nulls_as_nan,
                                                               on_error=on_error,
                                                               out_of_range_datetime=out_of_range_datetime)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_dask_dataframe(self,
                          sample_size: int = 10000,
                          dtypes: dict = None,
                          extended_types: bool = False,
                          nulls_as_nan: bool = True,
                          on_error: str = 'null',
                          out_of_range_datetime: str = 'null'):
        """
        Returns a Dask DataFrame that can lazily read the data in the Dataflow.

        .. remarks::
            Dask DataFrames allow for parallel and lazy processing of data by splitting the data into multiple
                partitions. Because Dask DataFrames don't actually read any data until a computation is requested,
                it is necessary to determine what the schema and types of the data will be ahead of time. This is done
                by reading a specific number of records from the Dataflow (specified by the `sample_size` parameter).
                However, it is possible for these initial records to have incomplete information. In those cases, it is
                possible to explicitly specify the expected columns and their types by providing a dict of the shape
                `{column_name: dtype}` in the `dtypes` parameter.

        :param sample_size: The number of records to read to determine schema and types.
        :param dtypes: An optional dict specifying the expected columns and their dtypes.
            `sample_size` is ignored if this is provided.
        :param extended_types: Whether to keep extended DataPrep types such as DataPrepError in the DataFrame. If False,
            these values will be replaced with None.
        :param nulls_as_nan: Whether to interpret nulls (or missing values) in number typed columns as NaN. This is
            done by pandas for performance reasons; it can result in a loss of fidelity in the data.
        :param on_error: How to handle any error values in the Dataflow, such as those produced by an error while parsing values.
            Valid values are 'null' which replaces them with null; and 'fail' which will result in an exception.
        :param out_of_range_datetime: How to handle date-time values that are outside the range supported by Pandas.
            Valid values are 'null' which replaces them with null; and 'fail' which will result in an exception.
        :return: A Dask DataFrame.
        """
        from ._dask_helper import _ensure_dask
        _ensure_dask()

        from ._pandas_helper import have_pandas
        from .errorhandlers import PandasImportError
        if not have_pandas():
            raise PandasImportError()

        import dask.dataframe as dd
        from dask.delayed import delayed
        import pandas

        def is_optimization_supported(source_dataflow):
            try:
                loader = source_dataflow[0]._to_yaml_dict()
            except Exception:
                return False

            if 'DATASET_DISABLE_DASK_OPTIMIZATION' in os.environ and os.environ['DATASET_DISABLE_DASK_OPTIMIZATION'] == 'True':
                _LoggerFactory.trace(get_logger(), '[to_dask_dataframe()] Optimization is disabled with env var: DATASET_DISABLE_DASK_OPTIMIZATION')
                import warnings
                warnings.warn("Experimental large dataset performance optimization is disabled with env var: DATASET_DISABLE_DASK_OPTIMIZATION")
                return False
            if "query_source" in loader:
                _LoggerFactory.trace(get_logger(), '[to_dask_dataframe()] Optimization of the dataflow is not supported for the source type: query_source')
                return False
            return True

        partition_count = None
        sources = None
        if is_optimization_supported(self):
            try:
                (partition_count, sources) = self.get_partition_info()
            except Exception as e:
                _LoggerFactory.trace_error(get_logger(), '[to_dask_dataframe()] Failed to get partition info: {}'.format(e))
                pass
        #  If the optimization is not supported or getting partition_info has failed, we will fallback to getting partition count.
        if partition_count is None:
            partition_count = self.get_partition_count()

        if partition_count <= 0:
            return dd.from_pandas(pandas.DataFrame(), chunksize=1)

        field_type_to_dtypes = {
            FieldType.STRING: object,
            FieldType.BOOLEAN: bool,
            FieldType.INTEGER: int,
            FieldType.DECIMAL: float,
            FieldType.DATE: 'datetime64[ns]',
            FieldType.UNKNOWN: object,
            FieldType.ERROR: object,
            FieldType.NULL: object,
            FieldType.DATAROW: object,
            FieldType.LIST: object,
            FieldType.STREAM: object
        }

        dataflow_dtypes = dtypes or {k: field_type_to_dtypes[v] for k, v in self.take(sample_size).dtypes.items()}
        delayed_functions = None

        if sources is not None:
            try:
                import warnings
                warnings.warn("Using experimental performance optimization for large datasets. To disable set 'DATASET_DISABLE_DASK_OPTIMIZATION' env var to True.")
                delayed_functions = [delayed(self._optimized_get_pandas_dataframe_from_partition_info)(source, index_in_source, extended_types, nulls_as_nan, on_error, out_of_range_datetime)
                                        for source, count_in_source in sources
                                        for index_in_source in range(count_in_source)]
                _LoggerFactory.trace(get_logger(), '[to_dask_dataframe()] Using optimized pandas dataframe conversion for {} partitions'.format(len(delayed_functions)))
            except Exception as e:
                _LoggerFactory.trace_error(get_logger(), '[to_dask_dataframe()] Failed to use optimized pandas dataframe conversion. Falling back to non-optimized conversion. Error: {}'.format(e))

        # either no sources or failed to use optimized conversion, fall back to non-optimized conversion
        if delayed_functions is None:
            delayed_functions = [delayed(self._partition_to_pandas_dataframe)(i, extended_types, nulls_as_nan, on_error, out_of_range_datetime) for i in range(0, partition_count)]

        ddf = dd.from_delayed(delayed_functions, meta=dataflow_dtypes)
        return ddf

    # noinspection PyUnresolvedReferences
    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_spark_dataframe(self, spark_session: 'pyspark.sql.SparkSession' = None) -> 'pyspark.sql.DataFrame':
        """
        Creates a Spark `Link pyspark.sql.DataFrame <https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame>`_ that can execute the transformation pipeline defined by this Dataflow.

        .. remarks::

            Since Dataflows do not require a fixed, tabular schema but Spark Dataframes do, an implicit tabularization
                step will be executed as part of this action. The resulting schema will be the union of the schemas of all
                records produced by this Dataflow. This tabularization step will result in a pull of the data.

            .. note::

                The Spark Dataframe returned is only an execution plan and doesn't actually contain any data, since Spark Dataframes are also lazily evaluated.

        :return: A Spark `Link pyspark.sql.DataFrame <https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame>`_.
        """
        from ._spark_helper import read_spark_dataframe

        if not spark_session:
            from pyspark.sql import SparkSession
            try:
                spark_session = SparkSession.getActiveSession()
            except AttributeError:
                spark_session = SparkSession.builder.getOrCreate()

        if not spark_session:
            raise EnvironmentError("Failed to acquire a spark session. Either populate the spark_session parameter, or invoke this method from a thread that has an active spark session.")

        return read_spark_dataframe(self._to_yaml_string(), spark_session)

    # noinspection PyUnresolvedReferences
    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_pandas_dataframe(self,
                            extended_types: bool = False,
                            nulls_as_nan: bool = True,
                            on_error: str = 'null',
                            out_of_range_datetime: str = 'null') -> 'pandas.DataFrame':
        """
        Pulls all of the data and returns it as a Pandas `Link pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_.

        .. remarks::

            This method will load all the data returned by this Dataflow into memory.

            Since Dataflows do not require a fixed, tabular schema but Pandas DataFrames do, an implicit tabularization
                step will be executed as part of this action. The resulting schema will be the union of the schemas of all
                records produced by this Dataflow.

        :param extended_types: Whether to keep extended DataPrep types such as DataPrepError in the DataFrame. If False,
            these values will be replaced with None.
        :param nulls_as_nan: Whether to interpret nulls (or missing values) in number typed columns as NaN. This is
            done by pandas for performance reasons; it can result in a loss of fidelity in the data.
        :param on_error: How to handle any error values in the Dataflow, such as those produced by an error while parsing values.
            Valid values are 'null' which replaces them with null; and 'fail' which will result in an exception.
        :param out_of_range_datetime: How to handle date-time values that are outside the range supported by Pandas.
            Valid values are 'null' which replaces them with null; and 'fail' which will result in an exception.
        :return: A Pandas `Link pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_.
        """
        with tracer.start_as_current_span('Dataflow.to_pandas_dataframe', trace.get_current_span()) as span:
            span_context = to_dprep_span_context(span.get_context())
            return get_dataframe_reader().to_pandas_dataframe(self._to_yaml_string(),
                                                              extended_types,
                                                              nulls_as_nan,
                                                              on_error,
                                                              out_of_range_datetime,
                                                              span_context)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def mount(self, mount_point: str, options: 'MountOptions', files_column=None, client_id=None, identity_endpoint_type=None):
        from azureml.dataprep.fuse.dprepfuse import MountOptions
        from azureml.dataprep.rslex import PyMountOptions
        from azureml.dataprep.api._rslex_executor import ensure_rslex_environment
        
        ensure_rslex_environment()

        options = options or MountOptions()
        rs_mount_options = PyMountOptions(
            options.max_size,
            options.data_dir,
            options.free_space_required,
            options.allow_other,
            options.read_only,
            options.default_permission,
            options.create_destination)

        return self._py_rs_dataflow.mount(mount_point, rs_mount_options, files_column, client_id, identity_endpoint_type)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_json(self):
        import json
        if self._dataflow_json is not None:
            # this means that the dataflow was created in legacy code and was converted to RsDataflow in the service.
            # we need to return the original json to avoid redundant conversions in the service
            dflow_obj = json.loads(self._dataflow_json)
            if 'meta' not in dflow_obj:
                dflow_obj['meta'] = {}
            dflow_obj['meta']['RsDataflow'] = self._to_yaml_string()
            return json.dumps(dflow_obj)

        # this means that either the dataflow was created as Engineless or there were transformations added to it.
        # At this point we will need service to convert it back to v1 dataflow if needed.
        dflow_obj = {'blocks':[], 'meta': {'RsDataflow': self._to_yaml_string()}}
        return json.dumps(dflow_obj)

    @staticmethod
    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def from_json(dataflow_json: str) -> 'EnginelessDataflow':
        import json
        dflow_obj = json.loads(dataflow_json)
        if 'meta' not in dflow_obj:
            raise ValueError("Dataflow JSON must contain 'meta' key")

        meta = dflow_obj['meta']
        if 'RsDataflow' not in meta:
            raise ValueError("Dataflow JSON 'meta' key must contain 'RsDataflow' key")
        has_blocks = 'blocks' in dflow_obj and len(dflow_obj['blocks']) > 0
        return EnginelessDataflow(meta['RsDataflow'], dataflow_json if has_blocks else None)

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_datetime(self,
                    columns: MultiColumnSelection,
                    date_time_formats: Optional[List[str]] = None,
                    date_constant: Optional[str] = None) -> 'EnginelessDataflow':
        """
        Converts the values in the specified columns to DateTimes.

        :param columns: The source columns.
        :param date_time_formats: The formats to use to parse the values. If none are provided, a partial scan of the
            data will be performed to derive one.
        :param date_constant: If the column contains only time values, a date to apply to the resulting DateTime.
        :return: The modified Dataflow.
        """
        columns = _column_selection_to_py_rs_dataflow_selector(columns)
        if date_time_formats is None or len(date_time_formats) == 0:
            return self._set_column_types([(columns, FieldType.DATE)])
        args = {'formats': date_time_formats}
        if date_constant is not None:
            args['date_constant'] = date_constant
        return self._add_transformation('convert_column_types', [{'columns': columns, 'column_type': {'datetime': args}}])

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_number(self,
                  columns: MultiColumnSelection,
                  decimal_point: DecimalMark = DecimalMark.DOT) -> 'EnginelessDataflow':
        """
        Converts the values in the specified columns to floating point numbers.

        :param columns: The source columns.
        :param decimal_point: The symbol to use as the decimal mark.
        :return: The modified Dataflow.
        """
        columns = _column_selection_to_py_rs_dataflow_selector(columns)
        if decimal_point == DecimalMark.DOT:
            transformed = self
        else: # decimal_point == DecimalMark.COMMA:
            transformed = self._with_columns_to_transform_decimal_marks(columns)

        return transformed._add_transformation('convert_column_types', [{
                                               'columns': columns,
                                               'column_type': 'float'
                                               }])

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_bool(self,
                columns: MultiColumnSelection,
                true_values: List[str],
                false_values: List[str],
                mismatch_as: MismatchAsOption = MismatchAsOption.ASERROR) -> 'EnginelessDataflow':
        """
        Converts the values in the specified columns to booleans.

        :param columns: The source columns.
        :param true_values: The values to treat as true.
        :param false_values: The values to treat as false.
        :param mismatch_as: How to treat values that don't match the values in the true or false values lists.
        :return: The modified Dataflow.
        """
        if mismatch_as == MismatchAsOption.ASERROR:
            mismatch_as = 'error'
        elif mismatch_as == MismatchAsOption.ASFALSE:
            mismatch_as = 'false'
        else:
            mismatch_as = 'true'

        return self._add_transformation('convert_column_types', [{
                                        'columns': _column_selection_to_py_rs_dataflow_selector(columns),
                                        'column_type': {
                                            'boolean':{
                                                'true_values': true_values,
                                                'false_values': false_values,
                                                'mismatch_as': mismatch_as
                                                }}}])

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_string(self,
                  columns: MultiColumnSelection) -> 'EnginelessDataflow':
        """
        Converts the values in the specified columns to strings.

        :param columns: The source columns.
        :return: The modified Dataflow.
        """
        return self._add_transformation('convert_column_types', [{
                                        'columns': _column_selection_to_py_rs_dataflow_selector(columns),
                                        'column_type': 'string'
                                    }])

    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def to_long(self,
                columns: MultiColumnSelection) -> 'EnginelessDataflow':
        """
        Converts the values in the specified columns to 64 bit integers.

        :param columns: The source columns.
        :return: The modified Dataflow.
        """
        return self._add_transformation('convert_column_types', [{
                                        'columns': _column_selection_to_py_rs_dataflow_selector(columns),
                                        'column_type': 'int'
                                    }])

    @property
    def dtypes(self) -> Dict[str, FieldType]:
        """
        Gets column data types for the current dataset by calling :meth:`get_profile` and extracting just dtypes from
            the resulting DataProfile.

        .. note:

            This will trigger a data profile calculation.
            To avoid calculating profile multiple times, get the profile first by calling \
            `get_profile()` and then inspect it.

        :return: A dictionary, where key is the column name and value is :class:`azureml.dataprep.FieldType`.
        """
        profile = self._get_profile()
        return profile.dtypes

    @property
    def row_count(self) -> int:
        """
        Count of rows in this Dataflow.

        .. note::

            This will trigger a data profile calculation. To avoid calculating profile multiple times, get the profile first by calling \
            `get_profile()` and then inspect it.

        .. note::

            If current Dataflow contains `take_sample` step or 'take' step, this will return number of rows in the \
            subset defined by those steps.

        :return: Count of rows.
        :rtype: int
        """
        profile = self._get_profile()
        return profile.row_count

    @property
    def shape(self) -> Tuple[int,int]:
        """
        Shape of the data produced by the Dataflow.

        .. note::

            This will trigger a data profile calculation. To avoid calculating profile multiple times, get the profile first by calling \
            `get_profile()` and then inspect it.

        .. note::

            If current Dataflow contains `take_sample` step or 'take' step, this will return number of rows in the \
            subset defined by those steps.

        :return: Tuple of row count and column count.
        """
        profile = self._get_profile()
        return profile.shape

    @staticmethod
    def _paths_to_uris(paths: FilePath, force_file: bool = False) -> List[Dict[str, str]]:
        # handle case of datastore paths
        if _is_datapath(paths):
            return [{'pattern' : uri} for uri in file_datastores_to_uris([paths])]
        if _is_datapaths(paths):
            return [{'pattern' : uri} for uri in file_datastores_to_uris(paths)]

        # handle FileDataSource by extracting path for future processing
        if isinstance(paths, FileDataSource):
            paths = [resource_detail.to_pod()['path'] for resource_detail in paths.underlying_value.resource_details]
        if not isinstance(paths, list):
            paths = [paths]
        return [{'pattern' if can_search and not force_file else 'file' : uri} for can_search, uri in process_uris(paths)]

    @staticmethod
    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def from_paths(paths: FilePath, force_file: bool = False) -> 'EnginelessDataflow':
        uri_paths = EnginelessDataflow._paths_to_uris(paths, force_file)
        # this will ensure rslex is initialized
        get_rslex_executor()
        rs_df = PyRsDataflow.from_paths(uri_paths)
        return EnginelessDataflow(rs_df)

    @staticmethod
    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def from_stream_infos(stream_infos: List[StreamInfo], force_search=False) -> 'EnginelessDataflow':
    # if force_search is True, then streamInfo is considered unresolved and pattern input would be used
    # so that path expansion can happen, otherwise, file input would be used and streamInfo would be used directly to access data
        paths = [{'pattern' if force_search else 'file' : si.to_pod()} for si in stream_infos]
        # this will ensure rslex is initialized
        get_rslex_executor()
        rs_df = PyRsDataflow.from_paths(paths)
        return EnginelessDataflow(rs_df)

    @staticmethod
    @track(get_logger, custom_dimensions={'app_name': _APP_NAME})
    def from_query_source(handler, query, handler_arguments) -> 'EnginelessDataflow':
        arguments = {'handler': handler, 'query': query, 'handler_arguments': handler_arguments}
        # this will ensure rslex is initialized
        get_rslex_executor()
        rs_df = PyRsDataflow.from_query_source(arguments)
        return EnginelessDataflow(rs_df)

def _column_selection_to_py_rs_dataflow_selector(columns: MultiColumnSelection) -> Union[str, List[str], Dict[str, Any]]:
    if isinstance(columns, str):
        return columns
    if isinstance(columns, (list, set)):
        if not all(isinstance(column_selection, str) for column_selection in columns):
            raise ValueError('Unsupported value for column selection.')
        return list(columns)
    if isinstance(columns, ColumnSelector):
        # RSlex any match in the string
        pattern = columns.term
        if columns.ignore_case:
            # TODO this needs to be fixed in rslex
            pattern = f'(?i)({pattern})'
        if columns.match_whole_word:
            pattern = '^' + pattern + '$'
            return {'pattern': pattern,
                    'ignore_case': False}
        return {'pattern': pattern,
                'invert': columns.invert,
                'ignore_case': False}

    raise ValueError('Unsupported value for column selection.')
