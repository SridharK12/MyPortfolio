# Copyright (c) Microsoft Corporation. All rights reserved.
"""Contains classes for interactively building transformation steps for data preparation in Azure Machine Learning.
"""
from .engineapi.typedefinitions import FileEncoding, SummaryFunction

from ._loggerfactory import _LoggerFactory, _log_not_supported_api_usage_and_raise, trace
from ... import dataprep
from typing import List, Dict, Any, TypeVar, Optional

logger = None
tracer = trace.get_tracer(__name__)

def get_logger():
    global logger
    if logger is not None:
        return logger

    logger = _LoggerFactory.get_logger("Dataflow.Builders")
    return logger

# Used in EnginelessBuilders do not remove
class InferenceArguments:
    """
    Class to control data type inference behavior.

    :param day_first: If set to True, inference will choose date formats where day comes before month.
    :type day_first: bool
    """
    def __init__(self, day_first: bool):
        self.day_first = day_first


class ColumnTypesBuilder:
    """
    (DEPRECATED)
    Interactive object that can be used to infer column types and type conversion attributes.
    """
    def __init__(self, dataflow: 'dataprep.Dataflow', engine_api: 'EngineAPI'):
        _log_not_supported_api_usage_and_raise(get_logger(), "ColumnTypesBuilder with engine_api", "Use EnginelessDataflow and EnginelessBuilder instead")


class FileFormatArguments:
    """
    (DEPRECATED)
    Defines and stores the arguments which can affect learning on a 'FileFormatBuilder'.
    """

    def __init__(self, all_files: bool):
        """
        :param all_files: Specifies whether learning will occur on all files (True) or just the first one (False).
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "FileFormatArguments", "Update your code to use file format specific reader instead.")


class FileFormatBuilder:
    """
    (DEPRECATED)
    Interactive object that can learn the file format and properties required to read a given file.

    .. remarks::

        This Builder is generally used on a Dataflow which has had a 'get_files' step applied to it. After the path(s)
            to files have been resolved, the appropriate method of interpreting those files can be learned and modified
            using this Builder.

    :var file_format: Result of file format detection.
    """

    def __init__(self, dataflow: 'dataprep.Dataflow', engine_api):
        _log_not_supported_api_usage_and_raise(get_logger(), "FileFormatBuilder", "Update your code to use file format specific reader instead.")


class JsonTableBuilder:
    """
    (DEPRECATED)
    Interactive object that can learn program for table extraction from json document.

    .. remarks::

        This Builder is generally used on a Dataflow which has had a 'get_files' step applied to it. After the path(s)
            to files have been resolved, if files are json files, a program to extract data into tabular form can be learned
            using this Builder.
    """

    def __init__(self,
                 dataflow: 'dataprep.Dataflow',
                 engine_api,
                 flatten_nested_arrays: bool = False,
                 encoding = None):
        _log_not_supported_api_usage_and_raise(get_logger(), "FileFormatBuilder", "Deprecated, downgrade to previous version of azureml-dataprep package.")



# noinspection PyUnresolvedReferences
SourceData = TypeVar('SourceData', Dict[str, str], 'pandas.Series')


class DeriveColumnByExampleBuilder:
    """
    (DEPRECATED)
    Interactive object that can be used to learn program for deriving a column based on a set of source columns and
        examples.
    """

    def __init__(self,
                 dataflow: 'dataprep.Dataflow',
                 engine_api: 'EngineAPI',
                 source_columns: List[str],
                 new_column_name: str):
        _log_not_supported_api_usage_and_raise(get_logger(), "DeriveColumnByExampleBuilder", "Deprecated, downgrade to previous version of azureml-dataprep package.")


class PivotBuilder:
    """
    (DEPRECATED)
    Interactive object that can be used to generate pivoted columns from the selected pivot columns.

    .. remarks::

        This Builder allows for generation, modification and preview of pivoted columns.
    """

    def __init__(self,
                 dataflow: 'dataprep.Dataflow',
                 engine_api: 'EngineAPI',
                 columns_to_pivot: List[str],
                 value_column: str,
                 summary_function: SummaryFunction = None,
                 group_by_columns: List[str] = None,
                 null_value_replacement: str = None,
                 error_value_replacement: str = None):
        _log_not_supported_api_usage_and_raise(get_logger(), "PivotBuilder", "Deprecated, downgrade to previous version of azureml-dataprep package.")


class SplitColumnByExampleBuilder:
    """
    (DEPRECATED)
    Interactive object that can be used to learn program for splitting a column based into a set of columns based on
        provided examples.
    """

    def __init__(self,
                 dataflow: 'dataprep.Dataflow',
                 engine_api: 'EngineAPI',
                 source_column: str,
                 keep_delimiters: bool = False,
                 delimiters: List[str] = None):
        _log_not_supported_api_usage_and_raise(get_logger(), "SplitColumnByExampleBuilder", "Deprecated, downgrade to previous version of azureml-dataprep package.")


class ImputeColumnArguments:
    """
    (DEPRECATED)
    Defines and stores the arguments which can affect learning on a 'ImputeMissingValuesBuilder'.

    :var column_id: Column to impute.
    :var impute_function: The function to calculate the value to impute missing.
    :var custom_impute_value: The custom value used to impute missing.
    :var string_missing_option: The option to specify string values to be considered as missing.
    """

    def __init__(self,
                 column_id: str,
                 impute_function = None,
                 custom_impute_value: Optional[Any] = None,
                 string_missing_option = None):
        _log_not_supported_api_usage_and_raise(get_logger(), "ImputeColumnArguments", "Deprecated, downgrade to previous version of azureml-dataprep package.")


class ImputeMissingValuesBuilder:
    """
    (DEPRECATED)
    Interactive object that can be used to learn a fixed program that imputes missing values in specified columns.
    """

    def __init__(self,
                 dataflow: 'dataprep.Dataflow',
                 engine_api: 'EngineAPI',
                 impute_columns = None,
                 group_by_columns: Optional[List[str]] = None):
        _log_not_supported_api_usage_and_raise(get_logger(), "ImputeMissingValuesBuilder", "Deprecated, downgrade to previous version of azureml-dataprep package.")


class Builders:
    """
    Exposes all available builders for a given Dataflow.
    """
    def __init__(self, dataflow: 'dataprep.Dataflow', engine_api: 'EngineAPI'):
        self._dataflow = dataflow
        self._engine_api = engine_api

    def detect_file_format(self) -> FileFormatBuilder:
        """
        Constructs an instance of :class:`FileFormatBuilder`.
        """
        return FileFormatBuilder(self._dataflow, self._engine_api)

    def set_column_types(self) -> ColumnTypesBuilder:
        """
        Constructs an instance of :class:`ColumnTypesBuilder`.
        """
        return ColumnTypesBuilder(self._dataflow, self._engine_api)

    def extract_table_from_json(self, encoding: FileEncoding = FileEncoding.UTF8) -> JsonTableBuilder:
        """
        Constructs an instance of :class:`JsonTableBuilder`.
        """
        return JsonTableBuilder(self._dataflow, self._engine_api, encoding=encoding)

    def derive_column_by_example(self, source_columns: List[str], new_column_name: str) -> DeriveColumnByExampleBuilder:
        """
        Constructs an instance of :class:`DeriveColumnByExampleBuilder`.
        """
        return DeriveColumnByExampleBuilder(self._dataflow, self._engine_api, source_columns, new_column_name)

    def pivot(self,
              columns_to_pivot: List[str],
              value_column: str,
              summary_function: SummaryFunction = None,
              group_by_columns: List[str] = None,
              null_value_replacement: str = None,
              error_value_replacement: str = None) -> PivotBuilder:
        """
        Constructs an instance of :class:`PivotBuilder`.
        """
        return PivotBuilder(self._dataflow,
                            self._engine_api,
                            columns_to_pivot,
                            value_column,
                            summary_function,
                            group_by_columns,
                            null_value_replacement,
                            error_value_replacement)

    def split_column_by_example(self,
                                source_column: str,
                                keep_delimiters: bool = False,
                                delimiters: List[str] = None) -> SplitColumnByExampleBuilder:
        """
        Constructs an instance of :class:`SplitColumnByExampleBuilder`.
        """
        return SplitColumnByExampleBuilder(self._dataflow,
                                           self._engine_api,
                                           source_column,
                                           keep_delimiters,
                                           delimiters)

    def impute_missing_values(self,
                              impute_columns: List[ImputeColumnArguments] = None,
                              group_by_columns: Optional[List[str]] = None) -> ImputeMissingValuesBuilder:
        """
        Constructs an instance of :class:`ImputeMissingValuesBuilder`.
        """
        return ImputeMissingValuesBuilder(self._dataflow,
                                          self._engine_api,
                                          impute_columns,
                                          group_by_columns)
