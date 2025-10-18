# Copyright (c) Microsoft Corporation. All rights reserved.
from .errorhandlers import DataPrepException
from .datasources import MSSQLDataSource, PostgreSQLDataSource, FileOutput, FileDataSource
from .builders import Builders
from ._loggerfactory import _LoggerFactory, _log_not_supported_api_usage_and_raise, track, trace
from typing import List, Dict, Tuple, TypeVar, Optional, Any
import datetime
from enum import Enum

FilePath = TypeVar('FilePath', FileDataSource, str, List[str])
DatabaseSource = TypeVar('DatabaseSource', MSSQLDataSource, PostgreSQLDataSource)
DataflowReference = TypeVar('DataflowReference', 'Dataflow', 'Dataset', 'DatasetDefinition')
DatabaseDestination = TypeVar('DatabaseDestination')

logger = None
tracer = trace.get_tracer(__name__)
_DEFAULT_STREAM_INFO_COLUMN = 'Path'
_PORTABLE_PATH_COLUMN_NAME = 'PortablePath'

def get_logger():
    global logger
    if logger is not None:
        return logger

    logger = _LoggerFactory.get_logger("Dataflow")
    return logger


class DataflowValidationError(DataPrepException):
    def __init__(self, message: str):
        super().__init__(message, "Empty", message)


# >>> BEGIN GENERATED CLASSES
class SummaryFunction(Enum):
    """
    Enum SummaryFunction.

    .. remarks::

        Possible enum values are:
        - MIN: Minimum value.
        - MAX: Maximum value.
        - MEAN: Mean value.
        - MEDIAN: Median value.
        - VAR: Variance value.
        - SD: Standard deviance value.
        - COUNT: Total count.
        - SUM: Sum of all value.
        - SKEWNESS: Skewness value.
        - KURTOSIS: Kurtosis value.
        - TOLIST: Aggregate values into a list.
        - TOPVALUES: Top values.
        - BOTTOMVALUES: Bottom values.
        - SINGLE: Single value.
        - COMMONPATH: Common Path.
    """
    MIN = 0
    MAX = 1
    MEAN = 2
    MEDIAN = 3
    VAR = 4
    SD = 5
    COUNT = 8
    SUM = 11
    SKEWNESS = 18
    KURTOSIS = 19
    TOLIST = 25
    TOPVALUES = 30
    BOTTOMVALUES = 31
    SINGLE = 32
    COMMONPATH = 33


class MismatchAsOption(Enum):
    """
    Enum MismatchAsOption.

    .. remarks::

        Possible enum values are:
        - ASTRUE: Mismatch value as true.
        - ASFALSE: Mismatch value as false.
        - ASERROR: Mismatch value as error.
    """
    ASTRUE = 0
    ASFALSE = 1
    ASERROR = 2


class TrimType(Enum):
    """
    Enum TrimType.

    .. remarks::

        Possible enum values are:
        - WHITESPACE: Trim white space.
        - CUSTOM: Trim custom characters.
    """
    WHITESPACE = 0
    CUSTOM = 1


class DecimalMark(Enum):
    """
    Enum DecimalMark.

    .. remarks::

        Possible enum values are:
        - DOT: Using dot as decimal mark.
        - COMMA: Using comma as decimal mark.
    """
    DOT = 0
    COMMA = 1


class JoinType(Enum):
    """
    Describes different possible types of join.

    .. remarks::

        Possible enum values are:
        - MATCH: Only records with matching join keys will be returned.
        - INNER: Only records with matching join keys will be returned.
        - UNMATCHLEFT: Records from left data set that did not match with anything.
        - LEFTANTI: Records from left data set that did not match with anything.
        - LEFTOUTER: All records from left data set and only matching records from the right data set.
        - UNMATCHRIGHT: Records from right data set that did not match with anything.
        - RIGHTANTI: Records from right data set that did not match with anything.
        - RIGHTOUTER: All records from right data set and only matching records from the right data set.
        - FULLANTI: Only unmatched records from left and right data sets.
        - FULL: All records from both left and right data sets.
    """
    NONE = 0
    MATCH = 2
    INNER = 2
    UNMATCHLEFT = 4
    LEFTANTI = 4
    LEFTOUTER = 6
    UNMATCHRIGHT = 8
    RIGHTANTI = 8
    RIGHTOUTER = 10
    FULLANTI = 12
    FULL = 14


class SkipMode(Enum):
    """
    Defines a strategy to skip rows when reading files

    .. remarks::

        Possible enum values are:
        - NONE: Don't skip any rows.
        - UNGROUPED: Skip rows from the first file only.
        - FIRSTFILE: Skip rows from the first file only.
        - GROUPED: Skip rows from every file.
        - ALLFILES: Skip rows from every file.
    """
    NONE = 0
    UNGROUPED = 1
    FIRSTFILE = 1
    GROUPED = 2
    ALLFILES = 2


class PromoteHeadersMode(Enum):
    """
    Defines strategy to promote headers when reading files

    .. remarks::

        Possible enum values are:
        - NONE: Do not promote headers. Use when file(s) has no headers.
        - UNGROUPED: Promote headers from the first file. All subsequent files are considered to be just data.
        - FIRSTFILE: Promote headers from the first file. All subsequent files are considered to be just data.
        - GROUPED: Promote headers from every file. Allows to read files with inconsistent schema.
        - ALLFILES: Promote headers from every file. Allows to read files with inconsistent schema.
        - CONSTANTGROUPED: Optimized option for the case when all the files have exactly same headers. In effect this will promote headers from the first file and skip a row for every other file.
        - SAMEALLFILES: Optimized option for the case when all the files have exactly same headers. In effect this will promote headers from the first file and skip a row for every other file.
    """
    NONE = 0
    UNGROUPED = 1
    FIRSTFILE = 1
    GROUPED = 2
    ALLFILES = 2
    CONSTANTGROUPED = 3
    SAMEALLFILES = 3


class SType(Enum):
    """
    Defines suported semantic types

    .. remarks::

        Possible enum values are:
        - EMAILADDRESS: Email address.
        - GEOGRAPHICCOORDINATE: Common representations of geographic coordinates.
        - IPV4ADDRESS: IPv4 address.
        - IPV6ADDRESS: IPv6 address.
        - USPHONENUMBER: Common formats of US phone numbers.
        - ZIPCODE: Common formats of US ZIP code.
    """
    EMAILADDRESS = 0
    GEOGRAPHICCOORDINATE = 1
    IPV4ADDRESS = 2
    IPV6ADDRESS = 3
    USPHONENUMBER = 4
    ZIPCODE = 5


class DatastoreValue:
    """
    Properties uniquely identifying a datastore.

    :param subscription: The subscription the workspace belongs to.
    :param resource_group: The resource group the workspace belongs to.
    :param workspace_name: The workspace the datastore belongs to.
    :param datastore_name: The datastore to read the data from.
    :param path: The path on the datastore.
    """
    def __init__(self,
                 subscription: str,
                 resource_group: str,
                 workspace_name: str,
                 datastore_name: str,
                 path: str = ''):
        self.subscription = subscription
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.datastore_name = datastore_name
        self.path = path

    def _to_pod(self) -> Dict[str, Any]:
        return {
            'subscription': self.subscription,
            'resourceGroup': self.resource_group,
            'workspaceName': self.workspace_name,
            'datastoreName': self.datastore_name,
            'path': self.path,
        }


class _HistogramArgumentsValue:
    """
    Additional arguments required for Histogram summary function.

    :param histogram_bucket_count: Number of buckets to use.
    """
    def __init__(self,
                 histogram_bucket_count: int):
        self.histogram_bucket_count = histogram_bucket_count

    def _to_pod(self) -> Dict[str, Any]:
        return {
            'histogramBucketCount': self.histogram_bucket_count,
        }


class _KernelDensityArgumentsValue:
    """
    Additional arguments required for KernelDensity summary function.

    :param kernel_density_point_count: Number of kernel density points to calculate.
    :param kernel_density_bandwidth: Kernel density bandwidth.
    """
    def __init__(self,
                 kernel_density_point_count: int,
                 kernel_density_bandwidth: float):
        self.kernel_density_point_count = kernel_density_point_count
        self.kernel_density_bandwidth = kernel_density_bandwidth

    def _to_pod(self) -> Dict[str, Any]:
        return {
            'kernelDensityPointCount': self.kernel_density_point_count,
            'kernelDensityBandwidth': self.kernel_density_bandwidth,
        }


class _SummaryColumnsValue:
    """
    Column summarization definition.

    :param column_id: Column to summarize.
    :param summary_function: Aggregation function to use.
    :param summary_column_name: Name of the new column holding the aggregate values.
    :param histogram_arguments: Additional arguments required for Histogram summary function.
    :param kernel_density_arguments: Additional arguments required for KernelDensity summary function.
    :param quantiles: Quantile boundary values required for Quantiles summary function.
    """
    def __init__(self,
                 column_id: str,
                 summary_function: SummaryFunction,
                 summary_column_name: str,
                 histogram_arguments: Optional[_HistogramArgumentsValue] = None,
                 kernel_density_arguments: Optional[_KernelDensityArgumentsValue] = None,
                 quantiles: Optional[List[float]] = None):
        self.column_id = column_id
        self.summary_function = summary_function
        self.summary_column_name = summary_column_name
        self.histogram_arguments = histogram_arguments
        self.kernel_density_arguments = kernel_density_arguments
        self.quantiles = quantiles

    def _to_pod(self) -> Dict[str, Any]:
        return {
            'columnId': self.column_id,
            'summaryFunction': self.summary_function,
            'summaryColumnName': self.summary_column_name,
            'histogramArguments': self.histogram_arguments,
            'kernelDensityArguments': self.kernel_density_arguments,
            'quantiles': self.quantiles,
        }


class _SummaryFunctionsValue:
    """
    Summarization definition for each column.

    :param summary_function: Aggregation function to use.
    :param histogram_arguments: Additional arguments required for Histogram summary function.
    :param kernel_density_arguments: Additional arguments required for KernelDensity summary function.
    :param quantiles: Quantile boundary values required for Quantiles summary function.
    """
    def __init__(self,
                 summary_function: SummaryFunction,
                 histogram_arguments: Optional[_HistogramArgumentsValue] = None,
                 kernel_density_arguments: Optional[_KernelDensityArgumentsValue] = None,
                 quantiles: Optional[List[float]] = None):
        self.summary_function = summary_function
        self.histogram_arguments = histogram_arguments
        self.kernel_density_arguments = kernel_density_arguments
        self.quantiles = quantiles

    def _to_pod(self) -> Dict[str, Any]:
        return {
            'summaryFunction': self.summary_function,
            'histogramArguments': self.histogram_arguments,
            'kernelDensityArguments': self.kernel_density_arguments,
            'quantiles': self.quantiles,
        }
# <<< END GENERATED CLASSES


# Export as public classes
HistogramArgumentsValue = _HistogramArgumentsValue
KernelDensityArgumentsValue = _KernelDensityArgumentsValue
SummaryColumnsValue = _SummaryColumnsValue


class Dataflow:
    """
    A Dataflow represents a series of lazily-evaluated, immutable operations on data.
        It is only an execution plan. No data is loaded from the source until you get data from the Dataflow using one of `head`, `to_pandas_dataframe`, `get_profile` or the write methods.

    .. remarks::

        Dataflows are usually created by supplying a data source. Once the data source has been provided, operations
            can be added by invoking the different transformation methods available on this class. The result of adding
            an operation to a Dataflow is always a new Dataflow.

        The actual loading of the data and execution of the transformations is delayed as much as possible and will not
            occur until a 'pull' takes place. A pull is the action of reading data from a Dataflow, whether by asking to
            look at the first N records in it or by transferring the data in the Dataflow to another storage mechanism
            (a Pandas Dataframe, a CSV file, or a Spark Dataframe).

        The operations available on the Dataflow are runtime-agnostic. This allows the transformation pipelines
            contained in them to be portable between a regular Python environment and Spark.
    """

    def __init__(self,
                 engine_api: 'EngineAPI',
                 steps = None,
                 meta: Dict[str, str] = None,
                 rs_dataflow_yaml: str = None):
        if engine_api is not None:
            _log_not_supported_api_usage_and_raise(get_logger(), "Dataflow with engine_api", "Deprecated, downgrade to a previous version of azureml-dataprep")

        self._engine_api = None
        self._steps = steps if steps is not None else []

        self.builders = Builders(self, None)
        self._meta = meta if meta is not None else {}
        self._rs_dataflow_yaml = rs_dataflow_yaml

    # Methods used by EnginelessDataflow
    def _raise_if_multi_char(self, name: str, value: str):
        if value and len(value) > 1:
           raise ValueError('Only single character is supported for argument: ' + name)

    def _validate_partition_size(self, partition_size: Optional[int] = None):
        if partition_size is not None:
            if partition_size < 1024 * 1024 * 4:
                raise ValueError('partition_size must be at least 4 MB (4 * 1024 * 1024).')
            if partition_size > 2 * 1024 * 1024 * 1024:
                # set arbitrary limit of 2GB as larger partition size can cause OOM
                raise ValueError('partition_size must be at most 2 GB (2 * 1024 * 1024 * 1024).')

    # Methods deprecated in Dataflow but overridden in EnginelessDataflow
    def head(self, count: int = 10) -> 'pandas.DataFrame':
        """
        Pulls the number of records specified from the top of this Dataflow and returns them as a `Link pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_.

        :param count: The number of records to pull. 10 is default.
        :return: A Pandas `Link pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "get_profile", "Use EnginelessDataflow.head() instead")

    def _copy_and_update_metadata(self,
                                  action: str,
                                  source: str,
                                  **kwargs) -> 'Dataflow':
        _log_not_supported_api_usage_and_raise(get_logger(), "_copy_and_update_metadata", "Use EnginelessDataflow._copy_and_update_metadata() instead")

    def _with_partition_size(self, partition_size: int) -> 'Dataflow':
        _log_not_supported_api_usage_and_raise(get_logger(), "_with_partition_size", "Use EnginelessDataflow._with_partition_size() instead")

    # noinspection PyUnresolvedReferences
    @track(get_logger)
    def get_profile(self, include_stype_counts: bool = False, number_of_histogram_bins: int = 10):
        """
        Requests the data profile which collects summary statistics on the full data produced by the Dataflow.
            A data profile can be very useful to understand the input data, identify anomalies and missing values,
            and verify that data preparation operations produced the desired result.

        :param include_stype_counts: Whether to include checking if values look like some well known semantic types of
            information. For Example, "email address". Turning this on will impact performance.
        :type include_stype_counts: bool
        :param number_of_histogram_bins: Number of bins in a histogram. If not specified will be set to 10.
        :type number_of_histogram_bins: int
        :return: DataProfile object
        :rtype: azureml.dataprep.DataProfile
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "get_profile", "Use EnginelessDataflow.get_profile() instead")

    def _get_profile(self,
                     include_stype_counts: bool = False,
                     number_of_histogram_bins: int = 10,
                     include_average_spaces_count: bool = False,
                     include_string_lengths: bool = False):
        """
        Actual get_profile implementation
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "_get_profile", "Use EnginelessDataflow._get_profile() instead")

    def to_path(self, activity, source, stream_column=_DEFAULT_STREAM_INFO_COLUMN):
        _log_not_supported_api_usage_and_raise(get_logger(), "to_path", "Use EnginelessDataflow.to_path() instead")

    @property
    def dtypes(self):
        """
        Gets column data types for the current dataset by calling :meth:`get_profile` and extracting just dtypes from
            the resulting DataProfile.

        .. note:

            This will trigger a data profile calculation.
            To avoid calculating profile multiple times, get the profile first by calling \
            `get_profile()` and then inspect it.

        :return: A dictionary, where key is the column name and value is :class:`azureml.dataprep.FieldType`.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "dtypes", "Use EnginelessDataflow.dtypes instead")

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
        _log_not_supported_api_usage_and_raise(get_logger(), "row_count", "Use EnginelessDataflow.row_count instead")

    @property
    def shape(self):
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
        _log_not_supported_api_usage_and_raise(get_logger(), "shape", "Use EnginelessDataflow.shape instead")

    @track(get_logger)
    def get_partition_count(self) -> int:
        """
        Calculates the partitions for the current Dataflow and returns their count. Partitioning is guaranteed to be stable for a specific execution mode.

        :return: The count of partitions.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "get_partition_count", "Use EnginelessDataflow.get_partition_count() instead")

    @track(get_logger)
    def get_partition_info(self) -> Tuple[int, List[Tuple['StreamInfo', int]]]:
        """
        Calculates the partitions for the current Dataflow and returns total count and a list of partition source and partition count in that source.
        Partitioning is guaranteed to be stable for a specific execution mode.
        Partition source list would be a tuple of (StreamInfo, count) if it can be tied directly to a file or None in other cases.

        :return: The count of partitions.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "get_partition_info", "Use EnginelessDataflow.get_partition_info() instead")

    @track(get_logger)
    def run_local(self) -> None:
        """
        Runs the current Dataflow using the local execution runtime.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "run_local", "Use EnginelessDataflow.run_local() instead")

    # noinspection PyUnresolvedReferences
    @track(get_logger)
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
        _log_not_supported_api_usage_and_raise(get_logger(), "to_pandas_dataframe", "Use EnginelessDataflow.to_pandas_dataframe() instead.")

    @track(get_logger)
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
        _log_not_supported_api_usage_and_raise(get_logger(), "to_dask_dataframe", "Use EnginelessDataflow.to_dask_dataframe() instead.")

    def _to_pyrecords(self):
        _log_not_supported_api_usage_and_raise(get_logger(), "_to_pyrecords", "Use EnginelessDataflow._to_pyrecords() instead.")

    # noinspection PyUnresolvedReferences
    @track(get_logger)
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
        _log_not_supported_api_usage_and_raise(get_logger(), "to_spark_dataframe", "Use EnginelessDataflow.to_spark_dataframe() instead.")

    # noinspection PyUnresolvedReferences
    @track(get_logger)
    def verify_has_data(self):
        """
        Verifies that this Dataflow would produce records if executed. An exception will be thrown otherwise.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "verify_has_data", "Use EnginelessDataflow.verify_has_data() instead.")

    def parse_delimited(self,
                        separator,
                        headers_mode,
                        encoding,
                        quoting,
                        skip_rows,
                        skip_mode,
                        comment,
                        partition_size = None,
                        empty_as_string = False) -> 'Dataflow':
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
        _log_not_supported_api_usage_and_raise(get_logger(), "parse_delimited", "Use EnginelessDataflow.parse_delimited() instead.")

    def parse_json_lines(self,
                         encoding,
                         partition_size = None,
                         invalid_lines = None) -> 'Dataflow':
        """
        Creates a new Dataflow with the operations required to read JSON lines files.

        :param invalid_lines: How to handle invalid JSON lines.
        :param encoding: The encoding of the files being read.
        :param partition_size: Desired partition size.
        :return: A new Dataflow with Read JSON line Step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "parse_json_lines", "Use EnginelessDataflow.parse_json_lines() instead.")

    def read_parquet_file(self) -> 'Dataflow':
        """
        Adds step to parse Parquet files.

        :return: A new Dataflow with Parse Parquet File Step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "read_parquet_file", "Use read_parquet_file reader instead.")

    def read_preppy(self) -> 'Dataflow':
        """
        Adds step to read a directory containing Preppy files.

        :return: A new Dataflow with Read Preppy Step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "read_preppy", "Use read_preppy instead.")

    def set_column_types(self, type_conversions) -> 'Dataflow':
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
        _log_not_supported_api_usage_and_raise(get_logger(), "set_column_types", "Use EnginelessDataflow.set_column_types() instead.")

    def take_sample(self,
                    probability: float,
                    seed: Optional[int] = None) -> 'Dataflow':
        """
        Takes a random sample of the available records.

        :param probability: The probability of a record being included in the sample.
        :param seed: The seed to use for the random generator.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "take_sample", "Use EnginelessDataflow.take_sample() instead.")

    def _add_columns_from_record(self,
                                expression,
                                prior_column: str = None,
                                new_column_prefix: str = None) -> 'Dataflow':
        """
        Add new columns to the dataset based on the record generated by expression

        .. remarks::

           Expressions are built using the expression builders in the expressions module and the functions in
           the functions module. The resulting expression will be lazily evaluated for each record when a data pull
           occurs and not where it is defined.

        :param expression: The expression to evaluate to generate the values in the column.
        :param prior_column: The name of the column after which the new column should be added. The default is to add
            the new column as the last column.
        :param new_column_prefix: string value to be prepend in the name of the new columns added to dataset. This
            might be needed to avoid column name conflicts.
        :return: The modified Dataflow.
        """

        _log_not_supported_api_usage_and_raise(get_logger(), "_add_columns_from_record", "Use EnginelessDataflow._add_columns_from_record() instead.")

    def _add_columns_from_partition_format(self,
                                           column: str,
                                           partition_format: str,
                                           ignore_error: bool) -> 'Dataflow':
        """
        Add new columns to the dataset based on matching the partition format for provided column.

        :param partition_format: The partition format matching the column to create columns.
        :param ignore_error: Indicate whether or not to fail the execution if there is any error.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "_add_columns_from_partition_format", "Use EnginelessDataflow._add_columns_from_partition_format() instead.")

    def add_column(self, expression, new_column_name: str, prior_column: str) -> 'Dataflow':
        """
        Adds a new column to the dataset. The values in the new column will be the result of invoking the specified
        expression.

        .. remarks::

            Expressions are built using the expression builders in the expressions module and the functions in
            the functions module. The resulting expression will be lazily evaluated for each record when a data pull
            occurs and not where it is defined.

        :param expression: The expression to evaluate to generate the values in the column.
        :param new_column_name: The name of the new column.
        :param prior_column: The name of the column after which the new column should be added. The default is to add
            the new column as the last column.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "add_column", "Use EnginelessDataflow.add_column() instead.")

    def filter(self, expression) -> 'Dataflow':
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
        _log_not_supported_api_usage_and_raise(get_logger(), "filter", "Use EnginelessDataflow.filter() instead.")

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
        _log_not_supported_api_usage_and_raise(get_logger(), "random_split", "Use EnginelessDataflow.random_split() instead.")

    def replace_datasource(self, new_datasource) -> 'Dataflow':
        """
        Returns new Dataflow with its DataSource replaced by the given one.

        .. remarks::

            The given 'new_datasource' must match the type of datasource already in the Dataflow.
            For example a MSSQLDataSource cannot be replaced with a FileDataSource.

        :param new_datasource: DataSource to substitute into new Dataflow.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "replace_datasource", "Use EnginelessDataflow.replace_datasource() instead.")

    def to_datetime(self,
                    columns,
                    date_time_formats: Optional[List[str]] = None,
                    date_constant: Optional[str] = None) -> 'Dataflow':
        """
        Converts the values in the specified columns to DateTimes.

        :param columns: The source columns.
        :param date_time_formats: The formats to use to parse the values. If none are provided, a partial scan of the
            data will be performed to derive one.
        :param date_constant: If the column contains only time values, a date to apply to the resulting DateTime.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_datetime", "Use EnginelessDataflow.to_datetime() instead")

    def summarize(self,
                  summary_columns = None,
                  group_by_columns: Optional[List[str]] = None,
                  join_back: bool = False,
                  join_back_columns_prefix: Optional[str] = None) -> 'Dataflow':
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
        :param join_back: Whether to append subtotals or replace current data with them.
        :param join_back_columns_prefix: Prefix to use for subtotal columns when appending them to current data.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "summarize", "Use EnginelessDataflow.summarize() instead")

    @staticmethod
    def open(file_path: str) -> 'Dataflow':
        """
        Opens a Dataflow with specified name from the package file.

        :param file_path: Path to the package containing the Dataflow.
        :return: The Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "open", "Use EnginelessDataflow.open() instead")

    @staticmethod
    def from_json(dataflow_json: str) -> 'Dataflow':
        """
        Load Dataflow from 'package_json'.

        :param dataflow_json: JSON string representation of the Package.
        :return: New Package object constructed from the JSON string.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "from_json", "Use EnginelessDataflow.from_json() instead")

    def save(self, file_path: str):
        """
        Saves the Dataflow to the specified file

        :param file_path: The path of the file.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "save", "Use EnginelessDataflow.save() instead")

    def to_json(self) -> str:
        """
        Get the JSON string representation of the Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_json", "Use EnginelessDataflow.to_json() instead")

    def _partition_to_pandas_dataframe(self,
                                       i: int,
                                       extended_types: bool,
                                       nulls_as_nan: bool,
                                       on_error: str,
                                       out_of_range_datetime: str) -> 'pandas.DataFrame':
        _log_not_supported_api_usage_and_raise(get_logger(), "_partition_to_pandas_dataframe", "Use EnginelessDataflow._partition_to_pandas_dataframe() instead")

    def __getitem__(self, key):
        _log_not_supported_api_usage_and_raise(get_logger(), "__get_item__", "Use EnginelessDataflow__getitem__() instead")

    def __getstate__(self):
        _log_not_supported_api_usage_and_raise(get_logger(), "__getstate__", "Use EnginelessDataflow.__getstate__() instead")

    def __setstate__(self, newstate):
        _log_not_supported_api_usage_and_raise(get_logger(), "__setstate", "Use EnginelessDataflow.__setstate__() instead")

    def __add__(self, other):
        _log_not_supported_api_usage_and_raise(get_logger(), "__add__", "Use EnginelessDataflow.__add__() instead")

    def __repr__(self):
        _log_not_supported_api_usage_and_raise(get_logger(), "__repr__", "Use EnginelessDataflow.__repr__() instead")

    def keep_columns(self,
                     columns,
                     validate_column_exists: bool = False) -> 'Dataflow':
        """
        Keeps the specified columns and drops all others.

        :param columns: The source columns.
        :param validate_column_exists: Whether to validate the columns selected.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "keep_columns", "Use EnginelessDataflow.keep_columns() instead")

    def _find_path_prefix(self, stream_column = None, avoid_data_pull = False):
        _log_not_supported_api_usage_and_raise(get_logger(), "_find_path_prefix", "Use EnginelessData.flow_find_path_prefix() instead")

    def _get_prefix(self, path, avoid_data_pull = False):
        """Determine if there exists a common prefix for all files which may exist under the given path/dataflow.

        :param path: Path extracted from dataflow
        :return: Path which is common prefix of all files under path/dataflow, or None if a common prefix was not found.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "_get_prefix", "Use EnginelessDataflow._get_prefix() instead")

    def _get_partition_format(self) -> str:
        _log_not_supported_api_usage_and_raise(get_logger(), "_get_partition_format", "Use EnginelessDataflow._get_partition_format() instead")

    def _remove_data_read_steps_and_drop_columns(self, remove_drop_column):
        _log_not_supported_api_usage_and_raise(get_logger(), "_remove_data_read_steps_and_drop_columns", "Use EnginelessDataflow._remove_data_read_steps_and_drop_columns() instead")

    @staticmethod
    def _get_optimal_path(common_path_list, stored_path_list, verbose):
        """Merge common path segment list with stored path segment list and returns optimal path.

        :param common_path_list: list containing common path segments
        :type partition_keys: builtin.list[str]
        :param stored_path_list: list containing stored path segments
        :type stored_path_list: builtin.list[str]
        :param verbose: boolean value to print out the optimizations performed
        :type verbose: bool
        :return: str
        :rtype: str
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "_get_optimal_path", "Use EnginelessDataflow._get_optimal_path() instead")

    def _with_optimized_datasource(self, common_path, verbose: bool):
        _log_not_supported_api_usage_and_raise(get_logger(), "_with_optimized_datasource", "Use EnginelessDataflow._with_optimized_datasource() instead")

    def select_partitions(self,
                          partition_indices: List[int]) -> 'Dataflow':
        """
        Selects specific partitions from the data, dropping the rest.

        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "select_partitions", "Use EnginelessDataflow.select_partitions() instead")

    def to_csv_streams(self,
                       separator: str = ',',
                       na: str = 'NA',
                       error: str = 'ERROR') -> 'Dataflow':
        """
        Creates streams with the data in delimited format.

        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_csv_streams", "Use EnginelessDataflow.to_csv_streams() instead")

    def to_parquet_streams(self,
                           error: str = 'ERROR',
                           rows_per_group: int = 5000) -> 'Dataflow':
        """
        Creates streams with the data in parquet format.

        :param error: String to use for error values.
        :param rows_per_group: Number of rows to use per row group.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_parquet_steams", "Use EnginelessDataflow.to_parquet_streams() instead")

    def distinct_rows(self) -> 'Dataflow':
        """
        Filters out records that contain duplicate values in all columns, leaving only a single instance.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "distinct_rows", "Use EnginelessDataflow.distinct_rows() instead")

    def distinct(self,
                 columns) -> 'Dataflow':
        """
        Filters out records that contain duplicate values in the specified columns, leaving only a single instance.

        :param columns: The source columns.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "distinct", "Use EnginelessDataflow.distinct() instead")

    def skip(self,
             count: int) -> 'Dataflow':
        """
        Skips the specified number of records.

        :param count: The number of records to skip.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "skip", "Use EnginelessDataflow.skip() instead")

    def take(self,
             count: int) -> 'Dataflow':
        """
        Takes the specified count of records.

        :param count: The number of records to take.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "take", "Use EnginelessDataflow.take() instead")

    def rename_columns(self,
                       column_pairs: Dict[str, str]) -> 'Dataflow':
        """
        Renames the specified columns.

        :param column_pairs: The columns to rename and the desired new names.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "rename_columns", "Use EnginelessDataflow.rename_columns() instead")

    def drop_columns(self,
                     columns) -> 'Dataflow':
        """
        Drops the specified columns.

        :param columns: The source columns.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "drop_columns", "Use EnginelessDataflow.drop_columns() instead")

    def to_number(self,
                  columns,
                  decimal_point = None) -> 'Dataflow':
        """
        Converts the values in the specified columns to floating point numbers.

        :param columns: The source columns.
        :param decimal_point: The symbol to use as the decimal mark.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_number", "Use EnginelessDataflow.to_number() instead")

    def to_bool(self,
                columns,
                true_values: List[str],
                false_values: List[str],
                mismatch_as = None) -> 'Dataflow':
        """
        Converts the values in the specified columns to booleans.

        :param columns: The source columns.
        :param true_values: The values to treat as true.
        :param false_values: The values to treat as false.
        :param mismatch_as: How to treat values that don't match the values in the true or false values lists.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_bool", "Use EnginelessDataflow.to_bool() instead")

    def to_string(self,
                  columns) -> 'Dataflow':
        """
        Converts the values in the specified columns to strings.

        :param columns: The source columns.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_string", "Use EnginelessDataflow.to_string() instead")

    def to_long(self,
                columns) -> 'Dataflow':
        """
        Converts the values in the specified columns to 64 bit integers.

        :param columns: The source columns.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_long", "Use EnginelessDataflow.to_long() instead")

    def write_streams(self,
                      streams_column: str,
                      base_path: FileOutput,
                      file_names_column: Optional[str] = None,
                      prefix_path: Optional[str] = None) -> 'Dataflow':
        """
        Writes the streams in the specified column to the destination path. By default, the name of the files written will be the resource identifier
            of the streams. This behavior can be overriden by specifying a column which contains the names to use.

        :param streams_column: The column containing the streams to write.
        :param file_names_column: A column containing the file names to use.
        :param base_path: The path under which the files should be written.
        :param prefix_path: The prefix path that needs to be removed from the target paths.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "write_streams", "Use EnginelessDataflow.write_streams() instead")

    # Methods deprecated in Dataflow & not used in EnginelessDataflow but don't raise a error when used
    def assert_value(self,
                     columns,
                     expression,
                     policy = None,
                     error_code: str = 'AssertionFailed') -> 'Dataflow':
        """
        Ensures that values in the specified columns satisfy the provided expression. This is useful to identify anomalies in the dataset
            and avoid broken pipelines by handling assertion errors in a clean way.

        :param columns: Columns to apply evaluation to.
        :param expression: Expression that has to be evaluated to be True for the value to be kept.
        :param policy: Action to take when expression is evaluated to False. Options are `FAILEXECUTION` and `ERRORVALUE`.
            `FAILEXECUTION` ensures that any data that violates the assertion expression during execution will immediately fail the job. This is useful to save computing resources and time.
            `ERRORVALUE` captures any data that violates the assertion expression by replacing it with error_code. This allows you to handle these error values by either filtering or replacing them.
        :param error_code: Error message to use to replace values failing the assertion or failing an execution.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "assert_value", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def get_missing_secrets(self) -> List[str]:
        """
        Get a list of missing secret IDs.

        :return: A list of missing secret IDs.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "get_missing_secrets", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def use_secrets(self, secrets: Dict[str, str]):
        """
        Uses the passed in secrets for execution.

        :param secrets: A dictionary of secret ID to secret value. You can get the list of required secrets by calling
            the get_missing_secrets method on Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "use_secrets", "Deprecated, downgrade to a previous version of azureml-dataprep")

    # Methods totally deprecated in Dataflow & EnginelessDataflow that shouldn't be used anywhere
    def add_step(self,
                 step_type: str,
                 arguments: Dict[str, Any],
                 local_data: Dict[str, Any] = None) -> 'Dataflow':
        _log_not_supported_api_usage_and_raise(get_logger(), "add_step", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def _get_steps(self):
        _log_not_supported_api_usage_and_raise(get_logger(), "_get_steps", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def execute_inspector(self, inspector):
        _log_not_supported_api_usage_and_raise(get_logger(), "execute_inspector", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def _execute_inspector(self, inspector):
        _log_not_supported_api_usage_and_raise(get_logger(), "_execute_inspector", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def execute_inspectors(self, inspectors):
        _log_not_supported_api_usage_and_raise(get_logger(), "execute_inspectors", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def _execute_inspectors(self, inspectors):
        _log_not_supported_api_usage_and_raise(get_logger(), "_execute_inspectors", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @track(get_logger)
    def run_spark(self) -> None:
        """
        Runs the current Dataflow using the Spark runtime.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "run_spark", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @track(get_logger)
    def to_record_iterator(self) -> 'RecordIterable':
        """
        Creates an iterable object that returns the records produced by this Dataflow in sequence. This iterable
        must be closed before any other executions can run.

        :return: A RecordIterable object that can be used to iterate over the records in this Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_record_iterator", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @track(get_logger)
    def to_partition_iterator(self, on_error: str = 'null') -> 'PartitionIterable':
        """
        Creates an iterable object that returns the partitions produced by this Dataflow in sequence. This iterable
        must be closed before any other executions can run.

        :param on_error: How to handle any error values in the Dataflow, such as those produced by an error while parsing values.
            Valid values are 'null' which replaces them with null; and 'fail' which will result in an exception.
        :return: A PartitionsIterable object that can be used to iterate over the partitions in this Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_partition_iterator", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @track(get_logger)
    def has_invalid_source(self, return_validation_error=False):
        """
        Verifies if the Dataflow has invalid source.

        :param return_validation_error: Action to take when source is know to be invalid. Options are:
            `True` Returns error message. This is useful to gather more information regarding the failure
            occurred while checking whether the source is valid or not.
            `False` return True.
        :return: Return following based on the parameter checked
            - returns error message if show_error_message == True and source known to be invalid.
            - returns True if show_error_message == False and source known to be invalid.
            - returns False if source known to valid or unknown.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "has_invalid_source", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def parse_fwf(self,
                  offsets,
                  headers_mode,
                  encoding,
                  skip_rows,
                  skip_mode) -> 'Dataflow':
        """
        Adds step to parse fixed-width data.

        :param offsets: The offsets at which to split columns. The first column is always assumed to start at offset 0.
        :param headers_mode: How to determine column headers.
        :param encoding: The encoding of the files being read.
        :param skip_rows: How many rows to skip.
        :param skip_mode: The mode in which rows are skipped.
        :return: A new Dataflow with Parse FixedWidthFile Step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "parse_fwf", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def parse_lines(self,
                    headers_mode,
                    encoding,
                    skip_rows,
                    skip_mode,
                    comment,
                    partition_size = None) -> 'Dataflow':
        """
        Adds step to parse text files and split them into lines.

        :param headers_mode: How to determine column headers.
        :param encoding: The encoding of the files being read.
        :param skip_rows: How many rows to skip.
        :param skip_mode: The mode in which rows are skipped.
        :param comment: Character used to indicate a line is a comment instead of data in the files being read.
        :param partition_size: Desired partition size.
        :return: A new Dataflow with Parse Lines Step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "parse_lines", "Use EnginelessDataflow.read_delimited instead.")

    def read_sql(self, data_source, query: str, query_timeout: int = 30) -> 'Dataflow':
        """
        Adds step that can read data from an MS SQL database by executing the query specified.

        :param data_source: The details of the MS SQL database.
        :param query: The query to execute to read data.
        :param query_timeout: Sets the wait time (in seconds) before terminating the attempt to execute a command
            and generating an error. The default is 30 seconds.
        :return: A new Dataflow with Read SQL Step added.
        """

        _log_not_supported_api_usage_and_raise(get_logger(), "read_sql", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def write_to_sql(self,
        destination,
        table: str,
        batch_size: int = 500,
        default_string_length: int = None) -> 'Dataflow':
        """
        Adds step that writes out the data in the Dataflow into a table in MS SQL database.

        :param destination: The details of the MS SQL database or AzureSqlDatabaseDatastore.
        :param table: Name of the table to write data to.
        :param batch_size: Size of a batch of records to commit to SQL server in a single request.
        :param default_string_length: Length of string to be supported when creating a table.

        :return: A new Dataflow with write step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "write_to_sql", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def read_postgresql(self, data_source, query: str, query_timeout: int = 20) -> 'Dataflow':
        """
        Adds step that can read data from a PostgreSQL database by executing the query specified.

        :param data_source: The details of the PostgreSQL database.
        :param query: The query to execute to read data.
        :param query_timeout: Sets the wait time (in seconds) before terminating the attempt to execute a command
            and generating an error. The default is 20 seconds.
        :return: A new Dataflow with Read SQL Step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "read_postgresql", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def read_excel(self,
                   sheet_name: Optional[str] = None,
                   use_column_headers: bool = False,
                   skip_rows: int = 0) -> 'Dataflow':
        """
        Adds step to read and parse Excel files.

        :param sheet_name: The name of the sheet to load. The first sheet is loaded if none is provided.
        :param use_column_headers: Whether to use the first row as column headers.
        :param skip_rows: Number of rows to skip when loading the data.
        :return: A new Dataflow with Read Excel Step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "read_excel", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def read_json(self,
                  json_extract_program: str,
                  encoding):
        """
        (DEPRECATED)
        Creates a new Dataflow with the operations required to read JSON files.

        :param json_extract_program: PROSE program that will be used to parse JSON.
        :param encoding: The encoding of the files being read.
        :return: A new Dataflow with Read JSON Step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "read_json", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def read_npz_file(self) -> 'Dataflow':
        """
        (DEPRECATED)
        Adds step to parse npz files.

        :return: A new Dataflow with Read Npz File Step added.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "read_npz_file", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def take_stratified_sample(self,
                               columns,
                               fractions: Dict[Tuple, int],
                               seed: Optional[int] = None) -> 'Dataflow':
        """
        (DEPRECATED)
        Takes a random stratified sample of the available records according to input fractions.

        :param columns: The strata columns.
        :param fractions: The strata to strata weights.
        :param seed: The seed to use for the random generator.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "take_stratified_sample", "Use take_sample instead.")

    def derive_column_by_example(self, source_columns, new_column_name: str, example_data) -> 'Dataflow':
        """
        (DEPRECATED)
        Inserts a column by learning a program based on a set of source columns and provided examples.
            Dataprep will attempt to achieve the intended derivation inferred from provided examples.

        .. remarks::

            If you need more control of examples and generated program, create DeriveColumnByExampleBuilder instead.

        :param source_columns: Names of the columns from which the new column will be derived.
        :param new_column_name: Name of the new column to add.
        :param example_data: Examples to use as input for program generation.
            In case there is only one column to be used as source, examples could be Tuples of source value and intended
            target value. For example, you can have "example_data=[("2013-08-22", "Thursday"), ("2013-11-03", "Sunday")]".
            When multiple columns should be considered as source, each example should be a Tuple of dict-like sources
            and intended target value, where sources have column names as keys and column values as values.
        :return: The modified Dataflow.
        """

        self.builders.derive_column_by_example(None, None)

    def pivot(self,
              columns_to_pivot: List[str],
              value_column: str,
              summary_function = None,
              group_by_columns: List[str] = None,
              null_value_replacement: str = None,
              error_value_replacement: str= None) -> 'Dataflow':
        """
        (DEPRECATED)
        Returns a new Dataflow with columns generated from the values in the selected columns to pivot.

        .. remarks::

            The values of the new dataflow come from the value column selected.
            Additionally there is an optional summarization that consists of an aggregation and a group by.

        :param columns_to_pivot: The columns used to get the values from which the new dataflow's new columns are generated.
        :param value_column: The column used to get the values that will populate the new dataflow's rows.
        :summary_function: The summary function used to aggregate the values.
        :group_by_columns: The columns used to group the new dataflow rows.
        :null_value_replacement: String value to replace null values in columns_to_pivot. If unspecified, the string "NULL" will be used.
        :error_value_replacement: String value to replace error values in columns_to_pivot. If unspecified, the string "ERROR" will be used.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "pivot", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def multi_split(self, splits: int, seed: Optional[int] = None):
        """Split a Dataflow into multiple other Dataflows, each containing a random but exclusive sub-set of the data.

        :param splits: The number of splits.
        :param seed: The seed to use for the random split.
        :return: A list containing one Dataflow per split.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "multi_split", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def replace_reference(self, new_reference) -> 'Dataflow':
        """
        Returns new Dataflow with its reference DataSource replaced by the given one.

        :param new_reference: Reference to be substituted for current Reference in Dataflow.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "replace_reference", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def cache(self, directory_path: str) -> 'Dataflow':
        """
        (DEPRECATED)
        Pulls all the records from this Dataflow and cache the result to disk.

        .. remarks::

            This is very useful when data is accessed repeatedly, as future executions will reuse
            the cached result without pulling the same Dataflow again.

        :param directory_path: The directory to save cache files.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "cache", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def new_script_column(self,
                          new_column_name: str,
                          insert_after: str,
                          script: str) -> 'Dataflow':
        """
        (DEPRECATED)
        Adds a new column to the Dataflow using the passed in Python script.

        .. remarks::

            The Python script must define a function called newvalue that takes a single argument, typically
                called row. The row argument is a dict (key is column name, value is current value) and will be passed
                to this function for each row in the dataset. This function must return a value to be used in the new column.

            .. note::

                Any libraries that the Python script imports must exist in the environment where the dataflow is run.

            .. code-block:: python

                import numpy as np
                def newvalue(row):
                    return np.log(row['Metric'])

        :param new_column_name: The name of the new column being created.
        :param insert_after: The column after which the new column will be inserted.
        :param script: The script that will be used to create this new column.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "new_script_column", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def new_script_filter(self, script: str) -> 'Dataflow':
        """
        (DEPRECATED)
        Filters the Dataflow using the passed in Python script.

        .. remarks::

            The Python script must define a function called includerow that takes a single argument, typically
                called row. The row argument is a dict (key is column name, value is current value) and will be passed
                to this function for each row in the dataset. This function must return True or False depending on whether
                the row should be included in the dataflow. Any libraries that the Python script imports must exist in the
                environment where the dataflow is run.

            .. code-block:: python

                def includerow(row):
                    return row['Metric'] > 100

        :param script: The script that will be used to filter the dataflow.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "new_script_filter", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def map_partition(self,
                      fn,
                      data_format: str = 'lastProcessed') -> 'Dataflow':
        """
        Applies a transformation to each partition in the Dataflow.

        .. remarks::
            The function passed in must take in two parameters: a `Link pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ or
                `Link scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ containing the data for the partition and an index.
                The return value must be a `Link pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ or a
                `Link scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ containing the transformed data.
                By default the 'lastProcessed' format is passed into `fn`, i.e. if the data coming into map_partitions is Sparse it will be sent as a
                `Link scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_, if it is dense it will
                be sent as a `Link pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_. The desired
                input data format can be set explicitly using the `data_format` parameter.

            .. note::

                `df` does not usually contain all of the data in the dataflow, but a partition of the data as it is being processed in the \
                runtime. The number and contents of each partition is not guaranteed across runs.

            The transform function can fully edit the passed in dataframe or even create a new one, but must return a
                dataframe. Any libraries that the Python script imports must exist in the environment where the dataflow
                is run.

        :param fn: A callable that will be invoked to transform each partition.
        :param dataFormat: A optional string specifying the input format to fn. Supported Formats: 'dataframe', 'csr', 'lastProcessed'.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "map_partition", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def transform_partition(self, script: 'str') -> 'Dataflow':
        """
        Applies a transformation to each partition in the Dataflow.

        * This function has been deprecated and will be removed in a future version. Please use `map_partition` instead.

        .. remarks::

            The Python script must define a function called transform that takes two arguments, typically called `df` and
                `index`. The `df` argument will be a Pandas Dataframe passed to this function that contains the data for the
                partition and the `index` argument is a unique identifier of the partition.

            .. note::

                `df` does not usually contain all of the data in the dataflow, but a partition of the data as it is being processed in the \
                runtime. The number and contents of each partition is not guaranteed across runs.

            The transform function can fully edit the passed in dataframe or even create a new one, but must return a
                dataframe. Any libraries that the Python script imports must exist in the environment where the dataflow
                is run.

            .. code-block:: python

                # the script passed in should have this function defined
                def transform(df, index):
                    # perform any partition level transforms here and return resulting `df`
                    return df

        :param script: A script that will be used to transform each partition.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "transform_partition", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def transform_partition_with_file(self,
                                      script_path: str) -> 'Dataflow':
        """
        Transforms an entire partition using the Python script in the passed in file.

        .. remarks::

            The Python script must define a function called transform that takes two arguments, typically called `df` and
                `index`. The first argument `df` will be a Pandas Dataframe that contains the data for the partition and the
                second argument `index` will be a unique identifier for the partition.

            .. note::

                `df` does not usually contain all of the data in the dataflow, but a partition of the data as it is being processed in the runtime.\
                The number and contents of each partition is not guaranteed across runs.

            The transform function can fully edit the passed in dataframe or even create a new one, but must return a
                dataframe. Any libraries that the Python script imports must exist in the environment where the dataflow is run.

            .. code-block:: python

                # the script file passed in should have this function defined
                def transform(df, index):
                    # perform any partition level transforms here and return resulting `df`
                    return df

        :param script_path: Relative path to script that will be used to transform the partition.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "transform_partition_with_file", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def split_column_by_delimiters(self,
                                   source_column: str,
                                   delimiters,
                                   keep_delimiters: False) -> 'Dataflow':
        """
        Splits the provided column and adds the resulting columns to the dataflow.

        .. remarks::

            This will pull small sample of the data, determine number of columns it should expect as a result of the
                split and generate a split program that would ensure that the expected number of columns will be produced,
                so that there is a deterministic schema after this operation.

        :param source_column: Column to split.
        :param delimiters: String or list of strings to be deemed as column delimiters.
        :param keep_delimiters: Controls whether to keep or drop column with delimiters.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "split_column_by_delimiters", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def split_column_by_example(self, source_column: str, example = None) -> 'Dataflow':
        """
        Splits the provided column and adds the resulting columns to the dataflow based on the provided example.

        .. remarks::

            This will pull small sample of the data, determine the best program to satisfy provided example
                and generate a split program that would ensure that the expected number of columns will be produced, so that
                there is a deterministic schema after this operation.

            .. note::

                If example is not provided, this will generate a split program based on common split patterns, like splitting by space, punctuation, date parts and etc.

        :param source_column: Column to split.
        :param example: Example to use for program generation.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "split_column_by_example", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def replace(self,
                columns,
                find: Any,
                replace_with: Any) -> 'Dataflow':
        """
        Replaces values in a column that match the specified search value.

        .. remarks::

            The following types are supported for both the find or replace arguments: str, int, float,
                datetime.datetime, and bool.

        :param columns: The source columns.
        :param find: The value to find, or None.
        :param replace_with: The replacement value, or None.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "replace", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def error(self,
              columns,
              find: Any,
              error_code: str) -> 'Dataflow':
        """
        Creates errors in a column for values that match the specified search value.

        .. remarks::

            The following types are supported for the find argument: str, int, float,
                datetime.datetime, and bool.

        :param columns: The source columns.
        :param find: The value to find, or None.
        :param error_code: The error code to use in new errors, or None.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "error", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def fill_nulls(self,
                   columns,
                   fill_with: Any) -> 'Dataflow':
        """
        Fills all nulls in a column with the specified value.

        .. remarks::

            The following types are supported for the fill_with argument: str, int, float,
                datetime.datetime, and bool.

        :param columns: The source columns.
        :param fill_with: The value to fill nulls with.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "fill_nulls", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def fill_errors(self,
                    columns,
                    fill_with: Any) -> 'Dataflow':
        """
        Fills all errors in a column with the specified value.

        .. remarks::

            The following types are supported for the fill_with argument: str, int, float,
                datetime.datetime, and bool.

        :param columns: The source columns.
        :param fill_with: The value to fill errors with, or None.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "fill_errors", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def join(self,
             right_dataflow,
             join_key_pairs: List[Tuple[str, str]] = None,
             join_type = None,
             left_column_prefix: str = 'l_',
             right_column_prefix: str = 'r_',
             left_non_prefixed_columns: List[str] = None,
             right_non_prefixed_columns: List[str] = None) -> 'Dataflow':
        """
        Creates a new Dataflow that is a result of joining this Dataflow with the provided right_dataflow.

        :param right_dataflow: Right Dataflow or DataflowReference to join with.
        :param join_key_pairs: Key column pairs. List of tuples of columns names where each tuple forms a key pair to
            join on. For instance: [('column_from_left_dataflow', 'column_from_right_dataflow')]
        :param join_type: Type of join to perform. Match is default.
        :param left_column_prefix: Prefix to use in result Dataflow for columns originating from left_dataflow.
            Needed to avoid column name conflicts at runtime.
        :param right_column_prefix: Prefix to use in result Dataflow for columns originating from right_dataflow.
            Needed to avoid column name conflicts at runtime.
        :param left_non_prefixed_columns: List of column names from left_dataflow that should not be prefixed with
            left_column_prefix. Every other column appearing in the data at runtime will be prefixed.
        :param right_non_prefixed_columns: List of column names from right_dataflow that should not be prefixed with
            left_column_prefix. Every other column appearing in the data at runtime will be prefixed.
        :return: The new Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "join", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def write_to_csv(self,
                     file_path = None,
                     directory_path = None,
                     single_file: bool = False,
                     separator: str = ',',
                     na: str = 'NA',
                     error: str = 'ERROR',
                     if_destination_exists = None) -> 'Dataflow':
        """
        Write out the data in the Dataflow in a delimited text format. The output is specified as a directory
            which will contain multiple files, one per partition processed in the Dataflow.

        :param directory_path: The path to a directory in which to store the output files.
        :param separator: The separator to use.
        :param na: String to use for null values.
        :param error: String to use for error values.
        :param if_destination_exists: Behavior if destination exists.
        :return: The modified Dataflow. Every execution of the returned Dataflow will perform the write again.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "write_to_csv", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def write_to_parquet(self,
                         file_path = None,
                         directory_path = None,
                         single_file: bool = False,
                         error: str = 'ERROR',
                         row_groups: int = 0,
                         if_destination_exists = None,
                         partition_keys: List[str] = None ) -> 'Dataflow':
        """
        Writes out the data in the Dataflow into Parquet files.

        :param file_path: The path in which to store the output file.
        :param directory_path: The path in which to store the output files.
        :param single_file: Whether to store the entire Dataflow in a single file.
        :param error: String to use for error values.
        :param row_groups: Number of rows to use per row group.
        :param if_destination_exists: Behavior if destination exists.
        :param partition_keys: Optional, list of column names used to write the data by.
            Defaults to be None.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "write_to_parquet", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def write_to_preppy(self, directory_path,
                        if_destination_exists = None) -> 'Dataflow':
        """
        Writes out the data in the Dataflow into Preppy files, a DataPrep serialization format.

        :param directory_path: The path in which to store the output Preppy files.
        :param if_destination_exists: Behavior if destination exists.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "write_to_preppy", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def sort_asc(self, columns) -> 'Dataflow':
        """
        Sorts the dataset in ascending order by the specified columns.

        :param columns: The columns to sort in ascending order.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "sort_asc", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def sort_desc(self, columns) -> 'Dataflow':
        """
        Sorts the dataset in descending order by the specified columns.

        :param columns: The columns to sort in descending order.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "sort_desc", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def _assert_schema(self,
                       columns,
                       schema_assertion_policy = None,
                       policy= None,
                       error_code: str = 'AssertionFailed') -> 'Dataflow':
        """
        Compare schema based on the schema_assertion_policy. Currently we only support ComparePolicy.CONTAIN.

        :param columns: Columns to apply evaluation to.
        :param schema_assertion_policy: Evaluation to apply for the columns argument and record schema.
        :param policy: Action to take when assertion is evaluated to False. Options are `FAILEXECUTION` and `ERRORVALUE`.
            `FAILEXECUTION` ensures that any data that violates the assertion expression during execution will immediately fail the job. This is useful to save computing resources and time.
            `ERRORVALUE` captures any data that violates the assertion expression by replacing it with error_code. This allows you to handle these error values by either filtering or replacing them.
        :param error_code: Error message to use to replace values failing the assertion or failing an execution.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "_assert_schema", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def _get_source_data_hash(self) -> (str, datetime):
        """
        Get the hash of the source data in the dataflow
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "_get_source_data_hash", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def get_files(path) -> 'Dataflow':
        """
        Expands the path specified by reading globs and files in folders and outputs one record per file found.

        :param path: The path or paths to expand.
        :return: A new Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "get_files", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _dataflow_from_activity_data(activity_data, engine_api: 'EngineAPI') -> 'Dataflow':
        _log_not_supported_api_usage_and_raise(get_logger(), "_dataflow_from_activity_data", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _dataflow_to_anonymous_activity_data(dataflow: 'Dataflow'):
        _log_not_supported_api_usage_and_raise(get_logger(), "_dataflow_to_anonymous_activity_data", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _step_from_block_data(block_data):
        _log_not_supported_api_usage_and_raise(get_logger(), "_step_from_block_data", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _path_to_get_files_block(path, force_file: bool=False) -> 'EnginelessDataflow':
        _log_not_supported_api_usage_and_raise(get_logger(), "_path_to_get_files_block", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _path_to_create_dataset_block(path) -> 'Dataflow':
        _log_not_supported_api_usage_and_raise(get_logger(), "_path_to_create_dataset_block", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _create_dataset_files(path) -> 'Dataflow':
        _log_not_supported_api_usage_and_raise(get_logger(), "_create_dataset_files", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _datetime_for_message(dt: datetime):
        _log_not_supported_api_usage_and_raise(get_logger(), "_datetime_for_message", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _ticks(dt: datetime):
        _log_not_supported_api_usage_and_raise(get_logger(), "_ticks", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _get_field_type(data):
        _log_not_supported_api_usage_and_raise(get_logger(), "_get_field_type", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def _add_replace_step(self, columns, replace_dict: Dict, error_replace_with: str = None):
        _log_not_supported_api_usage_and_raise(get_logger(), "_add_replace_step", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def _raise_if_missing_secrets(self, secrets: Dict[str, str]=None):
        _log_not_supported_api_usage_and_raise(get_logger(), "_raise_if_missing_secrets", "Deprecated, downgrade to a previous version of azureml-dataprep")

    # Steps are immutable so we don't need to create a full deepcopy of them when cloning Dataflows.
    def __deepcopy__(self, memodict=None):
        _log_not_supported_api_usage_and_raise(get_logger(), "__deep_ocpy__", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def join(left_dataflow,
             right_dataflow,
             join_key_pairs: List[Tuple[str, str]] = None,
             join_type = None,
             left_column_prefix: str = 'l_',
             right_column_prefix: str = 'r_',
             left_non_prefixed_columns: List[str] = None,
             right_non_prefixed_columns: List[str] = None) -> 'Dataflow':
        """
        Creates a new Dataflow that is a result of joining two provided Dataflows.

        :param left_dataflow: Left Dataflow or DataflowReference to join with.
        :param right_dataflow: Right Dataflow or DataflowReference to join with.
        :param join_key_pairs: Key column pairs. List of tuples of columns names where each tuple forms a key pair to
            join on. For instance: [('column_from_left_dataflow', 'column_from_right_dataflow')]
        :param join_type: Type of join to perform. Match is default.
        :param left_column_prefix: Prefix to use in result Dataflow for columns originating from left_dataflow.
            Needed to avoid column name conflicts at runtime.
        :param right_column_prefix: Prefix to use in result Dataflow for columns originating from right_dataflow.
            Needed to avoid column name conflicts at runtime.
        :param left_non_prefixed_columns: List of column names from left_dataflow that should not be prefixed with
            left_column_prefix. Every other column appearing in the data at runtime will be prefixed.
        :param right_non_prefixed_columns: List of column names from right_dataflow that should not be prefixed with
            left_column_prefix. Every other column appearing in the data at runtime will be prefixed.
        :return: The new Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "join", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def _get_path_from_step(step_type, step_arguments):
        _log_not_supported_api_usage_and_raise(get_logger(), "_get_path_from_step", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def reference(reference: 'DataflowReference') -> 'Dataflow':
        """
        Creates a reference to an existing activity object.

        :param reference: The reference activity.
        :return: A new Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "reference", "Deprecated, downgrade to a previous version of azureml-dataprep")

    @staticmethod
    def read_parquet_dataset(path) -> 'Dataflow':
        """
        Creates a step to read parquet file.

        :param path: The path to the Parquet file.
        :return: A new Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "read_parquet_dataset", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def to_data_frame_directory(self,
                                error: str = 'ERROR',
                                rows_per_group: int = 5000) -> 'Dataflow':
        """
        Creates streams with the data in dataframe directory format.

        :param error: String to use for error values.
        :param rows_per_group: Number of rows to use per row group.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "to_data_frame_directory", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def null_coalesce(self,
                      columns: List[str],
                      new_column_id: str) -> 'Dataflow':
        """
        For each record, selects the first non-null value from the columns specified and uses it as the value of a new column.

        :param columns: The source columns.
        :param new_column_id: The name of the new column.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "null_coalesce", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def extract_error_details(self,
                              column: str,
                              error_value_column: str,
                              extract_error_code: bool = False,
                              error_code_column: Optional[str] = None) -> 'Dataflow':
        """
        Extracts the error details from error values into a new column.

        :param column: The source column.
        :param error_value_column: Name of a column to hold the original value of the error.
        :param extract_error_code: Whether to also extract the error code.
        :param error_code_column: Name of a column to hold the error code.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "extract_error_details", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def duplicate_column(self,
                         column_pairs: Dict[str, str]) -> 'Dataflow':
        """
        Creates new columns that are duplicates of the specified source columns.

        :param column_pairs: Mapping of the columns to duplicate to their new names.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "duplicate_column", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def split_stype(self,
                    column: str,
                    stype,
                    stype_fields: Optional[List[str]] = None,
                    new_column_names: Optional[List[str]] = None) -> 'Dataflow':
        """
        Creates new columns from an existing column, interpreting its values as a semantic type.

        :param column: The source column.
        :param stype: The semantic type used to interpret values in the column.
        :param stype_fields: Fields of the semantic type to use. If not provided, all fields will be used.
        :param new_column_names: Names of the new columns. If not provided new columns will be named with the source column name plus the semantic type field name.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "split_stype", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def replace_na(self,
                   columns,
                   use_default_na_list: bool = True,
                   use_empty_string_as_na: bool = True,
                   use_nan_as_na: bool = True,
                   custom_na_list: Optional[str] = None) -> 'Dataflow':
        """
        Replaces values in the specified columns with nulls. You can choose to use the default list, supply your own, or both.

        :param use_default_na_list: Use the default list and replace 'null', 'NaN', 'NA', and 'N/A' with null.
        :param use_empty_string_as_na: Replace empty strings with null.
        :param use_nan_as_na: Replace NaNs with Null.
        :param custom_na_list: Provide a comma separated list of values to replace with null.
        :param columns: The source columns.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "replace_na", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def trim_string(self,
                    columns,
                    trim_left: bool = True,
                    trim_right: bool = True,
                    trim_type = None,
                    custom_characters: str = '') -> 'Dataflow':
        """
        Trims string values in specific columns.

        :param columns: The source columns.
        :param trim_left: Whether to trim from the beginning.
        :param trim_right: Whether to trim from the end.
        :param trim_type: Whether to trim whitespace or custom characters.
        :param custom_characters: The characters to trim.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "trim_string", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def round(self,
              column: str,
              decimal_places: int) -> 'Dataflow':
        """
        Rounds the values in the column specified to the desired number of decimal places.

        :param column: The source column.
        :param decimal_places: The number of decimal places.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "round", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def clip(self,
             columns,
             lower: Optional[float] = None,
             upper: Optional[float] = None,
             use_values: bool = True) -> 'Dataflow':
        """
        Clips values so that all values are between the lower and upper boundaries.

        :param columns: The source columns.
        :param lower: All values lower than this value will be set to this value.
        :param upper: All values higher than this value will be set to this value.
        :param use_values: If true, values outside boundaries will be set to the boundary values. If false, those values will be set to null.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "clip", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def str_replace(self,
                    columns,
                    value_to_find: Optional[str] = None,
                    replace_with: Optional[str] = None,
                    match_entire_cell_contents: bool = False,
                    use_special_characters: bool = False) -> 'Dataflow':
        """
        Replaces values in a string column that match a search string with the specified value.

        :param columns: The source columns.
        :param value_to_find: The value to find.
        :param replace_with: The replacement value.
        :param match_entire_cell_contents: Whether the value to find must match the entire value.
        :param use_special_characters: If checked, you can use '#(tab)', '#(cr)', or '#(lf)' to represent special characters in the find or replace arguments.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "str_replace", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def drop_nulls(self,
                   columns,
                   column_relationship = None) -> 'Dataflow':
        """
        Drops rows where all or any of the selected columns are null.

        :param columns: The source columns.
        :param column_relationship: Whether all or any of the selected columns must be null.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "drop_nulls", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def drop_errors(self,
                    columns,
                    column_relationship = None) -> 'Dataflow':
        """
        Drops rows where all or any of the selected columns are an Error.

        :param columns: The source columns.
        :param column_relationship: Whether all or any of the selected columns must be an Error.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "drop_errors", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def promote_headers(self) -> 'Dataflow':
        """
        Sets the first record in the dataset as headers, replacing any existing ones.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "promote_headers", "use one of the factory methods to read the data with headers instead.")

    def convert_unix_timestamp_to_datetime(self,
                                           columns,
                                           use_seconds: bool = False) -> 'Dataflow':
        """
        Converts the specified column to DateTime values by treating the existing value as a Unix timestamp.

        :param columns: The source columns.
        :param use_seconds: Whether to use seconds as the resolution. Milliseconds are used if false.
        :return: The modified Dataflow.
        """
        return _log_not_supported_api_usage_and_raise(get_logger(), "convert_unix_timestamp_to_datetime", 'Consider converting numeric column to date in Pandas.')

    def _summarize(self,
                   summary_columns = None,
                   group_by_columns: Optional[List[str]] = None,
                   join_back: bool = False,
                   join_back_columns_prefix: Optional[str] = None) -> 'Dataflow':
        """
        Summarizes data by running aggregate functions over specific columns. The aggregate functions are independent and it is possible to aggregate
            the same column multiple times. Unique names have to be provided for the resulting columns. The aggregations can be grouped, in which
            case one record is returned per group; or ungrouped, in which case one record is returned for the whole dataset. Additionally, the
            results of the aggregations can either replace the current dataset or augment it by appending the result columns.

        :param summary_columns: Column summarization definition.
        :param group_by_columns: Columns to group by.
        :param join_back: Whether to append subtotals or replace current data with them.
        :param join_back_columns_prefix: Prefix to use for subtotal columns when appending them to current data.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "_summarize", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def _summarize_each(self,
                        summary_functions = None,
                        group_by_columns: Optional[List[str]] = None) -> 'Dataflow':
        _log_not_supported_api_usage_and_raise(get_logger(), "_summarize_each", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def append_columns(self,
                       dataflows,
                       parallelize: bool = True) -> 'Dataflow':
        """
        Appends the columns from the referenced dataflows to the current one. Duplicate columns will result in failure.

        :param dataflows: The dataflows to append.
        :param parallelize: Whether to parallelize the operation. If true, the data for all inputs will be loaded into memory.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "append_columns", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def append_rows(self,
                    dataflows) -> 'Dataflow':
        """
        Appends the records in the specified dataflows to the current one. If the schemas of the dataflows are distinct, this will result in records
            with different schemas.

        :param dataflows: The dataflows to append.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "append_rows", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def sort(self,
             sort_order: List[Tuple[str, bool]]) -> 'Dataflow':
        """
        Sorts the dataset by the specified columns.

        :param sort_order: The columns to sort by and whether to sort ascending or descending. True is treated as descending, false as ascending.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "sort", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def zip_partitions(self,
                       dataflows: List['DataflowReference']) -> 'Dataflow':
        """
        Appends the columns from the referenced dataflows to the current one. This is different from AppendColumns in that it assumes all dataflows
            being appended have the same number of partitions and same number of Records within each corresponding partition. If these two
            conditions are not true the operation will fail.

        :param dataflows: The dataflows to append.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "zip_partitions", "Deprecated, downgrade to a previous version of azureml-dataprep")

    def parse_json_column(self,
                          column: str) -> 'Dataflow':
        """
        Parses the values in the specified column as JSON objects and expands them into multiple columns.

        :param column: The source column.
        :return: The modified Dataflow.
        """
        _log_not_supported_api_usage_and_raise(get_logger(), "parse_json_columns", "Deprecated, downgrade to a previous version of azureml-dataprep")
