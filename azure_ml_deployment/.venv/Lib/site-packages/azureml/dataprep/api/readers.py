# Copyright (c) Microsoft Corporation. All rights reserved.
from .dataflow import Dataflow, FilePath, DatabaseSource, SkipMode, PromoteHeadersMode
from .engineless_dataflow import EnginelessDataflow, _log_not_supported_api_usage_and_raise, _warn_if_not_default
from ._datastore_helper import NotSupportedDatastoreTypeError, _set_auth_type
from .engineapi.typedefinitions import FileEncoding, InvalidLineHandling, DatabaseAuthType, DatabaseSslMode
from ._loggerfactory import _LoggerFactory
from typing import List, Optional

logger = None

def get_logger():
    global logger
    if logger is not None:
        return logger

    logger = _LoggerFactory.get_logger("dataprep.Readers")
    return logger


def read_csv(path: FilePath,
             separator: str = ',',
             header: PromoteHeadersMode = PromoteHeadersMode.CONSTANTGROUPED,
             encoding: FileEncoding = FileEncoding.UTF8,
             quoting: bool = False,
             inference_arguments = None,
             skip_rows: int = 0,
             skip_mode: SkipMode = SkipMode.NONE,
             comment: str = None,
             include_path: bool = False,
             archive_options = None,
             infer_column_types: bool = False,
             verify_exists: bool = True,
             partition_size: Optional[int] = None,
             empty_as_string: bool = False) -> Dataflow:
    """
    Creates a new Dataflow with the operations required to read CSV and other delimited text files (TSV, custom delimiters like semicolon, colon etc.).

    :param path: The path to the file(s) or folder(s) that you want to load. It can either be a local path or an Azure Blob url.
        Globbing is supported. For example, you can use path = "./data*" to read all files with name starting with "data".
    :param separator: The separator character to use to split columns.
    :param header: The mode in which header is promoted. The options are: `PromoteHeadersMode.CONSTANTGROUPED`, `PromoteHeadersMode.GROUPED`, `PromoteHeadersMode.NONE`, `PromoteHeadersMode.UNGROUPED`.
        The default is `PromoteHeadersMode.CONSTANTGROUPED`, which assumes all files have the same schema by promoting the first row of the first file as header, and dropping the first row of the rest of the files.
        `PromoteHeadersMode.GROUPED` will promote the first row of each file as header and aggregate the result.
        `PromoteHeadersMode.NONE` will not promote header.
        `PromoteHeadersMode.UNGROUPED` will promote only the first row of the first file as header.
    :param encoding: The encoding of the files being read.
    :param quoting: Whether to handle new line characters within quotes. The default is to interpret the new line characters as starting new rows,
        irrespective of whether the characters are within quotes or not. If set to True, new line characters inside quotes will not result in new rows, and file reading speed will slow down.
    :param inference_arguments: (Deprecated) Arguments that determine how data types are inferred.
        For example, to deal with ambiguous date format, you can specify inference_arguments = dprep.InferenceArguments(day_first = False)). Date values will then be read as MM/DD.
        Note that DataPrep will also attempt to infer and convert other column types.
    :param skip_rows: (Deprecated) How many rows to skip in the file(s) being read.
    :param skip_mode: (Deprecated) The mode in which rows are skipped. The options are: SkipMode.NONE, SkipMode.UNGROUPED, SkipMode.GROUPED.
        SkipMode.NONE (Default) Do not skip lines. Note that, if `skip_rows` is provided this is ignored and `SkipMode.UNGROUPED` is used instead.
        SkipMode.UNGROUPED will skip only for the first file. SkipMode.GROUPED will skip for every file.
    :param comment: (Deprecated) Character used to indicate a line is a comment instead of data in the files being read. Comment character has to be the first character of the row to be interpreted.
    :param include_path: Whether to include a column containing the path from which the data was read.
        This is useful when you are reading multiple files, and might want to know which file a particular record is originated from, or to keep useful information in file path.
    :param archive_options: (Deprecated) Options for archive file, including archive type and entry glob pattern. We only support ZIP as archive type at the moment.
        For example, by specifying archive_options = ArchiveOptions(archive_type = ArchiveType.ZIP, entry_glob = '*10-20.csv'), Dataprep will read all files with name ending with "10-20.csv" in ZIP.
    :param infer_column_types:(Deprecated) Attempt to infer columns types based on data. Apply column type conversions accordingly.
    :type infer_column_types: bool
    :param verify_exists: Checks that the file referenced exists and can be accessed by the current context. You can set this to False when creating Dataflows in an environment that does not have access
        to the data, but will be executed in an environment that does have access.
    :param partition_size: The desired partition size in bytes. Text readers parallelize their work by splitting the
        input into partitions which can be worked on independently. This parameter makes it possible to customize the
        size of those partitions. The minimum accepted value is 4 MB (4 * 1024 * 1024).
    :param empty_as_string: Whether to keep empty field values as empty strings. Default is read them as null.
    :return: A new Dataflow.
    """
    _warn_if_not_default('read_csv', 'skip_rows', skip_rows, 0)
    _warn_if_not_default('read_csv', 'skip_mode', skip_mode, SkipMode.NONE)
    _warn_if_not_default('read_csv', 'comment', comment, None)
    _warn_if_not_default('read_csv', 'archive_options', archive_options, None)
    _warn_if_not_default('read_csv', 'infer_column_types', infer_column_types, False)
    _warn_if_not_default('read_csv', 'inference_arguments', inference_arguments, None)
    df = EnginelessDataflow.from_paths(path)
    df = df.parse_delimited(separator=separator,
                            headers_mode=header,
                            encoding=encoding,
                            quoting=quoting,
                            partition_size=partition_size,
                            empty_as_string=empty_as_string,
                            inlucde_path=include_path)

    if verify_exists:
        df.verify_has_data()
    return df


def read_fwf(path: FilePath,
             offsets: List[int],
             header: PromoteHeadersMode = PromoteHeadersMode.CONSTANTGROUPED,
             encoding: FileEncoding = FileEncoding.UTF8,
             inference_arguments = None,
             skip_rows: int = 0,
             skip_mode: SkipMode = SkipMode.NONE,
             include_path: bool = False,
             infer_column_types: bool = False,
             verify_exists: bool = True) -> Dataflow:
    """
    (DEPRECATED) Creates a new Dataflow with the operations required to read fixed-width data.
    """
    _log_not_supported_api_usage_and_raise('read_fwf', suggestion='Use read_csv instead.')


def read_excel(path: FilePath,
               sheet_name: str = None,
               use_column_headers: bool = False,
               inference_arguments = None,
               skip_rows: int = 0,
               include_path: bool = False,
               infer_column_types: bool = False,
               verify_exists: bool = True) -> Dataflow:
    """
    (DEPRECATED) Creates a new Dataflow with the operations required to read Excel files.
    """
    _log_not_supported_api_usage_and_raise('read_excel')


def read_lines(path: FilePath,
               header: PromoteHeadersMode = PromoteHeadersMode.NONE,
               encoding: FileEncoding = FileEncoding.UTF8,
               skip_rows: int = 0,
               skip_mode: SkipMode = SkipMode.NONE,
               comment: str = None,
               include_path: bool = False,
               verify_exists: bool = True,
               partition_size: Optional[int] = None) -> Dataflow:
    """
    (DEPRECATED) Creates a new Dataflow with the operations required to read text files and split them into lines.
    """
    _log_not_supported_api_usage_and_raise('read_lines', suggestion='Use read_csv instead.')


def detect_file_format(path: FilePath):
    """
    (DEPRECATED) Analyzes the file(s) at the specified path and attempts to determine the type of file and the arguments required
        to read it. The result is a FileFormatBuilder which contains the results of the analysis.
        This method may fail due to unsupported file format. And you should always inspect the returned builder to ensure that it is as expected.
    """
    _log_not_supported_api_usage_and_raise('detect_file_format')


def smart_read_file(path: FilePath, include_path: bool = False) -> Dataflow:
    """
    (DEPRECATED) Analyzes the file(s) at the specified path and returns a new Dataflow containing the operations required to
        read them. The type of the file and the arguments required to read it are inferred automatically.
    """
    _log_not_supported_api_usage_and_raise('smart_read_file', suggestion='Use dedicated read methods instead.')


def auto_read_file(path: FilePath, include_path: bool = False) -> Dataflow:
    """
    (DEPRECATED) Analyzes the file(s) at the specified path and returns a new Dataflow containing the operations required to
        read them. The type of the file and the arguments required to read it are inferred automatically.
        If this method fails or produces results not as expected, you may consider using :func:`azureml.dataprep.detect_file_format` or other read methods with file types specified.
    """
    _log_not_supported_api_usage_and_raise('auto_read_file', suggestion='Use dedicated read methods instead.')

def read_sql(data_source: DatabaseSource, query: str, query_timeout: int = 30) -> EnginelessDataflow:
    """
    Creates a new Dataflow that can read data from a Microsoft SQL or Azure SQL database by executing the query specified.

    :param data_source: The details of the Microsoft SQL or Azure SQL database.
    :param query: The query to execute to read data.
    :param query_timeout: Sets the wait time (in seconds) before terminating the attempt to execute a command
        and generating an error. The default is 30 seconds.
    :return: A new Dataflow.
    """
    try:
        from azureml.data.abstract_datastore import AbstractDatastore
        from azureml.data.azure_sql_database_datastore import AzureSqlDatabaseDatastore

        if isinstance(data_source, AzureSqlDatabaseDatastore):
            _set_auth_type(data_source.workspace)
            handler_arguments = {
                'query_timeout': query_timeout,
                'datastore_name': data_source.name,
                'subscription': data_source.workspace.subscription_id,
                'resource_group': data_source.workspace.resource_group,
                'workspace_name': data_source.workspace.name }
            return EnginelessDataflow.from_query_source('AmlDatastore', query, handler_arguments)
        if isinstance(data_source, AbstractDatastore):
            raise NotSupportedDatastoreTypeError(data_source)
    except ImportError:
        pass
    if data_source.credentials_type.value != DatabaseAuthType.SERVER.value:
        _log_not_supported_api_usage_and_raise('read_sql', suggestion='Only server authentication is supported for SQL Server and Azure SQL Database.')
    # keeping this for dataprep e2e tests to work
    server_auth = {'auth_type': 0, 'login': data_source.user_name, 'password': data_source.password}
    handler_arguments = {'query_timeout': query_timeout, 'server': data_source.server, 'database': data_source.database, 'trust_server': data_source.trust_server, 'server_auth': server_auth}
    return EnginelessDataflow.from_query_source('MSSQL', query, handler_arguments)


def read_postgresql(data_source: DatabaseSource, query: str, query_timeout: int = 20) -> Dataflow:
    """
    Creates a new Dataflow that can read data from a PostgreSQL database by executing the query specified.

    :param data_source: The details of the PostgreSQL database.
    :param query: The query to execute to read data.
    :param query_timeout: Sets the wait time (in seconds) before terminating the attempt to execute a command
        and generating an error. The default is 20 seconds.
    :return: A new Dataflow.
    """
    try:
        from azureml.data.abstract_datastore import AbstractDatastore
        from azureml.data.azure_postgre_sql_datastore import AzurePostgreSqlDatastore

        if isinstance(data_source, AzurePostgreSqlDatastore):
            handler_arguments = {
                'query_timeout': query_timeout,
                'datastore_name': data_source.name,
                'subscription': data_source.workspace.subscription_id,
                'resource_group': data_source.workspace.resource_group,
                'workspace_name': data_source.workspace.name }
            return EnginelessDataflow.from_query_source('AmlDatastore', query, handler_arguments)
        if isinstance(data_source, AbstractDatastore):
            raise NotSupportedDatastoreTypeError(data_source)
    except ImportError:
        pass
     # keeping this for dataprep e2e tests to work
    server_auth = {'user_id': data_source.user_name, 'user_password': data_source.password}
    ssl_mode = 'prefer' if data_source.ssl_mode.value == DatabaseSslMode.PREFER.value else 'require' if data_source.ssl_mode.value == DatabaseSslMode.REQUIRE.value else 'disable'
    handler_arguments = {'query_timeout': query_timeout,
                         'server': data_source.server,
                         'database': data_source.database,
                         'server_auth': server_auth,
                         'port': int(data_source.port_number),
                         'ssl_mode': ssl_mode}
    return EnginelessDataflow.from_query_source('PostgreSQL', query, handler_arguments)


def read_parquet_file(path: FilePath,
                      include_path: bool = False,
                      verify_exists: bool = True) -> EnginelessDataflow:
    """
    Creates a new Dataflow with the operations required to read Parquet files.

    :param path: The path to the file(s) or folder(s) that you want to load. It can either be a local path or an Azure Blob url.
        Globbing is supported. For example, you can use path = "./data*" to read all files with name starting with "data".
    :param include_path: Whether to include a column containing the path from which the data was read.
        This is useful when you are reading multiple files, and might want to know which file a particular record is originated from, or to keep useful information in file path.
    :param verify_exists: Checks that the file referenced exists and can be accessed by the current context. You can set this to False when creating Dataflows in an environment that does not have access
        to the data, but will be executed in an environment that does have access.
    :return: A new Dataflow.
    """
    df = EnginelessDataflow.from_paths(path)

    df = df.read_parquet_file(include_path)

    if verify_exists:
        df.verify_has_data()
    return df


def read_parquet_dataset(path: FilePath, include_path: bool = False) -> Dataflow:
    """
    (DEPRECATED) Creates a new Dataflow with the operations required to read Parquet Datasets.
    """
    _log_not_supported_api_usage_and_raise('read_parquet_dataset', suggestion='Use read_parquet_file with partition format instead')


def read_preppy(path: FilePath, include_path: bool = False, verify_exists: bool = True) -> EnginelessDataflow:
    """
    Creates a new Dataflow with the operations required to read Preppy files, a file serialization format specific to Data Prep.

    :param path: The path to the file(s) or folder(s) that you want to load. It can either be a local path or an Azure Blob url.
        Globbing is supported. For example, you can use path = "./data*" to read all files with name starting with "data".
    :param include_path: Whether to include a column containing the path from which the data was read.
        This is useful when you are reading multiple files, and might want to know which file a particular record is originated from, or to keep useful information in file path.
    :param verify_exists: Checks that the file referenced exists and can be accessed by the current context. You can set this to False when creating Dataflows in an environment that does not have access
        to the data, but will be executed in an environment that does have access.
    :return: A new Dataflow.
    """
    df = EnginelessDataflow.from_paths(path)

    df = df.read_preppy(include_path)

    if verify_exists:
        df.verify_has_data()
    return df


def read_json_lines(path: FilePath,
                    encoding: FileEncoding = FileEncoding.UTF8,
                    partition_size: Optional[int] = None,
                    include_path: bool = False,
                    verify_exists: bool = True,
                    invalid_lines: InvalidLineHandling = InvalidLineHandling.ERROR) -> EnginelessDataflow:
    """
        Creates a new Dataflow with the operations required to read JSON lines files.

        :param path: The path to the file(s) or folder(s) that you want to load. It can either be a local path or an Azure Blob url.
            Globbing is supported. For example, you can use path = "./data*" to read all files with name starting with "data".
        :param invalid_lines: How to handle invalid JSON lines.
        :param encoding: The encoding of the files being read.
        :param partition_size: The desired partition size in bytes. Text readers parallelize their work by splitting the
            input into partitions which can be worked on independently. This parameter makes it possible to customize the
            size of those partitions. The minimum accepted value is 4 MB (4 * 1024 * 1024).
        :param include_path: Whether to include a column containing the path from which the data was read.
            This is useful when you are reading multiple files, and might want to know which file a particular record is originated from, or to keep useful information in file path.
        :param verify_exists: Checks that the file referenced exists and can be accessed by the current context. You can set this to False when creating Dataflows in an environment that does not have access
            to the data, but will be executed in an environment that does have access.
    """
    df = EnginelessDataflow.from_paths(path)

    df = df.parse_json_lines(encoding=encoding, partition_size=partition_size, invalid_lines=invalid_lines, include_path=include_path)

    if verify_exists:
        df.verify_has_data()
    return df


def read_json(path: FilePath,
              encoding: FileEncoding = FileEncoding.UTF8,
              flatten_nested_arrays: bool = False,
              include_path: bool = False) -> Dataflow:
    """
    (DEPRECATED) Creates a new Dataflow with the operations required to read JSON files.
    """
    _log_not_supported_api_usage_and_raise('read_json')


def read_pandas_dataframe(df: 'pandas.DataFrame',
                          temp_folder: str = None,
                          overwrite_ok: bool = True,
                          in_memory: bool = False) -> Dataflow:
    """
    (DEPRECATED) Creates a new Dataflow based on the contents of a given pandas DataFrame.
    """
    _log_not_supported_api_usage_and_raise('read_pandas_dataframe', suggestion='Write dataframe out as Parquet or csv and use read_parquet or read_csv instead.')


def read_npz_file(path: FilePath,
                  include_path: bool = False,
                  verify_exists: bool = True) -> Dataflow:
    """
    (DEPRECATED) Creates a new Dataflow with the operations required to read npz files.
    """
    _log_not_supported_api_usage_and_raise('read_npz_file', suggestion='Write dataframe out as Parquet or csv and use read_parquet or read_csv instead.')


def from_file_path(path: FilePath,
                   force_file: bool = False) -> Dataflow:
    return EnginelessDataflow.from_paths(path, force_file)