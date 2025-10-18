# Copyright (c) Microsoft Corporation. All rights reserved.
from .engineapi.typedefinitions import AuthType
from ... import dataprep
from typing import TypeVar, List
from ._loggerfactory import _log_not_supported_api_usage_and_raise, _LoggerFactory

_logger = _LoggerFactory.get_logger('_datastore_helper')

DEFAULT_SAS_DURATION = 30  # this aligns with our SAS generation in the UI BlobStorageManager.ts
Datastore = TypeVar('Datastore', 'AbstractDatastore', 'DataReference', 'DataPath')
Datastores = TypeVar('Datastores', Datastore, List[Datastore])

def file_datastores_to_uris(data_sources: List[Datastore]) -> List[str]:
    datastore_uris = []
    def encode_path(path):
        from urllib.parse import quote
        path_parts = path.split('?')
        # quote the path, but keep / and * intact
        path_parts[0] = quote(path_parts[0], '/*')
        return '?'.join(path_parts)
    for source in data_sources:
        datastore, datastore_value = get_datastore_value(source)
        if not _is_fs_datastore(datastore):
            raise NotSupportedDatastoreTypeError(datastore)
        datastore_uris.append("azureml://subscriptions/{}/resourcegroups/{}/workspaces/{}/datastores/{}/paths/{}"
                              .format(datastore_value.subscription,
                                      datastore_value.resource_group,
                                      datastore_value.workspace_name,
                                      datastore_value.datastore_name,
                                      encode_path(datastore_value.path).lstrip('/')))
    return datastore_uris

def datastore_to_dataflow(data_source: Datastores, query_timeout: int=-1, is_file: bool=False) -> 'dataprep.Dataflow':
    _log_not_supported_api_usage_and_raise(_logger, 'datastore_to_dataflow', 'Datastore to Dataflow not directly supported anymore due to Clex engine deprecation')


def get_datastore_value(data_source: Datastore) -> ('AbstractDatastore', 'dataprep.api.dataflow.DatastoreValue'):
    from .dataflow import DatastoreValue
    try:
        from azureml.data.abstract_datastore import AbstractDatastore
        from azureml.data.data_reference import DataReference
        from azureml.data.datapath import DataPath
    except ImportError as e:
        raise _decorate_import_error(e)

    datastore = None
    path_on_storage = ''

    if isinstance(data_source, AbstractDatastore):
        datastore = data_source
    elif isinstance(data_source, DataReference):
        datastore = data_source.datastore
        path_on_storage = data_source.path_on_datastore or path_on_storage
    elif isinstance(data_source, DataPath):
        datastore = data_source._datastore
        path_on_storage = data_source.path_on_datastore or path_on_storage
    else:
        from azureml.exceptions import UserErrorException
        raise UserErrorException(f'Data source path {data_source} is unsupported. Expected a DataPath, DataReference, or Datastore')

    _ensure_supported(datastore)

    workspace = datastore.workspace
    _set_auth_type(workspace)
    return (datastore, DatastoreValue(
        subscription=workspace.subscription_id,
        resource_group=workspace.resource_group,
        workspace_name=workspace.name,
        datastore_name=datastore.name,
        path=path_on_storage
    ))


def to_stream_info_pod(datastore_value: 'dataprep.api.dataflow.DatastoreValue') -> dict:
    return {
        'handler': 'AmlDatastore',
        'resourceidentifier': datastore_value.datastore_name + '/' + datastore_value.path.lstrip('/'),
        'arguments': {
            'subscription': datastore_value.subscription,
            'resourceGroup': datastore_value.resource_group,
            'workspaceName': datastore_value.workspace_name,
            'datastoreName': datastore_value.datastore_name
        }
    }


def login():
    try:
        from azureml.core.authentication import InteractiveLoginAuthentication
    except ImportError as e:
        raise _decorate_import_error(e)

    auth = InteractiveLoginAuthentication()
    auth.get_authentication_header()


def _ensure_supported(datastore: 'AbstractDatastore'):
    try:
        from azureml.data.azure_sql_database_datastore import AzureSqlDatabaseDatastore
        from azureml.data.azure_postgre_sql_datastore import AzurePostgreSqlDatastore
    except ImportError as e:
        raise _decorate_import_error(e)

    if not (_is_fs_datastore(datastore) or isinstance(datastore, (AzureSqlDatabaseDatastore, AzurePostgreSqlDatastore))):
        raise NotSupportedDatastoreTypeError(datastore)


auth_type = AuthType.DERIVED
auth_value = {}


def _get_auth():
    global auth_type
    global auth_value
    return auth_type, auth_value

def _get_ml_cient():
    global customer_ml_client
    return customer_ml_client


def _set_auth_type(workspace: 'Workspace'):
    global auth_type
    global auth_value

    from ._aml_auth_resolver import WorkspaceContextCache
    try:
        from azureml.core.authentication import ServicePrincipalAuthentication
    except ImportError as e:
        raise _decorate_import_error(e)

    WorkspaceContextCache.add(workspace)

    auth = {}
    _try_update_auth(lambda: workspace._auth._cloud_type.name, auth, 'cloudType')
    _try_update_auth(lambda: workspace._auth._tenant_id, auth, 'tenantId')
    _try_update_auth(lambda: type(workspace._auth).__name__, auth, 'authClass')
    _try_update_auth(lambda: workspace._auth._cloud_type.endpoints.active_directory, auth, 'authority')
    if isinstance(workspace._auth, ServicePrincipalAuthentication):
        auth = {
            **auth,
            'servicePrincipalId': workspace._auth._service_principal_id,
            'password': workspace._auth._service_principal_password
        }
        auth_type = AuthType.SERVICEPRINCIPAL
    else:
        auth_type = AuthType.DERIVED

    auth_value = auth


def _set_auth_from_dict(auth_dict):
    global auth_type
    global auth_value

    auth = {}
    _try_update_auth(lambda: auth_dict.get('cloudType'), auth, 'cloudType')
    _try_update_auth(lambda: auth_dict.get('tenantId'), auth, 'tenantId')
    if auth_dict.get('authority'):
        _try_update_auth(lambda: auth_dict.get('authority'), auth, 'authority')
    if auth_dict.get('authType') == 'ServicePrincipal':
        auth = {
            **auth,
            'servicePrincipalId': auth_dict.get('service_principal_id'),
            'password': auth_dict.get('service_principal_password')
        }
        auth_type = AuthType.SERVICEPRINCIPAL
    elif auth_dict.get('authType') == 'Managed':
        _try_update_auth(lambda: auth_dict.get('endpointType'), auth, 'endpointType')
        _try_update_auth(lambda: auth_dict.get('clientId'), auth, 'clientId')
        auth_type = AuthType.MANAGED
    elif auth_dict.get('authType') == 'Custom':
        _try_update_auth(lambda: auth_dict.get('credential'), auth, 'credential')
        auth_type = AuthType.CUSTOM
        auth_value = auth
        return
    else:
        auth_type = AuthType.DERIVED

    auth_value = auth

def _set_auth_ml_client(ml_client):
    global customer_ml_client
    customer_ml_client = ml_client


def map_auth_type(auth_type):
    if auth_type == AuthType.SERVICEPRINCIPAL:
        auth_type = 'SP'
    elif auth_type == AuthType.MANAGED:
        auth_type = 'Managed'
    elif auth_type == AuthType.CUSTOM:
        auth_type = 'Custom'
    else:
        auth_type = 'Derived'
    return auth_type


def _set_clould_type(cloud: "Cloud"):
    global auth_type
    global auth_value

    auth = auth_value.copy()
    _try_update_auth(lambda: cloud.name, auth, 'cloudType')
    _try_update_auth(lambda: cloud.endpoints.active_directory, auth, 'authority')
    auth_value = auth


def _try_update_auth(attr_getter, auth_dict, key):
    try:
        auth_dict[key] = attr_getter()
    except AttributeError:
        # this happens when the azureml-core SDK is old (prior to sovereign cloud changes)
        return


def _all(items, predicate):
    return len(list(filter(lambda ds: not predicate(ds), items))) == 0


def _is_datapath(data_path) -> bool:
    try:
        from azureml.data.data_reference import DataReference
        from azureml.data.abstract_datastore import AbstractDatastore
        from azureml.data.datapath import DataPath
    except ImportError as e:
        # can't be any of the above if azureml.data is not installed
        return False

    return isinstance(data_path, DataReference) or \
            isinstance(data_path, AbstractDatastore) or \
            isinstance(data_path, DataPath)


def _is_datapaths(data_paths) -> bool:
    return type(data_paths) is list and _all(data_paths, _is_datapath)


def _is_fs_datastore(datastore: 'AbstractDatastore') -> bool:
    try:
        from azureml.data.azure_storage_datastore import AzureBlobDatastore
        from azureml.data.azure_storage_datastore import AzureFileDatastore
        from azureml.data.azure_data_lake_datastore import AzureDataLakeDatastore
        from azureml.data.azure_data_lake_datastore import AzureDataLakeGen2Datastore
    except ImportError as e:
        raise _decorate_import_error(e)

    is_fs_datastore = False
    try:
        from azureml.data.hdfs_datastore import HDFSDatastore
        is_fs_datastore = is_fs_datastore or isinstance(datastore, HDFSDatastore)
    except ImportError as e:
        if datastore.datastore_type == "Hdfs":
            raise _decorate_import_error(e)

    return isinstance(datastore, AzureBlobDatastore) or \
            isinstance(datastore, AzureFileDatastore) or \
            isinstance(datastore, AzureDataLakeDatastore) or \
            isinstance(datastore, AzureDataLakeGen2Datastore) or \
            datastore.datastore_type == "Custom" or \
            is_fs_datastore


def _to_stream_info_value(datastore: 'AbstractDatastore', path_in_datastore: str = '') -> dict:
    def to_value(value):
        return {'string': value}

    return {
        'streaminfo': {
            'handler': 'AmlDatastore',
            'resourceidentifier': datastore.name + '/' + path_in_datastore.lstrip('/'),
            'arguments': {
                'subscription': to_value(datastore.workspace.subscription_id),
                'resourceGroup': to_value(datastore.workspace.resource_group),
                'workspaceName': to_value(datastore.workspace.name),
                'datastoreName': to_value(datastore.name)
            }
        }
    }


def _serialize_datastore(datastore):
    return {
        'workspace': datastore.workspace.name,
        'subscription_id': datastore.workspace.subscription_id,
        'resource_group': datastore.workspace.resource_group,
        'datastore': datastore.name
    }


def _deserialize_datastore(datastore_dict):
    try:
        from azureml.core import Workspace, Run, Datastore as DatastoreClient
    except ImportError as e:
        raise _decorate_import_error(e)

    try:
        workspace = Run.get_context().experiment.workspace
    except AttributeError:
        workspace = Workspace.get(
            name=datastore_dict['workspace'], subscription_id=datastore_dict['subscription_id'],
            resource_group=datastore_dict['resource_group']
        )
    return DatastoreClient(workspace, datastore_dict['datastore'])


def _decorate_import_error(e: ImportError) -> ImportError:
    return ImportError('Unable to import Azure Machine Learning SDK. In order to use datastore, please make '\
                       + 'sure the Azure Machine Learning SDK is installed.\n{}'.format(e))


class NotSupportedDatastoreTypeError(Exception):
    def __init__(self, datastore: 'AbstractDatastore'):
        super().__init__('Datastore "{}"\'s type "{}" is not supported.'.format(datastore.name, datastore.datastore_type))
        self.datastore = datastore
