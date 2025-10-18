import json
import os

from azureml.dataprep.api._loggerfactory import _LoggerFactory

from ._aml_helper import get_workspace_from_run, verify_workspace


logger = None

def get_logger():
    global logger
    if logger is not None:
        return logger

    logger = _LoggerFactory.get_logger("dprep._aml_auth_resolver")
    return logger


class WorkspaceContextCache:
    _cache = {}

    @staticmethod
    def add(workspace):
        try:
            key = WorkspaceContextCache._get_key(workspace.subscription_id, workspace.resource_group, workspace.name)
            WorkspaceContextCache._cache[key] = workspace
        except Exception as e:
            get_logger().info('Cannot cache workspace due to: {}'.format(repr(e)))

    @staticmethod
    def get(subscription_id, resource_group_name, workspace_name):
        try:
            key = WorkspaceContextCache._get_key(subscription_id, resource_group_name, workspace_name)
            workspace = WorkspaceContextCache._cache[key]
            return workspace
        except Exception as e:
            get_logger().info('Cannot find cached workspace due to: {}'.format(repr(e)))
            return None

    @staticmethod
    def _get_key(subscription_id, resource_group_name, workspace_name):
        return ''.join([subscription_id, resource_group_name, workspace_name])

def _resolve_auth_from_registry(registry_name, auth_type, creds):
    # azureml-core is not installed try aml run token auth for remote aml run
    return _resolve_auth_from_run_token_or_mlclient(auth_type, creds, None, None, None, registry_name)


def _resolve_auth_for_workspace_access(subscription, resource_group, ws_name, auth_type, creds):
    # There are two ways to resolve auth 1) from v1 workspace  2) from v2 run token or mlclient
    # The logic is as follows:
    # 1) If azureml.core is not installed, try resolve from v2 run token or mlclient
    # 2) If ml_client is not installed, try resolve from v1 workspace
    # 3) If both packages are installed, try to resolve from v2 run token or mlclient first, 
    # customer auth and ml client will be picked up at this step.
    # If failed, try resolve from v1 workspace.
    try:
        # if azureml.core's workspace is not installed, try resolve from run token or mlclient
        from azureml.core import Workspace
        from azureml._base_sdk_common.service_discovery import get_service_url
    except ImportError:
        get_logger().info('azureml-core is not installed,, try resolve with run token or mlclient')
        return _resolve_auth_from_run_token_or_mlclient(auth_type, creds, subscription, resource_group, ws_name)

    try:
        # if v2 sdk is not installed, resolve auth from workspace
        from azure.ai.ml import MLClient
    except ImportError:
        get_logger().info('MLClient is not installed, try resolve from workspace')
        return _resolve_auth_from_workspace(subscription, resource_group, ws_name, auth_type, creds)

    # if both packages are installed, check if creds is provided, use it to resolve with run token or mlclient
    try:
        get_logger().info('both azureml-core and mlclient are installed, try resolve from run token or mlclient first')
        return _resolve_auth_from_run_token_or_mlclient(auth_type, creds, subscription, resource_group, ws_name)
    except Exception as e:
        get_logger().info(f'failed to resolve auth from run token or mlclient due to: {e}; will try to resolve from workspace')

    return _resolve_auth_from_workspace(subscription, resource_group, ws_name, auth_type, creds)


def _resolve_auth_from_workspace(subscription, resource_group, ws_name, auth_type, creds):
    from azureml.core import Workspace
    from azureml._base_sdk_common.service_discovery import get_service_url
    
    if not ws_name or not subscription or not resource_group:
        raise ValueError('Invalid workspace details.')

    ws = WorkspaceContextCache.get(subscription, resource_group, ws_name)

    if not ws:
        get_logger().info("get workspace from run")
        ws = get_workspace_from_run()

    if not ws:
        get_logger().info("get workspace from auth config")
        auth = _get_auth_with_azureml_core_authentication(creds, auth_type)
        ws = Workspace.get(ws_name, auth=auth, subscription_id=subscription, resource_group=resource_group)

    verify_workspace(ws, subscription, resource_group, ws_name)

    try:
        host = os.environ.get('AZUREML_SERVICE_ENDPOINT') or \
               get_service_url(ws._auth, ws.service_context._get_workspace_scope(), ws._workspace_id,
                               ws.discovery_url)
    except AttributeError:
        # This check is for backward compatibility, handling cases where azureml-core package is pre-Feb2020,
        # as ws.discovery_url was added in this PR:
        # https://msdata.visualstudio.com/Vienna/_git/AzureMlCli/pullrequest/310794
        host = get_service_url(ws._auth, ws.service_context._get_workspace_scope(), ws._workspace_id)

    return {'service_endpoint': host, 'auth': ws._auth.get_authentication_header()}


def _resolve_auth_from_run_token_or_mlclient(auth_type, creds, subscription=None, resource_group=None, ws_name=None,
                                             registry_name=None):

    from ._aml_auth._azureml_token_authentication import AzureMLTokenAuthentication
    aml_auth = AzureMLTokenAuthentication._initialize_aml_token_auth()
    if aml_auth:
        get_logger().info('azureml-core is not installed, AML run token auth returned successfully')
        return \
            {
                'service_endpoint': os.environ.get('AZUREML_SERVICE_ENDPOINT'),
                'auth': aml_auth.get_authentication_header()
            }
    else:
        # this is a local job experience, use DefaultAzureCredential and get service endpoint from sdkv2
        get_logger().info('azureml-core is not installed and AML run token auth is None,'
                          'trying to get service endpoint from sdkv2')
        return _resolve_auth_from_mlclient(auth_type, creds, subscription, resource_group, ws_name, registry_name)


def _resolve_auth_from_mlclient(auth_type, creds, subscription=None, resource_group=None, ws_name=None,
                                registry_name=None):
    try:
        from azure.ai.ml import MLClient
    except ImportError:
        raise ImportError('The support of datastore requires dependency on azureml sdk v2 package,'
                          'please install package azure-ai-ml')

    try:
        from azureml.dataprep.api._datastore_helper import _get_ml_cient
        customer_ml_client = _get_ml_cient()
        if customer_ml_client and isinstance(customer_ml_client, MLClient):
            return get_auth_from_ml_client(customer_ml_client, registry_name)
    except Exception as e:
        get_logger().info(f'user passed ml_client failed with error {e}, trying to create new ml_client')
    
    credential = get_auth_with_azure_identity(auth_type, creds)
    ml_client = MLClient(credential, subscription, resource_group, ws_name, registry_name)
    return get_auth_from_ml_client(ml_client, registry_name)

def get_auth_from_ml_client(ml_client, registry_name):
    credential = ml_client._credential
    auth_object = \
        {
            'auth': {
                "Authorization":
                # get arm scope token
                # getting arm resource url from ml_client which already handles
                # different clouds
                    "Bearer " + credential.get_token(f'{ml_client._base_url}/.default').token
            }
        }

    if registry_name is None:
        # for none registry case, get service_endpoint
        auth_object['service_endpoint'] = ml_client.jobs._api_url
    return auth_object

def register_datastore_resolver(requests_channel):
    def resolve(request, writer, socket):
        from azureml.exceptions import RunEnvironmentException

        try:
            auth_type = request.get('auth_type')
            ws_name = request.get('ws_name')
            subscription = request.get('subscription')
            resource_group = request.get('resource_group')
            extra_args = json.loads(request.get('extra_args') or '{}')

            auth_info = _resolve_auth_for_workspace_access(subscription, resource_group, ws_name, auth_type, extra_args)
            writer.write(json.dumps({
                'result': 'success',
                'auth': json.dumps(auth_info['auth']),
                'host': auth_info['service_endpoint']
            }))
        except ValueError:
            writer.write(json.dumps({'result': 'error', 'error': 'InvalidWorkspace'}))
        except RunEnvironmentException as e:
            writer.write(json.dumps({
                'result': 'error',
                'error': 'Exception trying to get workspace information from the run. Error: {}'.format(e.message)
            }))
        except Exception as e:
            writer.write(json.dumps({'result': 'error', 'error': str(e)}))

    requests_channel.register_handler('resolve_auth_from_workspace', resolve)


def _get_auth_with_azureml_core_authentication(creds, auth_type):
    from azureml.core.authentication import InteractiveLoginAuthentication, AzureCliAuthentication, \
        ServicePrincipalAuthentication, MsiAuthentication
    from azureml.core import VERSION
    import azureml.dataprep as dprep

    def log_version_issues(exception):
        log.warning(
            "Failed to construct auth object. Exception : {}, AML Version: {}, DataPrep Version: {}".format(
                type(exception).__name__, VERSION, dprep.__version__
            )
        )

    log = get_logger()
    cloud = creds.get('cloudType')
    tenant_id = creds.get('tenantId')
    auth_class = creds.get('authClass')

    if auth_type == 'SP':
        sp_id = creds.get('servicePrincipalId')
        get_logger().info(
            'Getting authentication with Service Principal (id={}).'.format(sp_id))
        if not creds or not tenant_id:
            raise ValueError("InvalidServicePrincipalCreds")
        try:
            return ServicePrincipalAuthentication(tenant_id, sp_id,
                                                  creds.get('password'), cloud)
        except Exception as e:
            log_version_issues(e)
            return ServicePrincipalAuthentication(tenant_id, sp_id, creds.get('password'))
    elif auth_type == 'Managed':
        get_logger().info('Getting authentication with managed identity.')
        try:
            return MsiAuthentication(cloud)
        except Exception as e:
            log_version_issues(e)
            return MsiAuthentication()

    if auth_class == AzureCliAuthentication.__name__:
        get_logger().info('Getting authentication with AzureCliAuthentication.')
        try:
            return AzureCliAuthentication(cloud=cloud)
        except Exception as e:
            log_version_issues(e)
            return AzureCliAuthentication()

    # fallback to interactive authentication which has internal authentication fallback
    try:
        return InteractiveLoginAuthentication(tenant_id=tenant_id, cloud=cloud)
    except Exception as e:
        log_version_issues(e)
        return InteractiveLoginAuthentication()


def get_auth_with_azure_identity(auth_type, creds):
    tenant_id = creds.get('tenantId')
    authority = creds.get('authority')
    cloud = creds.get('cloudType')
    if authority is None:
        authority = get_authority_from_cloud_type(cloud)
    if auth_type == 'SP':
        if not tenant_id:
            raise ValueError("InvalidServicePrincipalCreds")
        from azure.identity import ClientSecretCredential
        sp_id = creds.get('servicePrincipalId')
        get_logger().info(f'Getting data access token with Service Principal (id={sp_id}).')
        credential = ClientSecretCredential(
            tenant_id, sp_id, creds.get('password'), authority=authority)
    elif auth_type == 'Managed':
        client_id = creds.get("clientId")
        endpoint_type = creds.get("endpointType")
        get_logger().info(
            f'Getting with Assigned Identity (client_id={client_id}) and endpoint type'
            f' {endpoint_type}')
        if endpoint_type == 'Imds':
            from azure.identity._credentials.imds import ImdsCredential
            credential = ImdsCredential(client_id=client_id)
        elif endpoint_type == 'MsiEndpoint':
            from azure.identity._credentials.app_service import AppServiceCredential
            credential = AppServiceCredential(client_id=client_id)
        else:
            from azure.identity import ManagedIdentityCredential
            credential = ManagedIdentityCredential(client_id=client_id)
    elif auth_type == 'Custom':
        credential = creds.get("credential")
    else:
        from azure.identity import DefaultAzureCredential
        get_logger().info('Getting DefaultAzureCredential.')
        credential = DefaultAzureCredential(authority=authority)
    return credential


def get_authority_from_cloud_type(cloud=None):
    cloudType_authority = {
        'AzureChinaCloud': 'login.chinacloudapi.cn',
        'AzureGermanCloud': 'login.microsoftonline.de',
        'AzureUSGovernment': 'login.microsoftonline.us',
        'AzureCloud': 'login.microsoftonline.com'
    }
    authority = None
    if cloud is not None:
        authority = cloudType_authority.get(cloud)
    return authority

