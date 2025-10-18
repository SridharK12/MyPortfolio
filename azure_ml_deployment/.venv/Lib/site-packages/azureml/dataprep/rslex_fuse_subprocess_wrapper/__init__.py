import os
import json
import subprocess
import sys
import tempfile
import time
from typing import Dict
from azure.core.credentials import TokenCredential
from azure.identity import AzureCliCredential, DefaultAzureCredential, EnvironmentCredential, ClientSecretCredential, ManagedIdentityCredential
from azure.identity._credentials.imds import ImdsCredential
from azure.identity._credentials.azure_ml import AzureMLCredential

def build_data_asset_uri(subscription_id: str, resource_group_name: str, workspace_name: str, path: str) -> str:
    if path.startswith('azureml:') and '/' not in path:
        path_parts = path.split(':')
        if len(path_parts) >= 2 and len(path_parts) <= 3:
            name = path_parts[1]
            version = None if len(path_parts) < 3 else path_parts[2]
            assert all([val is not None for val in [subscription_id, resource_group_name, workspace_name]]), "You are using short-form asset path. Please make sure all of subscription id, resource group name, workspace name are provided."
            uri = f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group_name}/workspaces/{workspace_name}/data/{name}"
            if version is not None:
                uri += f"/versions/{version}"
            return uri
    else:
        supported_schemas = ['azureml', 'http', 'https', 'wasb', 'wasbs', 'adl', 'abfss', 'azfs']
        for schema in supported_schemas:
            if path.startswith(f'{schema}://'):
                return path
    raise ValueError("data path should be in the form of eith" +
                     "er `azureml:<data_asset_name>` " +
                     "or `azureml:<data_asset_name>:<data_asset_version>` " +
                     "or `azureml://subscriptions/<subscription_id>/resourcegroups/<resource_group_name>/workspaces/<workspace_name>/data/<data_asset_name>` " +
                     "or `azureml://subscriptions/<subscription_id>/resourcegroups/<resource_group_name>/workspaces/<workspace_name>/data/<data_asset_name>/versions/<data_asset_version>` " +
                     "or URL with one of supported schemas: http, https, wasb, wasbs, adl, abfss, azfs.")

def build_datastore_uri(subscription_id: str, resource_group_name: str, workspace_name: str, path: str) -> str:
    if ':' not in path and '/' not in path:
        # assumed to be datastore name
        return f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group_name}/workspaces/{workspace_name}/datastores/{path}"
    else:
        if path.startswith('azureml://'):
            SHORT_URL_PREFIX='azureml://datastores/'
            if path.startswith(SHORT_URL_PREFIX):
                suffix=path[len(SHORT_URL_PREFIX):]
                assert all([val is not None for val in [subscription_id, resource_group_name, workspace_name]]), "You are using short-form datastore URL. Please make sure all of subscription id, resource group name, workspace name are provided."
                return f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group_name}/workspaces/{workspace_name}/datastores/{suffix}"
            else:
                return path
    raise ValueError("datastore path should be in the form of eith" +
                     "er `<datastore_name>` " +
                     "or `azureml://datastores/<datastore_name>` " +
                     "or `azureml://datastores/<datastore_name>/paths/<path_in_datastore>` " +
                     "or `azureml://subscriptions/<subscription_id>/resourcegroups/<resource_group_name>/workspaces/<workspace_name>/datastores/<datastore_name>` " +
                     "or `azureml://subscriptions/<subscription_id>/resourcegroups/<resource_group_name>/workspaces/<workspace_name>/datastores/<datastore_name>/paths/<path_in_datastore>`.")

def serialize_azure_credential(credential: TokenCredential) -> Dict[str, str]:
    def try_get_token(credential):
        try:
            print(f"trying to get token from {type(credential).__name__}")
            # TODO: sovereign clouds https://msdata.visualstudio.com/Vienna/_workitems/edit/2499949
            credential.get_token('https://management.azure.com/.default')
        except Exception as e:
            print(f"failed to get token from {type(credential).__name__}: {e}")
            pass


    if isinstance(credential, AzureCliCredential) or credential.__class__.__module__.startswith("azure.cli"):
        return {"type": "AzureCliCredential"}

    elif isinstance(credential, DefaultAzureCredential):
        try_get_token(credential)
        underlying_credential = credential._successful_credential
        if underlying_credential is not None:
            print(f"resolved {type(credential).__name__} to {type(underlying_credential).__name__}")
            return serialize_azure_credential(underlying_credential) # recursive call
        else:
            raise ValueError(f'Failed to resolve {type(credential).__name__} to underlying credential types.')

    elif isinstance(credential, ClientSecretCredential):
        return {"type": "ClientSecretCredential", 
                "tenant_id": credential._tenant_id, 
                "client_id": credential._client_id, 
                "client_secret": credential._client_credential}

    elif isinstance(credential, EnvironmentCredential):
        return {"type": "EnvironmentCredential"}

    elif isinstance(credential, ManagedIdentityCredential):
        underlying_credential = credential._credential
        if underlying_credential is not None:
            print(f"resolved {type(credential).__name__} to {type(underlying_credential).__name__}")
            return serialize_azure_credential(underlying_credential) # recursive call
        else:
            raise ValueError(f'Failed to resolve {type(credential).__name__} to underlying credential types.')

    elif isinstance(credential, ImdsCredential):
        return {"type": "ImdsCredential"} # TODO: object_id or client_id or msi_res_id

    elif isinstance(credential, AzureMLCredential):
        try_get_token(credential)
        if credential._client is None:
            raise ValueError(f'Failed to resolve {type(credential).__name__}.')
        return {"type": "AzureMLCredential",
                "client_id": credential._client._identity_config.get('clientid')}

    else:
        raise ValueError(f'credential type {type(credential).__name__} is not supported.')

def start_fuse_mount_subprocess(
        source_uri: str,
        mount_point: str,
        read_only: bool,
        allow_other: bool = None,
        credential: TokenCredential = DefaultAzureCredential(),
        debug: bool = False):
    
    assert not os.path.exists(mount_point) or (os.path.isdir(mount_point) and not os.listdir(mount_point)), \
        f'mount point `{mount_point}` already exists but is not an empty directory. please specify a different mount point.'
    try:
        os.makedirs(mount_point)
    except FileExistsError:
        pass

    assert not (os.name == 'nt' and os.path.exists('/dev')), \
        'it seems that you are inside WSL (Windows Subsystem for Linux) but you are invoking Azure CLI for Windows. this particular use case is not supported. ' + \
        'please install Azure CLI **inside** WSL and try again. ' + \
        '(you can verify by running `$ which az` in WSL: it should return a native Linux path (for example `/usr/bin/az`) instead of a translated Windows path (for example `/mnt/c/Program Files (x86)/Microsoft SDKs/Azure/CLI2/wbin/az`).).'

    assert not os.name == 'nt', \
        'mount is not supported on Windows. ' + \
        'please use Linux or WSL (Windows Subsystem for Linux). '

    assert os.path.exists('/dev/fuse'), \
        'file `/dev/fuse` does not exist. ' + \
        'mount is only supported on Linux with FUSE enabled. ' + \
        'try `$ sudo apt install fuse`. if you are inside a docker container, run the container with `--privileged` argument.'

    try:
        subprocess.check_output(['which', 'fusermount'])
    except subprocess.CalledProcessError:
        raise AssertionError('`fusermount` is not found. (command `$ which fusermount` failed.) try `$ sudo apt install fuse`.')

    log_directory = os.path.join(tempfile.gettempdir(), 'azureml-logs', 'dataprep', 'rslex-fuse-cli')
    print("Resolving credential...")
    serialized_credential = json.dumps({k: v for k, v in serialize_azure_credential(credential).items() if v is not None})
    print("Resolved credential.")
    try:
        os.makedirs(log_directory)
    except FileExistsError:
        pass

    if allow_other is None:
        allow_other = os.environ.get('AZUREML_DATAPREP_RSLEX_FUSE_MOUNT_NO_ALLOW_OTHER') is None

    print("Mount starting...")
    start_time = time.time()
    process = subprocess.Popen([sys.executable, "-c", f"""
import json
import sys
sys.path={sys.path}
from azureml.dataprep.rslex import PyMountOptions, RslexURIMountContext, init_environment

uri = sys.argv[1]
mount_point = sys.argv[2]
read_only = sys.argv[3] == str(True)
allow_other = sys.argv[4] == str(True)
debug = sys.argv[5] == str(True)
log_directory = sys.argv[6]
credential = json.loads(input())

print('Mount initializing...')
init_environment(log_directory, None, 'DEBUG' if debug else 'INFO', False, None, None, None, None, None, credential)
options = PyMountOptions(None, None, None, allow_other, read_only, 0o777, False)
mount_context = RslexURIMountContext(mount_point, uri, options, True)

print('Mount starting... ')
mount_context.start(True) # blocking

print('Mount ended.')
""",
    source_uri, mount_point, str(read_only), str(allow_other), str(debug), log_directory],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE, stderr=open(os.path.join(log_directory, 'wrapper-script-stderr.log'), 'a'))

    print(f"Subprocess PID: {process.pid}")
    process.stdin.write(serialized_credential.encode('utf-8'))
    process.stdin.close()

    while True:
        if time.time() - start_time > 15:
            process.kill()
            raise AssertionError(f'rslex-fuse-cli subprocess timed out. Logs can be found at {log_directory}')
        if process.poll() is not None:
            raise AssertionError(f'rslex-fuse-cli subprocess exited unexpectedly. Logs can be found at {log_directory}')
        output_line = process.stdout.readline()
        if output_line and 'Mount started.' in output_line.decode('utf-8'):
            print(f"Mount started successfully.")
            print(f"To unmount, run `$ umount {mount_point}`.")
            print(f"Logs can be found at {log_directory}")
            return

if __name__ == '__main__':
    # print(serialize_azure_credential(DefaultAzureCredential()))
    # print(serialize_azure_credential(AzureMLCredential(client_id='client')))
    # print(serialize_azure_credential(ClientSecretCredential('f4d3b0c5-8e5b-4aef-9c3d-1b930f41e5f5', 'client_id', 'client_secret')))
    os.environ['RUST_BACKTRACE'] = 'full'
    os.environ['MSI_ENDPOINT'] = 'http://host.docker.internal:8080/msi_endpoint'
    os.environ['MSI_SECRET'] = 'secret'
    start_fuse_mount_subprocess(
        "azureml://subscriptions/1b75927d-563b-49d2-bf8a-772f7d6a170e/resourcegroups/commonruntime-test/workspaces/cr-win-ws-wcus/datastores/workspaceblobstore/paths/test-dir", 
        "/mnt/test", 
        True, 
        credential=ManagedIdentityCredential(client_id="dummy_client_id"), 
        debug=True
    )
    time.sleep(86400)
    # os.environ['IDENTITY_ENDPOINT'] = 'https://endpoint'
    # os.environ['IDENTITY_HEADER'] = 'header'
    # # print(serialize_azure_credential(ManagedIdentityCredential()))
    # os.environ['IDENTITY_SERVER_THUMBPRINT'] = 'thumbprint'
    # print(serialize_azure_credential(ManagedIdentityCredential()))