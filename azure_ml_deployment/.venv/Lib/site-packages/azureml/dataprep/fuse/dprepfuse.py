import os
import uuid
from typing import Optional

from azureml.dataprep import Dataflow
from azureml.dataprep.api._loggerfactory import _LoggerFactory, trace, _log_not_supported_api_usage_and_raise
from azureml.dataprep.api._rslex_executor import ensure_rslex_environment
from azureml.dataprep.api.tracing._context import Context

log = _LoggerFactory.get_logger('dprep.fuse')
tracer = trace.get_tracer(__name__)


class MountOptions:
    def __init__(self,
                 data_dir: str = None,
                 max_size: int = None,
                 free_space_required: int = None,
                 default_permission=0o777,
                 allow_other=False,
                 **kwargs):
        """
        Configuration options for file mounting.

        .. remarks::

            Depending on the source of the streams mounted, it might be necessary to fully download a file locally
                before it can be opened by the file system. For sources that support streaming, access to the file
                can be provided without this requirement. In both cases, it is possible to configure the system
                to cache the data locally. This can be useful when specific files will be accessed multiple times
                and the source of the data is remote to the current compute. These downloaded and cached files will
                be stored in the system's tmp folder by default, but this can be overridden by manually specifying a
                data_dir.

            The max_size and free_space_required parameters can be used to limit how much data will be downloaded
                or cached locally. If accessing a file requires that it be downloaded, then the least recently used
                files will be deleted after the download completes in order to stay within these parameters. If a file
                that needs to be downloaded before it can be opened by the file system does not fit within the available
                space, an error will be returned.

        :param data_dir: The directory to use to download or cache files locally. If None is provided, the system's
            temp folder is used.
        :param max_size: The maximum amount of memory, in bytes, that can be stored in data_dir.
        :param free_space_required: How much space should be kept available in the data_dir volume.
        :param default_permission: The default permissions for all files.
        :param allow_other: By default fuse mounts are accessable by only the user which creates the mount.
            allow_other=True extends access permission to any user (including root).
        """
        self.data_dir = data_dir
        self.max_size = max_size
        self.free_space_required = free_space_required
        self.default_permission = default_permission
        self.allow_other = allow_other
        self._data_dir_suffix = kwargs.get(
            'data_dir_suffix', str(uuid.uuid4()))
        self._cached_dataflow_override = kwargs.get('cached_dataflow_override')
        self._disable_dataflow_cache = kwargs.get(
            'disableMetadataCache', "False").lower() == "true"
        self.read_only = kwargs.get('read_only', True)
        self.create_destination = kwargs.get('create_destination', False)

    @property
    def final_data_dir(self):
        return os.path.join(self.data_dir, self._data_dir_suffix) if self._data_dir_suffix else self.data_dir


def mount(dataflow: Optional[Dataflow],
          files_column: Optional[str],
          mount_point: str,
          base_path: str = None,
          options: MountOptions = None,
          destination: tuple = None,
          foreground=True,
          invocation_id: str = None,
          span_context: Optional[Context] = None,
          client_id=None,
          identity_endpoint_type=None,
          enable_rslex_mount: Optional[bool] = None,
          **kwargs):
    if dataflow is not None:
        try:
            os.makedirs(mount_point, exist_ok=True)
        except Exception as e:
            message = 'Failed to ensure mount point "{}" due to exception of type {} with message {}, proceeding with mount attempt.'.format(
            mount_point, type(e).__name__, e)
            _LoggerFactory.trace_warn(log, message)
            print(message)
        return dataflow.mount(mount_point, options, files_column)
    elif destination is not None:
        return rslex_direct_volume_mount(
            None, mount_point, options, destination, client_id=client_id,
            identity_endpoint_type=identity_endpoint_type, enable_rslex_mount=True,
            invocation_id=invocation_id, span_context=span_context)

    raise RuntimeError('Either Dataflow or Destination is required to perform mount.')


def clex_mount(dataflow: Optional[Dataflow],
               files_column: Optional[str],
               mount_point: str,
               base_path: str = None,
               options: MountOptions = None,
               destination: tuple = None,
               foreground=True,
               invocation_id: str = None,
               span_context: Optional[Context] = None,
               client_id=None,
               identity_endpoint_type=None,
               **kwargs) -> Optional['MountContext']:
    _log_not_supported_api_usage_and_raise(log, 'clex_mount', 'Clex is deprecated, use rslex_mount')

def rslex_uri_volume_mount(uri: str, mount_point: str, options: Optional[MountOptions] = None):
    try:
        _LoggerFactory.trace(log, 'Running rslex URI volume mount')
        from azureml.dataprep.rslex import PyMountOptions, RslexURIMountContext

        ensure_rslex_environment(None)

        max_size = None
        free_space_required = None
        allow_other = None
        cache_dir_path = None
        read_only = True
        create_destination = False
        permissions = 0o777

        if options:
            max_size = options.max_size
            free_space_required = options.free_space_required
            allow_other = options.allow_other
            cache_dir_path = options.data_dir
            permissions = options.default_permission
            read_only = options.read_only
            create_destination = options.create_destination

        options = PyMountOptions(
            max_size,
            cache_dir_path,
            free_space_required,
            allow_other,
            read_only,
            permissions,
            create_destination
        )

        try:
            os.makedirs(mount_point, exist_ok=True)
        except Exception as e:
            _LoggerFactory.trace_warn(log, 'Failed to ensure mount point "{}" due to exception of type {} with message {}'.format(
                mount_point, type(e).__name__, e))

        mount_context = RslexURIMountContext(mount_point, uri, options, True)
        _LoggerFactory.trace(log, "Rslex URI volume mount context created")
        return mount_context
    except Exception as e:
        _LoggerFactory.trace_warn(log, 'Failed to mount URI due to exception of type {} with message {}'.format(
            type(e).__name__, e))
        raise e


class VolumeMountNotEnabled(Exception):
    pass


class VolumeMountNotSupported(Exception):
    def __init__(self, message):
        super().__init__(message)


class VolumeMountFailed(Exception):
    def __init__(self, message):
        super().__init__(message)


def rslex_direct_volume_mount(dataflow: Optional[Dataflow],
                mount_point: str,
                options: MountOptions = None,
                destination: tuple = None,
                client_id: Optional[str] = None,
                identity_endpoint_type: Optional[str] = None,
                enable_rslex_mount: Optional[bool] = None,
                invocation_id: str = None,
                span_context: Optional[Context] = None):
    if dataflow is not None:
        raise RuntimeError('rslex_direct_volume_mount should only be used for destination mounting, to mount dataflow call dataflow.mount() directly.')
    
    def log_if_disabled(override: Optional[str]) -> bool:
        if override is not None:
            if override.lower() == 'false' or override == '0':
                _LoggerFactory.trace_warn(log, 'Direct volume mount was disabled by env variable, this is no longer supported. Ignoring.')

    rslex_writable_fuse_override = os.getenv('RSLEX_DIRECT_VOLUME_WRITABLE_MOUNT')
    log_if_disabled(rslex_writable_fuse_override)

    # == "1" is for backward compatibility for the people we already gave this flight
    # == "true" to be consistant with other arguments
    _LoggerFactory.trace_warn(log, 'Running rslex direct volume mount')
    from azureml.dataprep.api._rslex_executor import ensure_rslex_environment
    from azureml.dataprep.rslex import (PyDatastoreSource,
                                        PyMountOptions,
                                        RslexDirectMountContext)

    ensure_rslex_environment(None)
    datastore_source = None
    mount_options = None

    max_size = None
    free_space_required = None
    allow_other = None
    cache_dir_path = None
    datastore = None
    datastore_path = None
    read_only = True
    subscription_id = None
    resource_group = None
    workspace_name = None
    datastore_name = None
    permissions = None

    if options:
        max_size = options.max_size
        free_space_required = options.free_space_required
        allow_other = options.allow_other
        cache_dir_path = options.data_dir
        permissions = options.default_permission


    if destination is not None:
        datastore = destination[0]
        datastore_path = destination[1]
        read_only = False
        subscription_id = datastore.workspace.subscription_id
        resource_group = datastore.workspace.resource_group
        workspace_name = datastore.workspace.name
        datastore_name = datastore.name

    if datastore is not None:
        mount_options = PyMountOptions(
            max_size,
            cache_dir_path,
            free_space_required,
            allow_other,
            read_only,
            permissions
        )
        datastore_source = PyDatastoreSource(
            subscription_id,
            resource_group,
            workspace_name,
            datastore_name,
            datastore_path,
            client_id,
            identity_endpoint_type
        )
    else:
        _LoggerFactory.trace_warn(log,
            'enable_rslex_mount is set to True but we couldn\'t extract datastore information')
        raise VolumeMountNotSupported('No datastore info was found, volume mount would not be attempted.')

    try:
        os.makedirs(mount_point, exist_ok=True)
    except Exception as e:
        message = 'Failed to ensure mount point "{}" due to exception of type {} with message {}, proceeding with mount attempt.'.format(
            mount_point, type(e).__name__, e)
        _LoggerFactory.trace_warn(log, message)
        print(message)

    try:
        mount_context = RslexDirectMountContext(
            mount_point, datastore_source, mount_options)
        _LoggerFactory.trace(log, "Rslex direct volume mount context created")
    except BaseException as e:
        if (
            'DataAccess' in message or
            'InvalidOptionValue ' in message or
            'DestinationError' in message or
            'MountOnTopOfDisconnectedMount' in message):
            raise VolumeMountFailed(message)
        raise e

    return mount_context
