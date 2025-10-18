# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains helper methods for mltable apis."""
import re
import yaml
import os
from enum import Enum

from azureml.dataprep.api._loggerfactory import _LoggerFactory
from azureml.dataprep.api.errorhandlers import DataPrepException


_DATA_ASSET_URI_PATTERN = None
_REGULAR_CLOUD_PATTERN_URI = None
_DATASET_URI_PATTERN = None
_DATA_ASSET_SHORT_URI = None
logger = _LoggerFactory.get_logger("MLTableHelper")


class UserErrorException(DataPrepException):
    """Exception to separate user errors from system errors"""

    def __init__(self, err):
        if isinstance(err, DataPrepException):
            self.message = err.message
            self.error_code = err.error_code
            self.compliant_message = err.compliant_message
            self.inner_dprep_error = err
        else:
            self.inner_dprep_error = None
            err_message = err.message if hasattr(err, 'message') else str(err)
            super().__init__(err_message, 'UserError', err_message)

    def __repr__(self):
        if self.inner_dprep_error:
            return self.inner_dprep_error.__repr__()
        return super().__repr__()

    def __str__(self):
        return self.__repr__()


def _download_mltable_yaml(path: str):
    from azureml.dataprep.rslex import Copier, PyLocationInfo, PyIfDestinationExists
    from azureml.dataprep.api._rslex_executor import ensure_rslex_environment

    ensure_rslex_environment()

    # normalize path to MLTable yaml path
    normalized_path = "{}/MLTable".format(path.rstrip("/"))
    if_destination_exists = PyIfDestinationExists.MERGE_WITH_OVERWRITE
    try:

        from tempfile import mkdtemp
        local_path = mkdtemp()
        Copier.copy_uri(PyLocationInfo('Local', local_path, {}),
                        normalized_path, if_destination_exists, "")

        return local_path
    except Exception as e:
        error_message = str(e)
        if "InvalidUriScheme" in error_message or \
                "DataAccessError(NotFound)" in error_message:
            raise UserErrorException(e)
        elif "StreamError(NotFound)" in error_message:
            raise UserErrorException(error_message + "; Not able to find MLTable file")
        elif "StreamError(PermissionDenied" in error_message:
            raise UserErrorException(f'Getting permission error when trying to access MLTable,'
                                     f'please make sure proper access is configured on storage: {error_message}')
        elif "No identity was found on compute" in error_message:
            raise UserErrorException(e)
        else:
            raise SystemError(e)


def _is_tabular(mltable_yaml):
    if mltable_yaml is None:
        return False

    transformations_key = "transformations"
    if transformations_key not in mltable_yaml.keys():
        return False

    tabular_transformations = [
        "read_files",
        "read_delimited",
        "read_delta_lake",
        "read_parquet",
        "read_json_lines"
    ]
    if 'query_source' in mltable_yaml:
        return True

    if mltable_yaml['transformations'] and all(isinstance(e, dict) for e in mltable_yaml['transformations']):
        list_of_transformations = [k for d in mltable_yaml['transformations'] for k, v in d.items()]
    else:
        # case where transformations section is a list[str], not a list[dict].
        list_of_transformations = mltable_yaml['transformations']
    return any(key in tabular_transformations for key in list_of_transformations)


def _read_yaml(uri):
    local_yaml_path = os.path.join(uri, 'MLTable')
    if not os.path.exists(local_yaml_path):
        raise UserErrorException('Not able to find MLTable file from the MLTable folder')

    with open(local_yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            raise UserErrorException(f'MLTable yaml is invalid: {e}')


def _path_has_parent_directory_redirection(path):
    return path.startswith('..') or '/..' in path or '\\..' in path


class _PathType(Enum):
    local = 1
    cloud = 2
    legacy_dataset = 3
    data_asset_uri = 4
    data_asset_short_uri = 5


def _parse_path_format(path: str):
    global _DATA_ASSET_URI_PATTERN, _REGULAR_CLOUD_PATTERN_URI, _DATASET_URI_PATTERN, _DATA_ASSET_SHORT_URI

    if _DATA_ASSET_URI_PATTERN is None:
        _DATA_ASSET_URI_PATTERN = re.compile(r'^azureml://subscriptions/([^\/]+)/resourcegroups/([^\/]+)/'
                                             r'(?:providers/Microsoft.MachineLearningServices/)?workspaces/'
                                             r'([^\/]+)/data/([^\/]+)/versions/(.*)',
                                             re.IGNORECASE)
        _REGULAR_CLOUD_PATTERN_URI = re.compile(r'^https?://|adl://|wasbs?://|abfss?://|azureml://subscriptions',
                                                re.IGNORECASE)

        _DATASET_URI_PATTERN = re.compile(r'^azureml://locations/([^\/]+)/workspaces/([^\/]+)/data/([^\/]+)/versions/(.*)',
                                          re.IGNORECASE)
        _DATA_ASSET_SHORT_URI = re.compile(r'^azureml:([^:/\\]+)(:([^:/\\]+))?$', re.IGNORECASE)

    data_asset_uri_match = _DATA_ASSET_URI_PATTERN.match(path)
    if data_asset_uri_match:
        return _PathType.data_asset_uri, path, (data_asset_uri_match.group(1),
                                                data_asset_uri_match.group(2),
                                                data_asset_uri_match.group(3),
                                                data_asset_uri_match.group(4),
                                                data_asset_uri_match.group(5))

    if _REGULAR_CLOUD_PATTERN_URI.match(path):
        return _PathType.cloud, path, None

    dataset_uri_match = _DATASET_URI_PATTERN.match(path)
    if dataset_uri_match:
        return _PathType.legacy_dataset, path, (dataset_uri_match.group(3), dataset_uri_match.group(4))

    # pattern that allows 'azureml:data_asset_a:1' (with version) and 'azureml:dataset_asset_b' (without version)
    dataset_uri_match = _DATA_ASSET_SHORT_URI.match(path)
    if dataset_uri_match:
        # group 1 will be the dataset/data asset name, group 3 will be the version if it exists
        return _PathType.data_asset_short_uri, path, (dataset_uri_match.group(1),
                                                      dataset_uri_match.group(3) if dataset_uri_match.group(3) else None)

    return _PathType.local, path, None
