# Copyright (c) Microsoft Corporation. All rights reserved.
from .engineapi.typedefinitions import Secret as _Secret
from typing import Dict, List
from ._loggerfactory import _LoggerFactory, _log_not_supported_api_usage_and_raise


logger = None

def get_logger():
    global logger
    if logger is not None:
        return logger

    logger = _LoggerFactory.get_logger("secretmanager")
    return logger

class Secret:
    def __init__(self, id: str = None):
        self._secret = _Secret(id)

    @property
    def id(self) -> str:
        """
        Id of the secret.
        """
        return self._secret.id

    @id.setter
    def id(self, value: str):
        self._secret = value

    @classmethod
    def from_pod(cls, pod):
        obj = cls()
        obj._secret._pod = pod
        return obj

    def to_pod(self):
        return self._secret._pod

    def __repr__(self):
        return self._secret.__repr__()


def register_secrets(secrets: Dict[str, str]):
    """
    (DEPRECATED)
    Registers a set of secrets to be used during execution.

    :param secrets: Dictionary of secret id to secret value.
    """
    return [register_secret(value, sid) for sid, value in secrets.items()]


def register_secret(value: str, id: str = None):
    """
    (DEPRECATED)
    Registers a secret to be used during execution.

    :param value: Value to keep secret. This won't be persisted with the package.
    :param id: (Optional) Secret id to use. This will be persisted in the package. Default value is new Guid.
    """
    _log_not_supported_api_usage_and_raise(get_logger(), 'register_secret', 'Downgrade to previous version of azureml-dataprep.')

def create_secret(id: str) -> Secret:
    """
    (DEPRECATED)
    Creates a Secret. Secrets are used in remote data sources like :class:`azureml.dataprep.MSSQLDataSource`.

    .. remarks..

        In order for execution to succeed, you will need to call `register_secret()` first, providing the same `id` and secret `value` to use during execution.

    :param id: Secret id to use. This will be persisted in package.
    """
    _log_not_supported_api_usage_and_raise(get_logger(), 'create_secret', 'Downgrade to previous version of azureml-dataprep.')
