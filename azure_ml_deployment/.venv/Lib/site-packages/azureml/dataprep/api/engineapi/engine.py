# Copyright (c) Microsoft Corporation. All rights reserved.
"""Launch and exchange messages with the engine."""
from .._loggerfactory import  _LoggerFactory, _log_not_supported_api_usage_and_raise


log = _LoggerFactory.get_logger('dprep.engine')

use_multithread_channel = True


def launch_engine() -> 'AbstractMessageChannel':
    """Launch the engine process and set up a MessageChannel."""
    _log_not_supported_api_usage_and_raise(log, 'launch_engine', 'Clex engine deprecated')


def use_single_thread_channel():
    _LoggerFactory.trace_warn(log,
        f'[NOT_SUPPORTED_API_USE_ATTEMPT] The [use_single_thread_channel] API has been deprecated and is no longer supported',
        custom_dimensions={'api_name': 'use_single_thread_channel'})


def use_multi_thread_channel():
    _LoggerFactory.trace_warn(log,
        f'[NOT_SUPPORTED_API_USE_ATTEMPT] The [use_multi_thread_channel] API has been deprecated and is no longer supported',
        custom_dimensions={'api_name': 'use_multi_thread_channel'})


def _get_engine_path():
    _log_not_supported_api_usage_and_raise(log, '_get_engine_path', 'Clex engine deprecated')


def _get_engine_dll_path(dll_name):
    _log_not_supported_api_usage_and_raise(log, '_get_engine_dll_path', 'Clex engine deprecated')


def _set_sslcert_path(env):
    _log_not_supported_api_usage_and_raise(log, '_set_sslcert_path', 'Clex engine deprecated')


def _enable_globalization_invariant_if_needed(dotnet_path, env):
    _log_not_supported_api_usage_and_raise(log, '_enable_globalization_invariant_if_needed', 'Clex engine deprecated')


def _format_with_session_id(message):
    _log_not_supported_api_usage_and_raise(log, '_format_with_session_id', 'Clex engine deprecated')
