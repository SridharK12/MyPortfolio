# Copyright (c) Microsoft Corporation. All rights reserved.
# pylint: skip-file
# This file is auto-generated. Do not modify.
from .._loggerfactory import _LoggerFactory, _log_not_supported_api_usage_and_raise
from . import typedefinitions
from typing import Dict, List


logger = None

def get_logger():
    global logger
    if logger is not None:
        return logger

    logger = _LoggerFactory.get_logger("EngineAPI")
    return logger


def get_engine_api():
    _log_not_supported_api_usage_and_raise(get_logger(), 'get_engine_api', 'Use EnginelessDataflow instead.')


def kill_engine_api():
    pass

class EngineAPI:
    def __init__(self):
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.__init__', 'Clex engine is deprecated do not init')

    def add_block_to_list(self, message_args: typedefinitions.AddBlockToListMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.AnonymousBlockData:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.add_block_to_list', 'Clex engine deprecated')

    def add_temporary_secrets(self, message_args: Dict[str, str], cancellation_token: 'CancellationToken' = None) -> None:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.add_temporary_secretes', 'Clex engine deprecated')

    def anonymous_data_source_prose_suggestions(self, message_args: typedefinitions.AnonymousDataSourceProseSuggestionsMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.DataSourceProperties:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.anonymous_data_source_prose_suggestions', 'Clex engine deprecated')

    def anonymous_send_message_to_block(self, message_args: typedefinitions.AnonymousSendMessageToBlockMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.AnonymousSendMessageToBlockMessageResponseData:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.anonymous_send_message_to_block', 'Clex engine deprecated')

    def close_stream_info(self, message_args: str, cancellation_token: 'CancellationToken' = None) -> None:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.close_stream_info', 'Clex engine deprecated')

    def convert_rsflow_to_dataflow_json(self, message_args: typedefinitions.ConvertRsflowToDataflowJsonMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.AnonymousActivityData:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.convert_rsflow_to_dataflow_json', 'Clex engine deprecated')

    def create_anonymous_reference(self, message_args: typedefinitions.CreateAnonymousReferenceMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.ActivityReference:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.create_anonymous_reference', 'Clex engine deprecated')

    def create_folder(self, message_args: typedefinitions.CreateFolderMessageArguments, cancellation_token: 'CancellationToken' = None) -> int:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.create_folder', 'Clex engine deprecated')

    def delete(self, message_args: typedefinitions.DeleteMessageArguments, cancellation_token: 'CancellationToken' = None) -> int:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.delete', 'Clex engine deprecated')

    def download_stream_info(self, message_args: typedefinitions.DownloadStreamInfoMessageArguments, cancellation_token: 'CancellationToken' = None) -> int:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.download_stream_info', 'Clex engine deprecated')

    def execute_anonymous_activity(self, message_args: typedefinitions.ExecuteAnonymousActivityMessageArguments, cancellation_token: 'CancellationToken' = None) -> None:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.execute_anonymous_activity', 'Clex engine deprecated')

    def execute_data_diff(self, message_args: typedefinitions.ExecuteDataDiffMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.ExecuteDataDiffMessageResponse:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.execute_data_diff', 'Clex engine deprecated')

    def execute_inspector(self, message_args: typedefinitions.ExecuteInspectorCommonArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.ExecuteInspectorCommonResponse:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.execute_inspector', 'Clex engine deprecated')

    def execute_inspectors(self, message_args: List[typedefinitions.ExecuteInspectorsMessageArguments], cancellation_token: 'CancellationToken' = None) -> Dict[str, typedefinitions.ExecuteInspectorCommonResponse]:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.execute_inspectors', 'Clex engine deprecated')

    def export_script(self, message_args: typedefinitions.ExportScriptMessageArguments, cancellation_token: 'CancellationToken' = None) -> List[typedefinitions.SecretData]:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.export_script', 'Clex engine deprecated')

    def file_exists(self, message_args: typedefinitions.FileExistsMessageArguments, cancellation_token: 'CancellationToken' = None) -> bool:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.file_exists', 'Clex engine deprecated')

    def get_activity(self, message_args: str, cancellation_token: 'CancellationToken' = None) -> typedefinitions.AnonymousActivityData:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_activity', 'Clex engine deprecated')

    def get_block_descriptions(self, message_args: None, cancellation_token: 'CancellationToken' = None) -> List[typedefinitions.IBlockDescription]:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_block_descriptions', 'Clex engine deprecated')

    def get_data(self, message_args: typedefinitions.GetDataMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.GetDataMessageResponse:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_data', 'Clex engine deprecated')

    def get_inspector_descriptions(self, message_args: None, cancellation_token: 'CancellationToken' = None) -> List[typedefinitions.InspectorDescription]:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_inspector_descriptions', 'Clex engine deprecated')

    def get_inspector_lariat(self, message_args: typedefinitions.ExecuteInspectorCommonArguments, cancellation_token: 'CancellationToken' = None) -> str:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_inspector_lariat', 'Clex engine deprecated')

    def get_partition_count(self, message_args: List[typedefinitions.AnonymousBlockData], cancellation_token: 'CancellationToken' = None) -> int:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_partition_count', 'Clex engine deprecated')

    def get_program_step_descriptions(self, message_args: None, cancellation_token: 'CancellationToken' = None) -> List[typedefinitions.ProgramStepDescription]:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_program_step_descriptions', 'Clex engine deprecated')

    def get_script(self, message_args: typedefinitions.GetScriptMessageArguments, cancellation_token: 'CancellationToken' = None) -> str:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_script', 'Clex engine deprecated')

    def get_secrets(self, message_args: typedefinitions.GetSecretsMessageArguments, cancellation_token: 'CancellationToken' = None) -> List[typedefinitions.SecretData]:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_secrets', 'Clex engine deprecated')

    def get_source_data_hash(self, message_args: typedefinitions.GetSourceDataHashMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.GetSourceDataHashMessageResponse:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.get_source_data_hash', 'Clex engine deprecated')

    def infer_types(self, message_args: List[typedefinitions.AnonymousBlockData], cancellation_token: 'CancellationToken' = None) -> Dict[str, typedefinitions.FieldInference]:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.infer_types', 'Clex engine deprecated')

    def infer_types_with_span_context(self, message_args: typedefinitions.InferTypesWithSpanContextMessageArguments, cancellation_token: 'CancellationToken' = None) -> Dict[str, typedefinitions.FieldInference]:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.infer_types_with_span_context', 'Clex engine deprecated')

    def load_activity_from_json(self, message_args: str, cancellation_token: 'CancellationToken' = None) -> typedefinitions.AnonymousActivityData:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.load_activity_from_json', 'Clex engine deprecated')

    def load_activity_from_package(self, message_args: str, cancellation_token: 'CancellationToken' = None) -> typedefinitions.AnonymousActivityData:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.load_activity_from_package', 'Clex engine deprecated')

    def move_file(self, message_args: typedefinitions.MoveFileMessageArguments, cancellation_token: 'CancellationToken' = None) -> int:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.move_file', 'Clex engine deprecated')

    def open_stream_info(self, message_args: typedefinitions.Value, cancellation_token: 'CancellationToken' = None) -> str:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.open_stream_info', 'Clex engine deprecated')

    def read_stream_info(self, message_args: typedefinitions.ReadStreamInfoMessageArguments, cancellation_token: 'CancellationToken' = None) -> int:
       _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.read_stream_info', 'Clex engine deprecated')

    def register_secret(self, message_args: typedefinitions.RegisterSecretMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.Secret:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.register_secret', 'Clex engine deprecated')

    def save_activity_from_data(self, message_args: typedefinitions.SaveActivityFromDataMessageArguments, cancellation_token: 'CancellationToken' = None) -> None:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.save_activity_from_data', 'Clex engine deprecated')

    def save_activity_to_json(self, message_args: typedefinitions.AnonymousActivityData, cancellation_token: 'CancellationToken' = None) -> str:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.save_activity_to_json', 'Clex engine deprecated')

    def save_activity_to_package(self, message_args: typedefinitions.AnonymousActivityData, cancellation_token: 'CancellationToken' = None) -> str:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.save_activity_to_package', 'Clex engine deprecated')

    def set_aml_auth(self, message_args: typedefinitions.SetAmlAuthMessageArgument, cancellation_token: 'CancellationToken' = None) -> None:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.set_aml_auth', 'Clex engine deprecated')

    def set_executor(self, message_args: typedefinitions.ExecutorType, cancellation_token: 'CancellationToken' = None) -> None:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.set_executor', 'Clex engine deprecated')

    def sync_host_channel_port(self, message_args: int, cancellation_token: 'CancellationToken' = None) -> int:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.sync_host_channel_port', 'Clex engine deprecated')

    def sync_host_secret(self, message_args: str, cancellation_token: 'CancellationToken' = None) -> str:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.sync_host_secret', 'Clex engine deprecated')

    def update_environment_variable(self, message_args: Dict[str, str], cancellation_token: 'CancellationToken' = None) -> None:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.update_environment_variable', 'Clex engine deprecated')

    def upload_directory(self, message_args: typedefinitions.UploadDirectoryMessageArguments, cancellation_token: 'CancellationToken' = None) -> None:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.upload_directory', 'Clex engine deprecated')

    def upload_file(self, message_args: typedefinitions.UploadFileMessageArguments, cancellation_token: 'CancellationToken' = None) -> None:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.upload_file', 'Clex engine deprecated')

    def validate_activity_source(self, message_args: typedefinitions.ValidateActivitySourceMessageArguments, cancellation_token: 'CancellationToken' = None) -> typedefinitions.ValidationResult:
        _log_not_supported_api_usage_and_raise(get_logger(), 'EngineAPI.validate_activity_source', 'Clex engine deprecated')
