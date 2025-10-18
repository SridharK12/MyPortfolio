import json
import threading
from ._loggerfactory import _LoggerFactory, session_id, log_directory, verbosity, HBI_MODE
from typing import Optional, Dict
from azureml.dataprep.rslex import PyRsDataflow

log = _LoggerFactory.get_logger('rslex_executor')


class _RsLexExecutor:
    def _ensure_dataflow(self, script):
        if isinstance(script, str):
            return PyRsDataflow(script)
        if hasattr(script, '_py_rs_dataflow'):
            return script._py_rs_dataflow
        return script

    def execute_dataflow(self, script, collect_results=False, fail_on_error=False, fail_on_mixed_types=False,
                         fail_on_out_of_range_datetime=False, traceparent='', partition_ids=None):
        '''
        Takes a script to execute and execution properties. Attempts to execute the script in rslex and return record batches.
        '''
        script = self._ensure_dataflow(script)

        from azureml.dataprep.rslex import Executor
        (batches, num_partitions, stream_columns) = Executor().execute_dataflow(script,
                                                                                collect_results,
                                                                                partition_ids,
                                                                                fail_on_error,
                                                                                fail_on_mixed_types,
                                                                                fail_on_out_of_range_datetime,
                                                                                traceparent)

        log.info(f'Execution succeeded with {num_partitions} partitions.')
        return (batches, num_partitions, stream_columns)

    def to_pyrecords(self, script, traceparent=''):
        script = self._ensure_dataflow(script)

        from azureml.dataprep.rslex import Executor
        return Executor().execute_dataflow_to_pyrecords(script, traceparent)

    def get_partition_info(self, script, traceparent=''):
        script = self._ensure_dataflow(script)

        from azureml.dataprep.rslex import Executor
        (num_partitions, streams_with_partition_counts) = Executor().get_partition_info(script, traceparent)

        streams_present = 'not' if streams_with_partition_counts is None else ''
        log.info(f'Getting partition count succeeded with {num_partitions} partitions. Streams were {streams_present} present.')
        return (num_partitions, streams_with_partition_counts)

    def get_partition_count(self, script, traceparent=''):
        '''
        Takes a script to execute and execution properties. Attempts to execute the script without collecting in rslex and return partition count.
        '''
        return self.execute_dataflow(
            script,
            collect_results=False,
            fail_on_error=False,
            fail_on_mixed_types=False,
            fail_on_out_of_range_datetime=False,
            traceparent=traceparent
        )[1]

    def infer_types(self, script, sample_size = 200, traceparent=''):
        script = self._ensure_dataflow(script)

        from azureml.dataprep.rslex import Executor
        return Executor().type_inference_from_dataflow(script, sample_size, traceparent)

_rslex_executor = None
_rslex_environment_init = False
_rslex_environment_lock = threading.Lock()


def get_rslex_executor():
    global _rslex_executor
    if _rslex_executor is None:
        _rslex_executor = _RsLexExecutor()
    ensure_rslex_environment()

    return _rslex_executor


def _set_rslex_environment(value: bool):
    global _rslex_environment_init
    _rslex_environment_lock.acquire()
    _rslex_environment_init = value
    _rslex_environment_lock.release()


def ensure_rslex_environment(caller_session_id: str = None, disk_space_overrides: Optional[Dict[str, int]] = None, metrics_endpoint: Optional[str] = None):
    global _rslex_environment_init
    if _rslex_environment_init is False:
        try:
            # Acquire lock on mutable access to _rslex_environment_init
            _rslex_environment_lock.acquire()
            # Was _rslex_environment_init set while we held the lock?
            if _rslex_environment_init is True:
                return _rslex_environment_init
            # Initialize new RsLex Environment
            import atexit
            import azureml.dataprep.rslex as rslex
            run_info = _LoggerFactory._try_get_run_info()
            rslex.init_environment(
                log_directory,
                None,  # by default rslex uses its own app insights key
                verbosity,
                HBI_MODE,
                session_id,
                caller_session_id,
                json.dumps(run_info) if run_info is not None else None,
                disk_space_overrides = disk_space_overrides,
                metrics_endpoint = metrics_endpoint
            )
            _rslex_environment_init = True
            _LoggerFactory.add_default_custom_dimensions({'rslex_version': rslex.__version__})
            atexit.register(rslex.release_environment)
        except Exception as e:
            log.error('ensure_rslex_environment failed with {}'.format(e))
            raise
        finally:
            if _rslex_environment_lock.locked():
                _rslex_environment_lock.release()
    return _rslex_environment_init


def use_rust_execution(use: bool):
    # RSlex is the only option now
    if not use:
        print('Clex Engine has been deprecated, RSlex is now the only available option')
        log.warning('use_rslex_execution called with use=False')
    ensure_rslex_environment()
