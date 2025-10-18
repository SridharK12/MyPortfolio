# Copyright (c) Microsoft Corporation. All rights reserved.
from .engineapi.typedefinitions import FieldType
from ._loggerfactory import trace
from .tracing._open_telemetry_adapter import to_dprep_span_context
from .typeconversions import (CandidateConverter, CandidateDateTimeConverter, InferenceInfo,
                              get_converters_from_candidates)
from .builders import InferenceArguments
from ._rslex_executor import get_rslex_executor
from ._loggerfactory import _LoggerFactory
from typing import List, Dict


logger = None
tracer = trace.get_tracer(__name__)

def get_logger():
    global logger
    if logger is not None:
        return logger

    logger = _LoggerFactory.get_logger("EnginelessDataflow")
    return logger


def _string_to_fieldType(field_type: str) -> FieldType:
    if field_type == 'datetime':
        return FieldType.DATE
    if field_type == 'float':
        return FieldType.DECIMAL
    if field_type == 'boolean':
        return FieldType.BOOLEAN
    if field_type == 'int':
        return FieldType.INTEGER
    if field_type == 'string':
        return FieldType.STRING
    if field_type == 'stream_info':
        return FieldType.STREAM

    raise ValueError('Unexpected field type name: ' + field_type)


class ColumnTypesBuilder:
    """
    Interactive object that can be used to infer column types and type conversion attributes.
    """
    def __init__(self, dataflow: 'EnginelessDataflow'):
        self._dataflow = dataflow
        self._conversion_candidates = None

    def _run_type_inference(self) -> Dict[str, InferenceInfo]:
        def _type_converter_from_inference(inference) -> CandidateConverter:
            if inference.field_type == 'datetime':
                return CandidateDateTimeConverter(inference.datetime_formats, [inference.ambiguous_formats] if inference.ambiguous_formats else [])
            else:
                return CandidateConverter(_string_to_fieldType(inference.field_type))

        def get_inference_info(inference) -> InferenceInfo:
            return InferenceInfo([_type_converter_from_inference(inference)])

        ex = get_rslex_executor()
        error = None
        inference_result = None
        try:
            with tracer.start_as_current_span('EnginelessColumnTypesBuilder._run_type_inference', trace.get_current_span()) as span:
                inference_result = ex.infer_types(self._dataflow._py_rs_dataflow.to_yaml_string(), 200, to_dprep_span_context(span.get_context()).span_id)
                return {col: get_inference_info(inference) for col, inference in inference_result.items()}
        except Exception as e:
            error = e
            raise
        finally:
            builder = {"activity" : 'EnginelessColumnTypesBuilder._run_type_inference'}

            if error is not None:
                builder['rslex_failed'] = True
                builder["rslex_error"] = str(error)
            else:
                builder["execution_succeeded"] = True
                builder["inference_col_count"] = len(inference_result)
            try:
                _LoggerFactory.trace(get_logger(), "dataflow_execution", builder)
            except Exception:
                pass

    @property
    def conversion_candidates(self) -> Dict[str, InferenceInfo]:
        """
        Current dictionary of conversion candidates, where key is column name and value is list of conversion candidates.

        .. remarks::

            The values in the conversion_candidates dictionary could be of several types:

            * :class:`azureml.dataprep.InferenceInfo` (wraps a List of :class:`azureml.dataprep.CandidateConverter`) - populated based on available data by running :meth:`learn`.
            * :class:`azureml.dataprep.FieldType` - user override to force conversion to a specific type.
            * :class:`azureml.dataprep.TypeConverter` - another way to perform a user override to force conversion to a specific type.
            * Tuple of DATE (:class:`azureml.dataprep.FieldType`) and List of format strings (single format string is also supported) - user override for date conversions.

            .. code-block:: python

                import azureml.dataprep as dprep

                dataflow = dprep.read_csv(path='./some/path')
                builder = dataflow.builders.set_column_types()
                builder.conversion_candidates['MyNumericColumn'] = dprep.FieldType.DECIMAL    # force conversion to decimal
                builder.conversion_candidates['MyBoolColumn'] = dprep.FieldType.BOOLEAN       # force conversion to bool
                builder.conversion_candidates['MyDateColumnWithFormat'] = (dprep.FieldType.DATE, ['%m-%d-%Y'])  # force conversion to date with month before day
                builder.conversion_candidates['MyOtherDateColumn'] = dprep.DateTimeConverter(['%d-%m-%Y'])      # force conversion to data with day before month (alternative way)

            .. note::

                This will be populated automatically with inferred conversion candidates when :meth:`learn` is called.
                Any modifications made to this dictionary will be discarded any time :meth:`learn` is called.

        """
        return self._conversion_candidates

    @property
    def ambiguous_date_columns(self) -> List[str]:
        """
        List of columns, where ambiguous date formats were detected.

        .. remarks::

            Each of the ambiguous date columns must be resolved before calling :meth:`to_dataflow`. There are 3 ways to resolve
                ambiguity:

            * Override the value for the column in :func:`azureml.dataprep.api.builders.ColumnTypesBuilder.conversion_candidates` dictionary with a desired date conversion format.
            * Drop conversions for the ambiguous date columns by calling :func:`azureml.dataprep.api.builders.ColumnTypesBuilder.ambiguous_date_conversions_drop`
            * Resolve date conversion ambiguity for all columns by calling :func:`azureml.dataprep.api.builders.ColumnTypesBuilder.ambiguous_date_conversions_keep_day_month`
                or :func:`azureml.dataprep.api.builders.ColumnTypesBuilder.ambiguous_date_conversions_keep_month_day`

        :return: List of columns, where ambiguous date formats were detected.
        """
        if not self._conversion_candidates:
            return []
        result = []
        for col, inference_result in self._conversion_candidates.items():
            if not isinstance(inference_result, InferenceInfo):
                # user has overridden inference info, don't check it here
                continue
            date_converters = \
                (c for c in inference_result.candidate_converters if isinstance(c, CandidateDateTimeConverter))
            for candidate in date_converters:
                if candidate.ambiguous_formats is not None and len(candidate.ambiguous_formats) > 0:
                    result.append(col)
                    break

        return result

    def ambiguous_date_conversions_drop(self) -> None:
        """
        Resolves ambiguous date conversion candidates by removing them from the conversion dictionary.

        .. note::

            Resolving ambiguity this way will ensure that such columns remain unchanged.
        """
        if not self._conversion_candidates:
            return
        columns_to_skip = self.ambiguous_date_columns
        for col in columns_to_skip:
            del self._conversion_candidates[col]

    def _resolve_date_ambiguity(self, prefer_day_first: bool):
        if not self._conversion_candidates:
            return
        for col, inference_result in self._conversion_candidates.items():
            date_converters = \
                (c for c in inference_result.candidate_converters if isinstance(c, CandidateDateTimeConverter))
            for candidate in date_converters:
                candidate.resolve_ambiguity(prefer_day_first)

    def ambiguous_date_conversions_keep_day_month(self) -> None:
        """
        Resolves ambiguous date conversion candidates by only keeping date formats where day comes before month.
        """
        self._resolve_date_ambiguity(True)

    def ambiguous_date_conversions_keep_month_day(self) -> None:
        """
        Resolves ambiguous date conversion candidates by only keeping date formats where month comes before day.
        """
        self._resolve_date_ambiguity(False)

    def learn(self, inference_arguments: InferenceArguments = None) -> None:
        """
        Performs a pull on the data and populates :func:`ColumnTypesBuilder.conversion_candidates` with automatically inferred conversion candidates for each column.

        :param inference_arguments: (Optional) Argument that would force automatic date format ambiguity resolution for all columns.
        """
        with tracer.start_as_current_span('EnginelessColumnTypesBuilder.learn', trace.get_current_span()):
            if inference_arguments is not None and not isinstance(inference_arguments, InferenceArguments):
                raise ValueError('Unexpected inference arguments. Expected instance of InferenceArguments class')
            self._conversion_candidates = self._run_type_inference()
            if inference_arguments is not None:
                self._resolve_date_ambiguity(inference_arguments.day_first)

    def to_dataflow(self) -> 'EnginelessDataflow':
        """
        Uses current state of this object to add 'set_column_types' step to the original Dataflow.

        .. note::

            This call will fail if there are any unresolved date format ambiguities remaining.

        :return: The modified Dataflow.
        """
        if self._conversion_candidates is None:
            self.learn()
        if len(self.ambiguous_date_columns) > 0:
            raise ValueError('Please resolve date conversion ambiguity in column(s): ' + str(self.ambiguous_date_columns))
        candidates = {col: info.candidate_converters if isinstance(info, InferenceInfo) else info
                      for col, info in self._conversion_candidates.items()}
        converters = get_converters_from_candidates(candidates)
        return self._dataflow.set_column_types(converters) if len(converters) > 0 else self._dataflow

    def __repr__(self):
        if self._conversion_candidates is not None:
            return """Column types conversion candidates:
""" + ',\n'.join(["""{0!r}: {1!r}""".format(col, converters) for col, converters in self.conversion_candidates.items()])
        else:
            return """No column type conversion candidates available."""
