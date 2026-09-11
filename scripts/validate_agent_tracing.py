from __future__ import annotations

import json

import mlflow
from mlflow.entities import SpanType

from scripts.agent_eval_config import configure_mlflow
from scripts.agent_eval_harness import parse_claude_stream


SAMPLE_STREAM = "\n".join(
    [
        json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "trace-validation",
                "model": "test-model",
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "mcp__canon__get_context",
                            "input": {"task": "validate tracing", "scope": "global"},
                        }
                    ]
                },
            }
        ),
        json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "UNKNOWN",
                        }
                    ]
                },
            }
        ),
        json.dumps(
            {
                "type": "result",
                "session_id": "trace-validation",
                "result": "Done",
                "is_error": False,
            }
        ),
    ]
)


@mlflow.trace(name="validate_canon_agent_trace", span_type=SpanType.AGENT)
def emit_validation_trace():
    parsed = parse_claude_stream(SAMPLE_STREAM)
    for call in parsed["tool_calls"]:
        with mlflow.start_span(name=call["name"], span_type=SpanType.TOOL) as span:
            span.set_inputs(call["input"])
            span.set_outputs(call["output"])
    return {"response": parsed["final_response"], "canon_calls": parsed["canon_calls"]}


def main() -> None:
    configure_mlflow()
    emit_validation_trace()
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id is None:
        raise SystemExit("Tracing validation failed: no trace was created.")
    trace = mlflow.get_trace(trace_id, flush=True)
    if trace is None:
        raise SystemExit(f"Tracing validation failed: cannot load {trace_id}.")
    spans = trace.data.spans
    if not any(span.span_type == SpanType.TOOL for span in spans):
        raise SystemExit("Tracing validation failed: no TOOL span was created.")
    print(f"Trace validation passed: {trace.info.trace_id} ({len(spans)} spans)")


if __name__ == "__main__":
    main()
