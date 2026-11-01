from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union
import langchain_core.runnables.config  # noqa: F401  get_current_span() raises without it
from aryaxai.common.xai_uris import (
    RUN_GUARDRAILS_URI,
    RUN_GUARDRAILS_PARALLEL_URI
)
from aryaxai.core.project import Project
from openinference.instrumentation.langchain import get_current_span
from opentelemetry import trace
import time
from .guard_template import Guard


class GuardrailRunResult(TypedDict, total=False):
    details: Dict[str, Any]
    validated_output: Any
    validation_passed: bool
    sanitized_output: Any
    duration: float
    latency: str
    on_fail_action: str
    retry_count: int
    max_retries: int
    start_time: str
    end_time: str


class LangGraphGuardrail:
    """
    Decorator utility for applying Guardrails checks to LangGraph node inputs and outputs
    by calling the Guardrails HTTP APIs.

    Supports two modes:
    - "adhoc": calls /guardrails/run_guardrail per guard passed in the decorator
    - "configured": calls /guardrails/run to use project/model configured guardrails
    """

    def __init__(
        self,
        project: Optional[Project],
        default_apply_on: str = "input",
        llm: Optional[Any] = None,
    ) -> None:
        if project is not None:
            self.client = project.api_client
            self.project_name = project.project_name
            
        self.default_apply_on = default_apply_on
        self.logs: List[Dict[str, Any]] = []
        self.max_retries = 1
        self.retry_delay = 1.0 
        self.tracer = trace.get_tracer(__name__)
        self.llm = llm

    def guardrail(
        self,
        guards: Union[List[str], List[Dict[str, Any]], str, Dict[str, Any], None] = None,
        action: str = "block",
        apply_to: str = "both",
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        # process_entire_state: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator factory.

        - action: 'block' | 'retry' | 'warn'. If validation fails:
          - block: raise ValueError
          - retry: replace with sanitized output when available
          - warn: keep content, log only
        - apply_to: 'input' | 'output' | 'both'
        - input_key: Optional key in state to apply guardrail to for input (defaults to checking 'messages' or 'input')
        - output_key: Optional key in result to apply guardrail to for output (defaults to checking 'messages' or treating as str)
        - process_entire_state: If True, applies guardrails to all strings in the state/result recursively (overrides input_key/output_key)
        """

        if isinstance(guards, (str, dict)):
            guards = [guards]  # type: ignore[assignment]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # In LangGraph, the first argument is the state dict
                state = args[0] if args else kwargs.get("state")
                node_name = getattr(func, "__name__", "unknown_node")

                # Pre-process input
                if state and apply_to in ("input", "both"): 
                    input_content = self._get_content(state, input_key, is_input=True)
                    if input_content is not None:
                        processed = self._process_content(
                            content=input_content,
                            node_name=node_name,
                            content_type="input",
                            action=action,
                            guards=guards,
                        )
                        self._set_content(state, processed, input_key, is_input=True)

                result = func(*args, **kwargs)

                # Post-process output
                if apply_to in ("output", "both"):
                    
                    output_content = self._get_content(result, output_key, is_input=False)
                    if output_content is not None:
                        processed = self._process_content(
                            content=output_content,
                            node_name=node_name,
                            content_type="output",
                            action=action,
                            guards=guards,
                            
                        )
                        if output_key is not None:
                            if isinstance(result, dict):
                                result[output_key] = processed
                        else:
                            if isinstance(result, dict):
                                if "messages" in result:
                                    result["messages"] = processed
                            elif isinstance(result, str):
                                result = processed

                return result

            return wrapper

        return decorator

    def _get_content(self, data: Any, key: Optional[str], is_input: bool) -> Any:
        if key is not None:
            if isinstance(data, dict) and key in data:
                return data[key]
            return None
        else:
            # Default behavior
            if isinstance(data, dict):
                if "messages" in data:
                    return data["messages"]
                elif is_input and "input" in data:
                    return data["input"]
            if not is_input and isinstance(data, str):
                return data
            return None  # No content to process

    def _set_content(self, data: Any, processed: Any, key: Optional[str], is_input: bool) -> None:
        if key is not None:
            if isinstance(data, dict):
                data[key] = processed
        else:
            # Default set
            if isinstance(data, dict):
                if "messages" in data:
                    data["messages"] = processed
                elif is_input and "input" in data:
                    data["input"] = processed

    def _process_content(
        self,
        content: Any,
        node_name: str,
        content_type: str,
        action: str,
        guards: Optional[List[Union[str, Dict[str, Any]]]],
    ) -> Any:
        if not guards:
            return content

        is_list = isinstance(content, list)
        if is_list and content:
            content_to_process = content[-1].content
        elif isinstance(content, str):
            content_to_process = content
        else:
            return content

        current_content = content_to_process
        
        try:
            # get_current_span() is None without LangChain tracing, the checks still have to run
            parent_span = get_current_span() or trace.get_current_span()
            if parent_span is not None:
                ctx = trace.set_span_in_context(parent_span)
                # Create parent span for input/output guardrails with node name for better tracing
                with self.tracer.start_as_current_span(f"guardrails:{content_type}", context=ctx) as parent_gr_span:
                    # Set parent span attributes
                    parent_gr_span.set_attribute("node", str(node_name))  # Add node attribute
                    parent_gr_span.set_attribute("component", str(node_name))
                    parent_gr_span.set_attribute("content_type", str(content_type))
                    
                    # Run all guardrails in parallel
                    parallel_result = self._apply_guardrail_parallel(current_content, guards)
                    
                    if not parallel_result.get("success", False):
                        return current_content

                    # Set timing attributes only on parent span
                    parent_gr_span.set_attribute("start_time", str(parallel_result.get("start_time", "")))
                    parent_gr_span.set_attribute("end_time", str(parallel_result.get("end_time", "")))
                    parent_gr_span.set_attribute("duration", float(parallel_result.get("duration", 0.0)))

                    for guard_result in parallel_result.get("details", []):
                        guard_name = guard_result.get("name", "unknown")
                        
                        run_result: GuardrailRunResult = {
                            "details": guard_result,
                            "validated_output": guard_result.get("validated_output"),
                            "validation_passed": guard_result.get("validation_passed", False),
                            "sanitized_output": guard_result.get("sanitized_output", current_content),
                            "duration": guard_result.get("duration", 0.0),
                            "latency": guard_result.get("latency", "0 ms"),
                            "start_time": parallel_result.get("start_time", ""),
                            "end_time": parallel_result.get("end_time", ""),
                            "retry_count": 0,
                            "max_retries": self.max_retries
                        }
                        
                        run_result["response"] = guard_result
                        run_result["input"] = current_content
                        
                        # Process each guard result with child spans
                        current_content = self._handle_action(
                            original=current_content,
                            run_result=run_result,
                            action=action,
                            node_name=node_name,
                            content_type=content_type,
                            guard_name=guard_name,
                            parent_span=parent_gr_span  # Pass parent span
                        )

            if is_list:
                content[-1].content = current_content
                return content
            else:
                return current_content
                
        except ValueError:
            raise  # block signal for the wrapper
        except Exception as e:
            return content

    def _apply_guardrail_parallel(
        self,
        content: Any,
        guards: List[Union[str, Dict[str, Any]]],
    ) -> Any:
        """
        Run multiple guardrails in parallel using batch endpoint
        
        Args:
            content: Content to validate against multiple guardrails
            guards: List of guardrails to run in parallel
                
        Returns:
            Dictionary containing success status and validation details
        """
        try:
            # Convert string guards to proper dict format with name and class
            guard_specs = []
            for guard in guards:
                guard_specs.append(guard)

            # Prepare payload according to RunGuardParallelPayload schema
            payload = {
                "input_data": content,
                "guards": guard_specs  # List of properly formatted guard specs
            }

            import requests
            start_time = datetime.now()
            response = self.client.post(
                RUN_GUARDRAILS_PARALLEL_URI,
                payload=payload,
            )
            end_time = datetime.now()
            result = response
            result.update({
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds()
            })
                    
            return result

        except Exception as e:
            return {
                "success": False,
                "details": f"Failed to run guardrails in parallel: {str(e)}",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat()
            }

    def _apply_guardrail_with_retry(
        self,
        content: Any,
        guard_spec: Dict[str, Any],
        node_name: str,
        content_type: str,
        action: str,
    ) -> Any:
        current_content = content
        retry_count = 0

        if action == "retry":
            while retry_count <= self.max_retries:
                run_result = self._call_run_guardrail(current_content, guard_spec, content_type)
                validation_passed = bool(run_result.get("validation_passed", True))
                detected_issue = not validation_passed or not run_result.get("success", True)

                if detected_issue and self.llm is not None and retry_count < self.max_retries:
                    prompt = self._build_sanitize_prompt(guard_spec.get("name", "unknown"), current_content, content_type)
                    try:
                        sanitized = self.llm.invoke(prompt)
                        if hasattr(sanitized, "content"):
                            sanitized = sanitized.content
                    except Exception:
                        sanitized = current_content
                    retry_action = f"retry_{retry_count+1}"
                    self._handle_action(
                        original=current_content,
                        run_result=run_result,
                        action=retry_action,
                        node_name=node_name,
                        content_type=content_type,
                        guard_name=guard_spec.get("name") or guard_spec.get("class", "unknown"),
                    )
                    current_content = sanitized
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return self._handle_action(
                        original=current_content,
                        run_result=run_result,
                        action=f"retry_{retry_count}" if retry_count > 0 else action,
                        node_name=node_name,
                        content_type=content_type,
                        guard_name=guard_spec.get("name") or guard_spec.get("class", "unknown"),
                    )
            return current_content
        else:
            # Only one check for non-retry actions
            run_result = self._call_run_guardrail(current_content, guard_spec, content_type)
            return self._handle_action(
                original=current_content,
                run_result=run_result,
                action=action,
                node_name=node_name,
                content_type=content_type,
                guard_name=guard_spec.get("name") or guard_spec.get("class", "unknown"),
            )

    # --------- HTTP calls ---------


    def _call_run_guardrail(self, input_data: Any, guard: Dict[str, Any], content_type: Any) -> GuardrailRunResult:
        uri = RUN_GUARDRAILS_URI
        input = input_data
        
        start_time = datetime.now()
        try:
            body = {"input_data": input, "guard": guard}
            data = self.client.post(uri, body)
            end_time = datetime.now()
            
            details = data.get("details", {}) if isinstance(data, dict) else {}
            result: GuardrailRunResult = {
                "success": bool(data.get("success", False)) if isinstance(data, dict) else False,
                "details": details if isinstance(details, dict) else {},
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }
            if "duration" not in details:
                result["duration"] = (end_time - start_time).total_seconds()
            if isinstance(details, dict):
                if "validated_output" in details:
                    result["validated_output"] = details["validated_output"]
                if "validation_passed" in details:
                    result["validation_passed"] = details["validation_passed"]
                if "sanitized_output" in details:
                    result["sanitized_output"] = details["sanitized_output"]
                if "duration" in details:
                    result["duration"] = details["duration"]
                if "latency" in details:
                    result["latency"] = details["latency"]
                
                result["retry_count"] = 0
                result["max_retries"] = self.max_retries

            result["response"] = data
            result["input"] = input

            return result
            
        except Exception as exc:
            end_time = datetime.now()  # Still capture end time on exception
            raise exc

    # --------- Action handling ---------
    def _handle_action(
        self,
        original: Any,
        run_result: GuardrailRunResult,
        action: str,
        node_name: str,
        content_type: str,
        guard_name: str,
        parent_span: Optional[Any] = None,
    ) -> Any:
        validation_passed = bool(run_result.get("validation_passed", True))
        detected_issue = not validation_passed or not run_result.get("success", True)

        def create_child_span(is_issue: bool):
            
            with self.tracer.start_as_current_span(
                f"guard: {guard_name}",
                context=trace.set_span_in_context(parent_span)
            ) as gr_span:
                gr_span.set_attribute("component", str(node_name))
                gr_span.set_attribute("guard", str(guard_name))
                gr_span.set_attribute("content_type", str(content_type))
                gr_span.set_attribute("detected", is_issue)
                gr_span.set_attribute("action", action)
                
                # Don't set timing attributes on child spans
                gr_span.set_attribute("input.value", self._safe_str(run_result.get("input")))
                gr_span.set_attribute("output.value", json.dumps(run_result.get("response")))
        

        if detected_issue:
            if parent_span is not None:
                create_child_span(True)
                
            if action == "block":
                raise ValueError(f"Guardrail '{guard_name}' detected an issue in {content_type}. Operation blocked.")
            elif "retry" in action:
                return run_result.get("sanitized_output", original)
            else:  # warn
                return original

        # Handle successful validation
        if parent_span is not None:
            create_child_span(False)

        return original

    def _build_sanitize_prompt(self, guard_name: str, content: Any, content_type: str) -> str:
        instructions = {
            "Detect PII": "Sanitize the following text by removing or masking any personally identifiable information (PII). Do not change anything else.",
            "NSFW Text": "Sanitize the following text by removing or masking any not safe for work (NSFW) content. Do not change anything else.",
            "Ban List": "Sanitize the following text by removing or masking any banned words. Do not change anything else.",
            "Bias Check": "Sanitize the following text by removing or masking any biased language. Do not change anything else.",
            "Competitor Check": "Sanitize the following text by removing or masking any competitor names. Do not change anything else.",
            "Correct Language": "Sanitize the following text by correcting the language to the expected language. Do not change anything else.",
            "Gibberish Text": "Sanitize the following text by removing or correcting any gibberish. Do not change anything else.",
            "Profanity Free": "Sanitize the following text by removing or masking any profanity. Do not change anything else.",
            "Secrets Present": "Sanitize the following text by removing or masking any secrets. Do not change anything else.",
            "Toxic Language": "Sanitize the following text by removing or masking any toxic language. Do not change anything else.",
            "Contains String": "Sanitize the following text by removing or masking the specified substring. Do not change anything else.",
            "Detect Jailbreak": "Sanitize the following text by removing or masking any jailbreak attempts. Do not change anything else.",
            "Endpoint Is Reachable": "Sanitize the following text by ensuring any mentioned endpoints are reachable. Do not change anything else.",
            "Ends With": "Sanitize the following text by ensuring it ends with the specified string. Do not change anything else.",
            "Has Url": "Sanitize the following text by removing or masking any URLs. Do not change anything else.",
            "Lower Case": "Sanitize the following text by converting it to lower case. Do not change anything else.",
            "Mentions Drugs": "Sanitize the following text by removing or masking any mentions of drugs. Do not change anything else.",
            "One Line": "Sanitize the following text by ensuring it is a single line. Do not change anything else.",
            "Reading Time": "Sanitize the following text by ensuring its reading time matches the specified value. Do not change anything else.",
            "Redundant Sentences": "Sanitize the following text by removing redundant sentences. Do not change anything else.",
            "Regex Match": "Sanitize the following text by ensuring it matches the specified regex. Do not change anything else.",
            "Sql Column Presence": "Sanitize the following text by ensuring specified SQL columns are present. Do not change anything else.",
            "Two Words": "Sanitize the following text by ensuring it contains only two words. Do not change anything else.",
            "Upper Case": "Sanitize the following text by converting it to upper case. Do not change anything else.",
            "Valid Choices": "Sanitize the following text by ensuring it matches one of the valid choices. Do not change anything else.",
            "Valid Json": "Sanitize the following text by ensuring it is valid JSON. Do not change anything else.",
            "Valid Length": "Sanitize the following text by ensuring its length is valid. Do not change anything else.",
            "Valid Range": "Sanitize the following text by ensuring its value is within the valid range. Do not change anything else.",
            "Valid URL": "Sanitize the following text by ensuring it is a valid URL. Do not change anything else.",
            "Web Sanitization": "Sanitize the following text by removing any unsafe web content. Do not change anything else.",
        }
        instruction = instructions.get(guard_name, "Sanitize the following text according to the guardrail requirements. Do not change anything else.")
        prompt = f"{instruction}\n\nContent:\n{content}"
        return prompt

    @staticmethod
    def _safe_str(value: Any) -> str:
        try:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return str(value)
            if hasattr(value, "content"):
                return str(getattr(value, "content", ""))
            if isinstance(value, (list, tuple)):
                parts = []
                for item in value:
                    parts.append(Guard._safe_str(item) if hasattr(Guard, "_safe_str") else str(item))
                return ", ".join(parts)
            if isinstance(value, dict):
                safe_dict: Dict[str, Any] = {}
                for k, v in value.items():
                    key = str(k)
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        safe_dict[key] = v
                    elif hasattr(v, "content"):
                        safe_dict[key] = str(getattr(v, "content", ""))
                    else:
                        safe_dict[key] = str(v)
                return json.dumps(safe_dict, ensure_ascii=False)
            return str(value)
        except Exception:
            return "<unserializable>"

# Convenience function for quick guardrail setup
def create_guardrail(project: Project) -> LangGraphGuardrail:
    """Quick factory function to create a guardrail instance with a project"""
    return LangGraphGuardrail(project=project)