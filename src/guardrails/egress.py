import re
import json
from typing import Dict, List, Optional, Any
from loguru import logger


class EgressGuardrail:
    """
    EgressGuardrail validates and sanitizes outgoing LLM responses before returning to users.
    Redacts PII, validates JSON schema compliance, and ensures structural integrity.
    """

    def __init__(self) -> None:
        """Initialize the EgressGuardrail with compiled PII detection patterns."""
        # Compiled regex patterns for PII detection
        self.email_pattern: re.Pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        self.ssn_pattern: re.Pattern = re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
        )

        self.credit_card_pattern: re.Pattern = re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b'
        )

        self.phone_pattern: re.Pattern = re.compile(
            r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        )

        self.api_key_pattern: re.Pattern = re.compile(
            r'(?:api[_-]?key|token|secret|password)[\s]*[:=][\s]*[\'"]?([a-zA-Z0-9_\-]{20,})[\'"]?',
            re.IGNORECASE
        )

        self.ip_address_pattern: re.Pattern = re.compile(
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        )

        logger.info("EgressGuardrail initialized with PII detection patterns")

    async def validate_output(
        self,
        response_text: str,
        expected_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate and sanitize LLM output for PII leakage and schema compliance.

        Args:
            response_text: The raw LLM response text to validate.
            expected_schema: Optional JSON schema dict to validate structural compliance.

        Returns:
            A dictionary containing:
                - is_valid: Boolean indicating if output passed all checks
                - sanitized_text: Cleaned text with PII redacted
                - errors: List of validation errors encountered
        """
        try:
            errors: List[str] = []
            sanitized_text: str = response_text

            # Stage 1: Detect and redact PII
            pii_detections: Dict[str, int] = {}

            email_matches = self.email_pattern.findall(sanitized_text)
            if email_matches:
                pii_detections['emails'] = len(email_matches)
                sanitized_text = self.email_pattern.sub('[REDACTED_EMAIL]', sanitized_text)
                errors.append(f"Detected and redacted {len(email_matches)} email address(es)")

            ssn_matches = self.ssn_pattern.findall(sanitized_text)
            if ssn_matches:
                pii_detections['ssns'] = len(ssn_matches)
                sanitized_text = self.ssn_pattern.sub('[REDACTED_SSN]', sanitized_text)
                errors.append(f"Detected and redacted {len(ssn_matches)} SSN(s)")

            cc_matches = self.credit_card_pattern.findall(sanitized_text)
            if cc_matches:
                pii_detections['credit_cards'] = len(cc_matches)
                sanitized_text = self.credit_card_pattern.sub('[REDACTED_CC]', sanitized_text)
                errors.append(f"Detected and redacted {len(cc_matches)} credit card number(s)")

            phone_matches = self.phone_pattern.findall(sanitized_text)
            if phone_matches:
                pii_detections['phone_numbers'] = len(phone_matches)
                sanitized_text = self.phone_pattern.sub('[REDACTED_PHONE]', sanitized_text)
                errors.append(f"Detected and redacted {len(phone_matches)} phone number(s)")

            api_key_matches = self.api_key_pattern.findall(sanitized_text)
            if api_key_matches:
                pii_detections['api_keys'] = len(api_key_matches)
                sanitized_text = self.api_key_pattern.sub('\\1=[REDACTED_SECRET]', sanitized_text)
                errors.append(f"Detected and redacted {len(api_key_matches)} API key(s)/secret(s)")

            ip_matches = self.ip_address_pattern.findall(sanitized_text)
            if ip_matches:
                pii_detections['ip_addresses'] = len(ip_matches)
                sanitized_text = self.ip_address_pattern.sub('[REDACTED_IP]', sanitized_text)
                errors.append(f"Detected and redacted {len(ip_matches)} IP address(es)")

            if pii_detections:
                logger.warning("PII detections in egress output: {}", pii_detections)

            # Stage 2: Validate JSON schema if provided
            schema_valid: bool = True
            if expected_schema is not None:
                try:
                    parsed_json: Dict[str, Any] = json.loads(sanitized_text)
                    schema_valid = self._validate_schema(parsed_json, expected_schema)
                    if not schema_valid:
                        errors.append("Output does not match expected schema structure")
                        logger.warning("Schema validation failed for output")

                except json.JSONDecodeError as json_err:
                    schema_valid = False
                    errors.append(f"Output is not valid JSON: {str(json_err)}")
                    logger.error("JSON decode error during egress validation: {}", json_err)

            # Stage 3: Validate output length and structure
            if len(sanitized_text) == 0:
                schema_valid = False
                errors.append("Sanitized output is empty")
                logger.warning("Egress validation: output is empty after sanitization")

            if len(sanitized_text) > 1_000_000:
                logger.warning("Egress validation: output exceeds 1MB limit (size: {} bytes)", len(sanitized_text))
                errors.append("Output exceeds size limits")

            is_valid: bool = schema_valid and len(pii_detections) == 0

            logger.info(
                "Egress validation completed: is_valid={}, pii_count={}, errors={}",
                is_valid,
                len(pii_detections),
                len(errors)
            )

            return {
                "is_valid": is_valid,
                "sanitized_text": sanitized_text,
                "errors": errors,
            }

        except Exception as e:
            logger.error("Unexpected error during egress validation: {}", e)
            return {
                "is_valid": False,
                "sanitized_text": response_text,
                "errors": [f"Validation error: {str(e)}"],
            }

    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Recursively validate that data matches the expected schema structure.

        Args:
            data: The data to validate.
            schema: The expected schema structure.

        Returns:
            True if data matches schema, False otherwise.
        """
        try:
            # Check required keys exist
            required_keys: List[str] = schema.get("required", [])
            for key in required_keys:
                if key not in data:
                    logger.debug("Required schema key missing: {}", key)
                    return False

            # Validate properties
            properties: Dict[str, Any] = schema.get("properties", {})
            for key, prop_schema in properties.items():
                if key in data:
                    prop_type: str = prop_schema.get("type", "")

                    if prop_type == "string" and not isinstance(data[key], str):
                        logger.debug("Schema mismatch: key '{}' is not a string", key)
                        return False

                    elif prop_type == "number" and not isinstance(data[key], (int, float)):
                        logger.debug("Schema mismatch: key '{}' is not a number", key)
                        return False

                    elif prop_type == "boolean" and not isinstance(data[key], bool):
                        logger.debug("Schema mismatch: key '{}' is not a boolean", key)
                        return False

                    elif prop_type == "object" and isinstance(data[key], dict):
                        nested_result: bool = self._validate_schema(data[key], prop_schema)
                        if not nested_result:
                            return False

                    elif prop_type == "array" and isinstance(data[key], list):
                        items_schema: Dict[str, Any] = prop_schema.get("items", {})
                        for item in data[key]:
                            if isinstance(item, dict) and items_schema.get("type") == "object":
                                item_result: bool = self._validate_schema(item, items_schema)
                                if not item_result:
                                    return False

            logger.debug("Schema validation passed for data")
            return True

        except Exception as e:
            logger.error("Error during schema validation: {}", e)
            return False
