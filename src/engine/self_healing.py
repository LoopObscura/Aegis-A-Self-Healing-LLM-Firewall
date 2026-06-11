from typing import List
from loguru import logger
from openai import AsyncOpenAI
from config.settings import settings


class SelfHealingEngine:
    """
    SelfHealingEngine automatically corrects LLM output when validation errors are detected.
    Uses zero-temperature completion to deterministically repair structured output.
    """

    def __init__(self) -> None:
        """Initialize the SelfHealingEngine with OpenAI client."""
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model: str = settings.OPENAI_MODEL

        logger.info("SelfHealingEngine initialized with model: {}", self.model)

    async def heal_response(
        self,
        original_prompt: str,
        corrupted_response: str,
        validation_errors: List[str]
    ) -> str:
        """
        Attempt to repair a response that failed validation checks.

        Args:
            original_prompt: The original user prompt.
            corrupted_response: The LLM response that failed validation.
            validation_errors: List of specific validation errors encountered.

        Returns:
            A corrected response string that should pass validation.
        """
        try:
            logger.info(
                "SelfHealingEngine initiated: error_count={}, response_length={}",
                len(validation_errors),
                len(corrupted_response)
            )

            # Construct detailed engineering prompt
            errors_str: str = "\n".join([f"  - {error}" for error in validation_errors])

            engineering_prompt: str = f"""You are an AI output repair specialist. The following response failed validation checks.

ORIGINAL USER PROMPT:
{original_prompt}

CORRUPTED/INVALID RESPONSE:
{corrupted_response}

VALIDATION ERRORS ENCOUNTERED:
{errors_str}

YOUR TASK:
1. Analyze each validation error
2. Identify what made the response invalid
3. Generate a corrected version that:
   - Addresses all validation errors
   - Maintains semantic accuracy to the original prompt
   - Is properly formatted and valid
   - Contains NO PII (emails, SSNs, credit cards, API keys, phone numbers)
   - If JSON, is valid and well-formed

REQUIREMENTS:
- Output ONLY the corrected response
- Do NOT include explanations or preamble
- Do NOT include markdown formatting unless it was in the original
- Ensure strict compliance with all validation requirements

CORRECTED RESPONSE:"""

            logger.debug("Sending healing request to OpenAI with temperature=0.0")

            # Call OpenAI with temperature 0.0 for deterministic output
            healing_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at repairing and correcting AI-generated output while maintaining semantic fidelity.",
                    },
                    {
                        "role": "user",
                        "content": engineering_prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=4000,
                timeout=settings.REQUEST_TIMEOUT,
            )

            healed_response: str = healing_response.choices[0].message.content.strip()

            logger.info(
                "SelfHealingEngine completed: original_length={}, healed_length={}",
                len(corrupted_response),
                len(healed_response)
            )

            return healed_response

        except Exception as e:
            logger.error("SelfHealingEngine failed to repair response: {}. Returning original.", e)
            # Fallback: return original response if healing fails
            return corrupted_response

    async def heal_json_response(
        self,
        original_prompt: str,
        corrupted_json: str,
        expected_schema: dict
    ) -> str:
        """
        Specialized healing for JSON responses that fail schema validation.

        Args:
            original_prompt: The original user prompt.
            corrupted_json: The invalid JSON response.
            expected_schema: The expected JSON schema structure.

        Returns:
            A valid JSON string matching the schema.
        """
        try:
            import json

            schema_str: str = json.dumps(expected_schema, indent=2)

            engineering_prompt: str = f"""You are an expert JSON repair specialist. Fix the following invalid JSON to match the schema.

ORIGINAL PROMPT:
{original_prompt}

INVALID JSON:
{corrupted_json}

EXPECTED SCHEMA:
{schema_str}

YOUR TASK:
1. Parse and fix all JSON syntax errors
2. Ensure all required fields are present
3. Ensure all field types match the schema
4. Remove any fields not in the schema
5. Output ONLY valid JSON (no explanation)

REPAIRED JSON:"""

            logger.debug("Sending JSON healing request to OpenAI")

            healing_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at fixing broken JSON. Output only valid JSON, nothing else.",
                    },
                    {
                        "role": "user",
                        "content": engineering_prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=4000,
                timeout=settings.REQUEST_TIMEOUT,
            )

            healed_json: str = healing_response.choices[0].message.content.strip()

            # Validate the healed JSON is parseable
            try:
                json.loads(healed_json)
                logger.info("JSON healing completed successfully")
                return healed_json
            except json.JSONDecodeError as json_err:
                logger.error("Healed JSON is still invalid: {}. Returning original.", json_err)
                return corrupted_json

        except Exception as e:
            logger.error("JSON healing failed: {}. Returning original.", e)
            return corrupted_json
