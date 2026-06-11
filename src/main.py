import sys
import json
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from config.settings import settings
from src.guardrails.ingress import IngressGuardrail
from src.guardrails.egress import EgressGuardrail
from src.cache.semantic import SemanticCache
from src.engine.self_healing import SelfHealingEngine
from openai import AsyncOpenAI

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
logger.add(
    "logs/sentinel_ai.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="500 MB",
)

# Global component instances
ingress_guardrail: Optional[IngressGuardrail] = None
egress_guardrail: Optional[EgressGuardrail] = None
semantic_cache: Optional[SemanticCache] = None
self_healing_engine: Optional[SelfHealingEngine] = None
openai_client: Optional[AsyncOpenAI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize global components on startup and cleanup on shutdown.
    """
    global ingress_guardrail, egress_guardrail, semantic_cache, self_healing_engine, openai_client

    logger.info("Initializing Project SentinelAI components...")

    try:
        ingress_guardrail = IngressGuardrail()
        egress_guardrail = EgressGuardrail()
        semantic_cache = SemanticCache()
        self_healing_engine = SelfHealingEngine()
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        logger.info("All SentinelAI components initialized successfully")

        yield

        logger.info("Shutting down SentinelAI components...")

    except Exception as e:
        logger.error("Failed to initialize SentinelAI: {}", e)
        raise


# Initialize FastAPI app with lifespan
app: FastAPI = FastAPI(
    title="Project SentinelAI",
    description="Autonomous Adversarial Red-Teaming & Self-Healing LLM Firewall",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response Models
class ShieldExecuteRequest(BaseModel):
    """Request payload for the shield/execute endpoint."""
    prompt: str
    expected_schema: Optional[Dict[str, Any]] = None
    max_healing_attempts: int = 2


class ShieldExecuteResponse(BaseModel):
    """Response payload from the shield/execute endpoint."""
    success: bool
    data: Optional[str] = None
    errors: List[str] = []
    cache_hit: bool = False
    healed: bool = False
    risk_score: float = 0.0


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Project SentinelAI",
        "version": "1.0.0",
    }


@app.post("/v1/shield/execute", tags=["Shield"], response_model=ShieldExecuteResponse)
async def shield_execute(request: ShieldExecuteRequest) -> ShieldExecuteResponse:
    """
    Main endpoint for LLM prompt shielding with self-healing capabilities.

    Pipeline:
    1. Ingress guardrail: Detect adversarial prompts
    2. Semantic cache: Check for cached responses
    3. OpenAI completion: Generate safe response
    4. Egress guardrail: Validate output and redact PII
    5. Self-healing: Auto-correct validation failures
    6. Cache update: Store successful response

    Args:
        request: ShieldExecuteRequest containing prompt and optional schema

    Returns:
        ShieldExecuteResponse with final sanitized output
    """
    errors: List[str] = []
    cache_hit: bool = False
    healed: bool = False
    final_response: str = ""
    risk_score: float = 0.0

    try:
        logger.info("Shield execution initiated: prompt_length={}", len(request.prompt))

        # ========== STAGE 1: INGRESS GUARDRAIL ==========
        logger.debug("Stage 1: Running ingress guardrail...")
        ingress_result: Dict[str, Any] = await ingress_guardrail.inspect_prompt(request.prompt)

        if not ingress_result["is_safe"]:
            risk_score = ingress_result["risk_score"]
            error_msg: str = f"Prompt rejected by ingress guardrail. Risk score: {risk_score:.2f}. Flagged patterns: {ingress_result['flagged_patterns']}"
            logger.warning(error_msg)
            errors.append(error_msg)
            return ShieldExecuteResponse(
                success=False,
                data=None,
                errors=errors,
                cache_hit=False,
                healed=False,
                risk_score=risk_score,
            )

        risk_score = ingress_result["risk_score"]
        logger.info("Ingress guardrail passed: risk_score={:.2f}", risk_score)

        # ========== STAGE 2: SEMANTIC CACHE LOOKUP ==========
        logger.debug("Stage 2: Checking semantic cache...")
        cached_response: Optional[str] = await semantic_cache.get_cached_response(request.prompt)

        if cached_response is not None:
            logger.info("Cache HIT: Returning cached response")
            cache_hit = True
            final_response = cached_response
            return ShieldExecuteResponse(
                success=True,
                data=final_response,
                errors=errors,
                cache_hit=True,
                healed=False,
                risk_score=risk_score,
            )

        logger.debug("Cache MISS: Proceeding to LLM generation")

        # ========== STAGE 3: OPENAI COMPLETION ==========
        logger.debug("Stage 3: Generating response via OpenAI...")
        try:
            llm_response = await openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful, safe, and aligned AI assistant. "
                            "Follow all safety guidelines and provide accurate, harmless responses."
                        ),
                    },
                    {
                        "role": "user",
                        "content": request.prompt,
                    },
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=settings.REQUEST_TIMEOUT,
            )

            raw_response: str = llm_response.choices[0].message.content.strip()
            logger.info("OpenAI completion generated: response_length={}", len(raw_response))

        except Exception as openai_err:
            logger.error("OpenAI API error: {}", openai_err)
            errors.append(f"LLM generation failed: {str(openai_err)}")
            return ShieldExecuteResponse(
                success=False,
                data=None,
                errors=errors,
                cache_hit=False,
                healed=False,
                risk_score=risk_score,
            )

        # ========== STAGE 4: EGRESS GUARDRAIL ==========
        logger.debug("Stage 4: Running egress guardrail...")
        egress_result: Dict[str, Any] = await egress_guardrail.validate_output(
            raw_response,
            request.expected_schema
        )

        if egress_result["is_valid"]:
            logger.info("Egress validation passed")
            final_response = egress_result["sanitized_text"]

        else:
            logger.warning("Egress validation failed: errors={}", egress_result["errors"])

            # ========== STAGE 5: SELF-HEALING ==========
            logger.debug("Stage 5: Initiating self-healing engine...")

            healing_attempts: int = 0
            healed_response: str = raw_response

            while healing_attempts < request.max_healing_attempts and not egress_result["is_valid"]:
                healing_attempts += 1
                logger.info("Self-healing attempt {}/{}", healing_attempts, request.max_healing_attempts)

                try:
                    if request.expected_schema:
                        healed_response = await self_healing_engine.heal_json_response(
                            request.prompt,
                            healed_response,
                            request.expected_schema
                        )
                    else:
                        healed_response = await self_healing_engine.heal_response(
                            request.prompt,
                            healed_response,
                            egress_result["errors"]
                        )

                    # Re-validate healed response
                    egress_result = await egress_guardrail.validate_output(
                        healed_response,
                        request.expected_schema
                    )

                    if egress_result["is_valid"]:
                        logger.info("Self-healing successful on attempt {}", healing_attempts)
                        final_response = egress_result["sanitized_text"]
                        healed = True
                        break

                except Exception as healing_err:
                    logger.error("Self-healing attempt {} failed: {}", healing_attempts, healing_err)
                    errors.append(f"Healing attempt {healing_attempts} failed: {str(healing_err)}")
                    continue

            if not egress_result["is_valid"]:
                logger.error("Self-healing exhausted: {} attempts failed", healing_attempts)
                errors.extend(egress_result["errors"])
                return ShieldExecuteResponse(
                    success=False,
                    data=None,
                    errors=errors,
                    cache_hit=False,
                    healed=False,
                    risk_score=risk_score,
                )

        # ========== STAGE 6: CACHE UPDATE ==========
        logger.debug("Stage 6: Updating semantic cache...")
        try:
            await semantic_cache.set_cached_response(request.prompt, final_response)
            logger.info("Response cached successfully")
        except Exception as cache_err:
            logger.warning("Failed to cache response: {}. Continuing.", cache_err)

        # ========== STAGE 7: RETURN RESPONSE ==========
        logger.info("Shield execution completed successfully: healed={}, cache_hit={}", healed, cache_hit)

        return ShieldExecuteResponse(
            success=True,
            data=final_response,
            errors=errors,
            cache_hit=cache_hit,
            healed=healed,
            risk_score=risk_score,
        )

    except Exception as e:
        logger.error("Unexpected error during shield execution: {}", e)
        errors.append(f"Unexpected error: {str(e)}")
        return ShieldExecuteResponse(
            success=False,
            data=None,
            errors=errors,
            cache_hit=False,
            healed=False,
            risk_score=risk_score,
        )


@app.get("/v1/cache/stats", tags=["Cache"])
async def get_cache_stats() -> Dict[str, Any]:
    """Get semantic cache statistics."""
    try:
        stats: Dict[str, Any] = await semantic_cache.get_cache_stats()
        return stats
    except Exception as e:
        logger.error("Error retrieving cache stats: {}", e)
        return {
            "error": str(e),
            "collection_name": settings.VECTOR_COLLECTION_NAME,
        }


@app.delete("/v1/cache/clear", tags=["Cache"])
async def clear_cache() -> Dict[str, str]:
    """Clear the semantic cache (useful for testing)."""
    try:
        await semantic_cache.clear_cache()
        logger.info("Semantic cache cleared")
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error("Error clearing cache: {}", e)
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error("Global exception handler triggered: {} - {}", type(exc).__name__, exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.ENVIRONMENT != "production" else "An error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Project SentinelAI on 0.0.0.0:8000")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.LOG_LEVEL.lower(),
    )
