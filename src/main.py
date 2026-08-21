import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.middleware import APILoggingMiddleware

from sqlalchemy import text

from .core import tracing
from .core.database import Base, sessionmanager
from .api import router
from .core.configs import EnvironmentOption, settings
from .core.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("✅ Starting up...")
    
    sessionmanager.init(settings.DATABASE_URL)
    
    logger.info("--- Creating Tables ---")
    try:
        async with sessionmanager.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables ready")
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        raise e

    # LLM tracing. Off unless configured; a bad key or an unreachable Langfuse host must only
    # produce a warning here, never keep the service from starting.
    if tracing.tracing_enabled():
        try:
            # auth_check() is a blocking HTTP call — off the event loop, and time-boxed, so a
            # slow or black-holed Langfuse host cannot stall startup.
            await asyncio.wait_for(asyncio.to_thread(tracing.get_langfuse().auth_check), timeout=10)
            logger.info("✅ Langfuse credentials verified")
        except Exception as e:
            logger.warning(f"⚠️ Langfuse auth check failed; traces may not be delivered: {e}")

    yield
    # On shutdown, dispose of the connection pool
    logger.info("🔻 Shutting down...")
    # Flush spans buffered by background tasks, which would otherwise be lost with the process.
    # In a thread because delivery is blocking, and time-boxed so an unreachable Langfuse host
    # cannot hold up shutdown.
    try:
        await asyncio.wait_for(asyncio.to_thread(tracing.shutdown), timeout=10)
    except asyncio.TimeoutError:
        logger.warning("⚠️ Langfuse shutdown timed out; some traces may be lost")
    await sessionmanager.close()
    logger.info("🔻 DB connection closed")


app = FastAPI(
    root_path= settings.APP_ROOT_PATH,
    title=settings.APP_NAME,
    description=settings.APP_DESC,
    version=settings.APP_VERSION,
    docs_url = None if settings.ENVIRONMENT == EnvironmentOption.PRODUCTION else "/docs",
    redoc_url = None if settings.ENVIRONMENT == EnvironmentOption.PRODUCTION else "/redoc",
    openapi_url = None if settings.ENVIRONMENT == EnvironmentOption.PRODUCTION else "/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins
    allow_credentials=True,  # Must be False when using "*"
    allow_methods=["*"],      # Allow all HTTP methods
    allow_headers=["*"],      # Allow all headers
)

app.add_middleware(APILoggingMiddleware)

app.include_router(router)
