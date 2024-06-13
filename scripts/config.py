from pydantic import PostgresDsn, field_validator
from pydantic_core import MultiHostUrl
from sqlalchemy.orm import declarative_base
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import create_engine
from pydantic_core.core_schema import ValidationInfo

__all__ = [
    "settings",
    "DeclarativeBase",
    "Database",
    "db",
]


class Settings(BaseSettings):

    # DB
    POSTGRES_HOST: str
    POSTGRES_PORT: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB_NAME: str
    DATABASE_URI: str | None = None

    @field_validator("DATABASE_URI", mode="before")
    @classmethod
    def assemble_db_uri(cls, v: str | None, values: ValidationInfo) -> MultiHostUrl:
        if isinstance(v, str):
            return PostgresDsn(v)
        return PostgresDsn(
            f"postgresql+asyncpg://"
            f"{values.data['POSTGRES_USER']}:"
            f"{values.data['POSTGRES_PASSWORD']}@"
            f"{values.data['POSTGRES_HOST']}:"
            f"{values.data['POSTGRES_PORT']}/"
            f"{values.data['POSTGRES_DB_NAME']}"
        )

    LOG_CONFIG: dict | None = None

    class Config:
        case_sensitive = True


settings = Settings()

DeclarativeBase = declarative_base()


class Database:
    def __init__(self):
        self.__session = None
        self.engine = create_engine(
            str(settings.DATABASE_URI),
        )

    def connect(self):
        self.__session = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

    async def disconnect(self):
        await self.engine.dispose()

    @staticmethod
    async def get_db_session():
        async with db.__session() as session:
            yield session


db = Database()