from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DB_URL: str
    OPENAI_API_KEY: str
    FRONTEND_URL: str = "http://localhost:3000"
    APP_ENV: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()