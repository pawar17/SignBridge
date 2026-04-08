"""
SignBridge Configuration Module
Loads configuration from environment variables
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = "SignBridge"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_workers: int = 4

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]

    # Database
    database_url: str = "postgresql://user:password@localhost:5432/signbridge"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_echo_sql: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 300
    cache_max_size: int = 1000

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Machine Learning
    model_checkpoint_dir: str = "models/checkpoints"
    sign_recognition_model: Optional[str] = None
    sign_generation_model: Optional[str] = None
    ml_batch_size: int = 32
    ml_max_sequence_length: int = 150
    ml_device: str = "cuda"
    ml_num_workers: int = 4
    ml_use_fp16: bool = True
    model_cache_enabled: bool = True
    model_warmup: bool = True

    # MediaPipe
    mediapipe_model_complexity: int = 2
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    mediapipe_static_image_mode: bool = False

    # File Storage
    upload_dir: str = "uploads"
    max_upload_size: int = 104857600  # 100MB
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    allowed_image_extensions: List[str] = [".jpg", ".jpeg", ".png"]

    # AWS S3 (Optional)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: Optional[str] = None

    # Authentication & Security
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # External Services
    sentry_dsn: Optional[str] = None
    sentry_environment: str = "development"
    sentry_traces_sample_rate: float = 0.1

    # Feature Flags
    enable_sign_to_text: bool = True
    enable_text_to_sign: bool = True
    enable_learning_mode: bool = True
    enable_multi_language: bool = True
    enable_on_device_inference: bool = False

    # Supported Languages
    languages_enabled: List[str] = ["ASL", "ISL", "BSL", "JSL", "LSF", "GSL"]

    # Frontend
    frontend_url: str = "http://localhost:3000"
    websocket_url: str = "ws://localhost:8000/ws"

    # Monitoring
    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    # Logging
    log_file: str = "logs/signbridge.log"
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5

    # Advanced
    max_concurrent_requests: int = 100
    request_timeout: int = 300
    shutdown_timeout: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.app_env == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in test mode"""
        return self.app_env == "testing"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Use this function to access settings throughout the application
    """
    return Settings()


# Convenience instance for importing
settings = get_settings()


if __name__ == "__main__":
    # Print current configuration (for debugging)
    config = get_settings()
    print("=" * 60)
    print("SignBridge Configuration")
    print("=" * 60)
    print(f"App Name: {config.app_name}")
    print(f"Environment: {config.app_env}")
    print(f"Debug Mode: {config.debug}")
    print(f"API Host: {config.api_host}:{config.api_port}")
    print(f"Database: {config.database_url}")
    print(f"ML Device: {config.ml_device}")
    print(f"Languages: {', '.join(config.languages_enabled)}")
    print("=" * 60)
