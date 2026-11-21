"""MinIO / S3-compatible object storage client for guest images."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from app import config

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)


class MinioClient:
    """Thin wrapper around boto3 S3 for a single bucket on MinIO."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        self.bucket = bucket
        self._endpoint = endpoint

        scheme = "https" if secure else "http"
        endpoint_url = f"{scheme}://{endpoint}"

        self._client: S3Client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            verify=secure,
            use_ssl=secure,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
        logger.info("MinIO client initialized, endpoint=%s bucket=%s", endpoint, bucket)

    def ensure_bucket(self) -> None:
        """Create the configured bucket if it does not already exist."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
            logger.debug("MinIO bucket exists: %s", self.bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self.bucket)
            logger.info("MinIO bucket created: %s", self.bucket)

    def upload_bytes(self, data: bytes, object_name: str, content_type: str = "image/jpeg") -> str:
        """Upload raw bytes and return a presigned URL for the new object."""
        self._client.upload_fileobj(
            BytesIO(data),
            self.bucket,
            object_name,
            ExtraArgs={"ContentType": content_type},
        )
        logger.info("MinIO upload: %s (%d bytes)", object_name, len(data))
        return self.generate_presigned_url(object_name)

    def generate_presigned_url(self, object_name: str, expiration: int = 3600) -> str:
        """Return a presigned GET URL valid for *expiration* seconds."""
        url: str = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": object_name},
            ExpiresIn=expiration,
        )
        return url

    def delete_object(self, object_name: str) -> None:
        """Delete a single object from the bucket."""
        self._client.delete_object(Bucket=self.bucket, Key=object_name)
        logger.info("MinIO object deleted: %s", object_name)

    def download_bytes(self, object_name: str) -> bytes | None:
        """Download raw object bytes; returns None when the object does not exist."""
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=object_name)
            return bytes(response["Body"].read())
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise

    def list_objects(self, prefix: str) -> list[str]:
        """List all object keys under *prefix* in sorted order."""
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, object] = {"Bucket": self.bucket, "Prefix": prefix}
            if token is not None:
                kwargs["ContinuationToken"] = token
            page = self._client.list_objects_v2(**kwargs)
            for item in page.get("Contents", []):
                key = item.get("Key")
                if key:
                    keys.append(str(key))
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
        return sorted(keys)

    def delete_prefix(self, prefix: str) -> list[str]:
        """Delete every object whose key starts with *prefix*; return deleted keys."""
        keys = self.list_objects(prefix)
        for key in keys:
            self.delete_object(key)
        return keys


def create_minio_client() -> MinioClient:
    """Create a MinioClient from application settings."""
    endpoint = config.get("minio.endpoint", "localhost:9000")
    access_key = config.get("minio.access_key", "minioadmin")
    secret_key = config.get("minio.secret_key", "minioadmin")
    bucket = config.get("minio.bucket", "cognitive-companion")
    secure = config.get("minio.secure", False)

    client = MinioClient(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        secure=secure,
    )
    client.ensure_bucket()
    return client
