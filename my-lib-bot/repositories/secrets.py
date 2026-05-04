import base64
import json
import subprocess
from typing import Any

from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SECRET_MANAGER_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def get_secrets(
    project_id: str,
    secret_id: str,
    version_id: str = "latest",
) -> dict[str, Any]:
    """Fetch a JSON-formatted secret from GCP Secret Manager."""
    credentials = _get_credentials()
    service = build("secretmanager", "v1", credentials=credentials)
    resource_name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    try:
        response = service.projects().secrets().versions().access(
            name=resource_name,
        ).execute()
    except HttpError as exc:
        raise RuntimeError(f"Failed to access secret {resource_name!r}: {exc}") from exc

    payload_data = response.get("payload", {}).get("data")
    if payload_data is None:
        raise ValueError(f"Secret payload is missing for {resource_name!r}.")

    try:
        decoded = base64.b64decode(payload_data).decode("utf-8")
        secret = json.loads(decoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Secret payload for {resource_name!r} must be valid JSON."
        ) from exc

    if not isinstance(secret, dict):
        raise ValueError(f"Secret payload for {resource_name!r} must be a JSON object.")

    return secret


def _get_credentials() -> Credentials:
    try:
        credentials, _ = default(scopes=[SECRET_MANAGER_SCOPE])
        return credentials
    except DefaultCredentialsError:
        token = _get_gcloud_access_token()
        return Credentials(token=token, scopes=[SECRET_MANAGER_SCOPE])


def _get_gcloud_access_token() -> str:
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not find Application Default Credentials or a usable gcloud "
            "login. Run `gcloud auth application-default login`, or run "
            "`gcloud auth login` with the account that can access Secret Manager."
        ) from exc

    return result.stdout.strip()


