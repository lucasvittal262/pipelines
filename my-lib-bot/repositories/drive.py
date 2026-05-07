from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from repositories.secrets import get_secrets
from models.drive import DriveFileMetadata


class GoogleDriveClient:
    """Handles Google Drive OAuth and service creation."""

    DEFAULT_SCOPES = ("https://www.googleapis.com/auth/drive.readonly",)

    def __init__(
        self,
        credentials_path: str | Path | None = None,
        token_path: str | Path | None = None,
        scopes: list[str] | tuple[str, ...] | None = None,
        secret_project_id: str | None = None,
        secret_id: str | None = None,
        secret_version_id: str = "latest",
        local_server_host: str = "localhost",
        local_server_port: int = 8080,
        local_server_trailing_slash: bool = False,
    ) -> None:
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self.token_path = (
            Path(token_path)
            if token_path
            else Path(__file__).resolve().parent / "token.json"
        )
        self.scopes = list(scopes or self.DEFAULT_SCOPES)
        self.secret_project_id = secret_project_id
        self.secret_id = secret_id
        self.secret_version_id = secret_version_id
        self.local_server_host = local_server_host
        self.local_server_port = local_server_port
        self.local_server_trailing_slash = local_server_trailing_slash
        self._service: Any | None = None

    def connect(self) -> Any:
        """Return an authenticated Google Drive API service."""
        if self._service is None:
            credentials = self._load_credentials()
            self._service = build("drive", "v3", credentials=credentials)

        return self._service

    def list_files(
        self,
        folder_id: str | None = None,
        page_size: int = 10,
        query: str | None = None,
        fields: str = "nextPageToken, files(id, name, mimeType, webViewLink, parents, size)",
    ) -> list[DriveFileMetadata]:
        """List files from Google Drive."""
        service = self.connect()
        parent_folder = None

        if folder_id:
            folder_id = self._normalize_drive_id(folder_id)
            parent_folder = self.get_file(
                folder_id,
                fields="id, name, mimeType",
            )
            folder_query = f"'{self._escape_query_value(folder_id)}' in parents"
            files_only_query = "mimeType != 'application/vnd.google-apps.folder'"
            folder_query = f"{folder_query} and {files_only_query}"
            query = f"{folder_query} and ({query})" if query else folder_query

        files = []
        page_token = None

        while True:
            request = service.files().list(
                pageSize=page_size,
                q=query,
                fields=fields,
                pageToken=page_token,
            )
            result = request.execute()
            page_files = result.get("files", [])

            if parent_folder:
                for file in page_files:
                    file["parent_folder_id"] = parent_folder["id"]
                    file["parent_folder_name"] = parent_folder["name"]

            for file in page_files:
                if file.get("size"):
                    file["size_mb"] = int(file["size"]) / 1024 / 1024

            files.extend([DriveFileMetadata.from_dict(file) for file in page_files])

            page_token = result.get("nextPageToken")
            if not page_token:
                return files

    def get_file(
        self,
        file_id: str,
        fields: str = "id, name, mimeType, parents, size",
    ) -> dict[str, Any]:
        """Get metadata for a single Drive file or folder."""
        file_id = self._normalize_drive_id(file_id)
        return self.connect().files().get(
            fileId=file_id,
            fields=fields,
        ).execute()

    def _load_credentials(self) -> Credentials:
        credentials = None

        if self.token_path.exists():
            credentials = Credentials.from_authorized_user_file(
                str(self.token_path),
                self.scopes,
            )

        if not credentials or not credentials.valid:
            credentials = self._refresh_or_authorize(credentials)

        return credentials

    def _refresh_or_authorize(self, credentials: Credentials | None) -> Credentials:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = self._load_oauth_flow()
            redirect_uri = f"http://{self.local_server_host}:{self.local_server_port}"
            if self.local_server_trailing_slash:
                redirect_uri += "/"
            print(f"OAuth redirect URI: {redirect_uri}")
            credentials = flow.run_local_server(
                host=self.local_server_host,
                port=self.local_server_port,
                redirect_uri_trailing_slash=self.local_server_trailing_slash,
            )

        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(credentials.to_json())
        return credentials

    def _load_oauth_flow(self) -> InstalledAppFlow:
        if self.secret_project_id and self.secret_id:
            client_config = get_secrets(
                project_id=self.secret_project_id,
                secret_id=self.secret_id,
                version_id=self.secret_version_id,
            )
            client_config = self._normalize_client_config(client_config)
            self._set_local_server_from_client_config(client_config)
            return InstalledAppFlow.from_client_config(
                client_config,
                self.scopes,
            )

        if self.credentials_path is None:
            raise ValueError(
                "Provide either secret_project_id and secret_id, or credentials_path."
            )

        return InstalledAppFlow.from_client_secrets_file(
            str(self.credentials_path),
            self.scopes,
        )

    def _set_local_server_from_client_config(
        self,
        client_config: dict[str, Any],
    ) -> None:
        for redirect_uri in self._get_redirect_uris(client_config):
            parsed = urlparse(redirect_uri)
            if parsed.scheme != "http":
                continue

            if parsed.hostname not in {"localhost", "127.0.0.1"}:
                continue

            self.local_server_host = parsed.hostname
            if parsed.port:
                self.local_server_port = parsed.port
            self.local_server_trailing_slash = parsed.path == "/"
            return

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _normalize_drive_id(value: str) -> str:
        parsed = urlparse(value.strip())

        if parsed.netloc:
            query = parse_qs(parsed.query)
            if query.get("id"):
                return query["id"][0]

            path_parts = [part for part in parsed.path.split("/") if part]
            for marker in ("folders", "d"):
                if marker in path_parts:
                    marker_index = path_parts.index(marker)
                    if len(path_parts) > marker_index + 1:
                        return path_parts[marker_index + 1]

        return value.split("?", 1)[0].strip()

    @staticmethod
    def _get_redirect_uris(client_config: dict[str, Any]) -> list[str]:
        for client_type in ("installed", "web"):
            config = client_config.get(client_type)
            if isinstance(config, dict):
                redirect_uris = config.get("redirect_uris", [])
                if isinstance(redirect_uris, list):
                    return [
                        redirect_uri
                        for redirect_uri in redirect_uris
                        if isinstance(redirect_uri, str)
                    ]

        return []

    @staticmethod
    def _normalize_client_config(client_config: dict[str, Any]) -> dict[str, Any]:
        if "installed" in client_config or "web" in client_config:
            return client_config

        if isinstance(client_config.get("client_secret"), dict):
            nested_config = client_config["client_secret"]
            if "installed" in nested_config or "web" in nested_config:
                return nested_config

        required_keys = {"client_id", "client_secret"}
        if required_keys.issubset(client_config):
            return {
                "installed": {
                    "client_id": client_config["client_id"],
                    "client_secret": client_config["client_secret"],
                    "project_id": client_config.get("project_id"),
                    "auth_uri": client_config.get(
                        "auth_uri",
                        "https://accounts.google.com/o/oauth2/auth",
                    ),
                    "token_uri": client_config.get(
                        "token_uri",
                        "https://oauth2.googleapis.com/token",
                    ),
                    "auth_provider_x509_cert_url": client_config.get(
                        "auth_provider_x509_cert_url",
                        "https://www.googleapis.com/oauth2/v1/certs",
                    ),
                    "redirect_uris": client_config.get(
                        "redirect_uris",
                        ["http://localhost:8080"],
                    ),
                }
            }

        available_keys = ", ".join(sorted(client_config))
        raise ValueError(
            "OAuth client secret must be either the full Google client JSON with "
            "a top-level 'installed' or 'web' key, or a JSON object containing "
            f"'client_id' and 'client_secret'. Available keys: {available_keys}"
        )


def get_drive_service(
    credentials_path: str | Path | None = Path(__file__).resolve().parent / "credentials.json",
    token_path: str | Path | None = None,
    scopes: list[str] | tuple[str, ...] | None = None,
    secret_project_id: str | None = None,
    secret_id: str | None = None,
    secret_version_id: str = "latest",
    local_server_host: str = "localhost",
    local_server_port: int = 8080,
    local_server_trailing_slash: bool = False,
) -> Any:
    """Backward-compatible helper for code that only needs the Drive service."""
    return GoogleDriveClient(
        credentials_path=credentials_path,
        token_path=token_path,
        scopes=scopes,
        secret_project_id=secret_project_id,
        secret_id=secret_id,
        secret_version_id=secret_version_id,
        local_server_host=local_server_host,
        local_server_port=local_server_port,
        local_server_trailing_slash=local_server_trailing_slash,
    ).connect()


if __name__ == "__main__":
    client = GoogleDriveClient(
        credentials_path=Path(__file__).resolve().parent / "credentials.json",
    )

    for file in client.list_files(page_size=10):
        print(file["name"], file["id"], file["mimeType"])
