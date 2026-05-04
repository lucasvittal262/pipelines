from pprint import pprint

from repositories.drive import GoogleDriveClient

PROJECT_ID = "pure-league-482018-a7"
CLIENT_SECRET_ID = "my-client-secret"


if __name__ == "__main__":
    folder_id = "1lTPttVB2IXCtvdD4rCxw3--qsSW3PvvN"
    client = GoogleDriveClient(
        secret_project_id=PROJECT_ID,
        secret_id=CLIENT_SECRET_ID,
    )

    for file in client.list_files(folder_id=folder_id, page_size=10):
        pprint(file)
        print("")
