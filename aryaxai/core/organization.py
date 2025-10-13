import pandas as pd
from pydantic import BaseModel
from typing import Dict, List, Optional
from aryaxai.client.client import APIClient
from aryaxai.common.validation import Validate
from aryaxai.common.xai_uris import (
    AVAILABLE_CUSTOM_SERVERS_URI,
    CREATE_WORKSPACE_URI,
    GET_WORKSPACES_URI,
    INVITE_USER_ORGANIZATION_URI,
    ORGANIZATION_MEMBERS_URI,
    REMOVE_USER_ORGANIZATION_URI,
)
from aryaxai.core.workspace import Workspace
from aryaxai.common.types import GCSConfig, S3Config, GDriveConfig, SFTPConfig
from aryaxai.common.xai_uris import (
    AVAILABLE_CUSTOM_SERVERS_URI,
    CREATE_DATA_CONNECTORS,
    LIST_DATA_CONNECTORS,
    DELETE_DATA_CONNECTORS,
    TEST_DATA_CONNECTORS,
    DROPBOX_OAUTH,
    LIST_BUCKETS,
    LIST_FILEPATHS,
    COMPUTE_CREDIT_URI,
)
from aryaxai.core.utils import build_url, build_list_data_connector_url


class Organization(BaseModel):
    """Class to work with AryaXAI organizations"""

    organization_id: Optional[str] = None
    name: str
    created_by: str
    created_at: Optional[str] = None

    api_client: APIClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client = kwargs.get("api_client")

    def add_user_to_organization(self, user_email: str) -> str:
        """Add user to Organization

        :param user_email: Email of user to be added to organization.
        :return: response
        """
        payload = {
            "email": user_email,
            "organization_id": self.organization_id,
        }
        res = self.api_client.post(INVITE_USER_ORGANIZATION_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details", "Failed to add user to organization"))

        return res.get("details", "User added successfully")

    def remove_user_from_organization(self, user_email: str) -> str:
        """Remove user from Organization

        :param user_email: Email of user to be removed from organization.
        :return: response
        """
        payload = {
            "organization_user_email": user_email,
            "organization_id": self.organization_id,
        }
        res = self.api_client.post(REMOVE_USER_ORGANIZATION_URI, payload)

        if not res["success"]:
            raise Exception(
                res.get("details", "Failed to remove user from organization")
            )

        return res.get("details", "User removed successfully")

    def member_details(self) -> pd.DataFrame:
        """Organization Member details

        :return: member details dataframe
        """
        res = self.api_client.get(
            f"{ORGANIZATION_MEMBERS_URI}?organization_id={self.organization_id}"
        )

        if not res["success"]:
            raise Exception(
                res.get("details", "Failed to get organization member details")
            )

        member_details_df = pd.DataFrame(
            res.get("details").get("users"),
            columns=[
                "full_name",
                "email",
                "organization_owner",
                "organization_admin",
                "created_at",
            ],
        )

        return member_details_df

    def workspaces(self) -> pd.DataFrame:
        """get user workspaces

        :return: workspace details dataframe
        """

        url = GET_WORKSPACES_URI
        if self.organization_id:
            url = url + f"?organization_id={self.organization_id}"
        workspaces = self.api_client.get(url)

        workspace_df = pd.DataFrame(
            workspaces["details"],
            columns=[
                "user_workspace_name",
                "access_type",
                "created_by",
                "created_at",
                "updated_at",
                "instance_type",
                "instance_status",
            ],
        )

        return workspace_df

    def workspace(self, workspace_name: str) -> Workspace:
        """select specific workspace

        :param workspace_name: Name of the workspace to be used
        :return: Workspace
        """

        url = GET_WORKSPACES_URI
        if self.organization_id:
            url = url + f"?organization_id={self.organization_id}"
        workspaces = self.api_client.get(url)
        user_workspaces = [
            Workspace(api_client=self.api_client, **workspace)
            for workspace in workspaces["details"]
        ]

        workspace = next(
            filter(
                lambda workspace: workspace.user_workspace_name == workspace_name,
                user_workspaces,
            ),
            None,
        )

        if workspace is None:
            raise Exception("Workspace Not Found")

        return workspace

    def create_workspace(
        self, workspace_name: str, server_type: Optional[str] = None
    ) -> Workspace:
        """create user workspace

        :param workspace_name: name for the workspace
        :param server_type: dedicated instance to run workloads
            for all available instances check xai.available_custom_servers()
            defaults to shared
        :return: response
        """
        payload = {"workspace_name": workspace_name}

        if server_type:
            custom_servers = self.api_client.get(AVAILABLE_CUSTOM_SERVERS_URI)
            Validate.value_against_list(
                "server_type",
                server_type,
                [server["name"] for server in custom_servers],
            )

            payload["instance_type"] = server_type
            payload["server_config"] = {}

        if self.organization_id:
            payload["organization_id"] = self.organization_id

        res = self.api_client.post(CREATE_WORKSPACE_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        workspace = Workspace(api_client=self.api_client, **res["workspace_details"])

        return workspace

    def __print__(self) -> str:
        return f"Organization(name='{self.name}', created_by='{self.created_by}', created_at='{self.created_at}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()

    def create_data_connectors(
        self,
        data_connector_name: str,
        data_connector_type: str,
        gcs_config: Optional[GCSConfig] = None,
        s3_config: Optional[S3Config] = None,
        gdrive_config: Optional[GDriveConfig] = None,
        sftp_config: Optional[SFTPConfig] = None,
        hf_token: Optional[str] = None,
    ) -> str:
        """Create Data Connectors for project

        :param data_connector_name: str # name for data connector
        :param data_connector_type: str # type of data connector (s3 | gcs | gdrive)
        :param gcs_config: dict # credentials from service account json
        :param s3_config: dict # credentials of s3 storage
        :param gdrive_config: dict # credentials from service account json
        :param sftp_config: dict # hostname, port, username and password for sftp connection
        :return: response
        """
        if not self.organization_id:
            return "No Organization id found"
        if data_connector_type.lower() == "s3":
            if not s3_config:
                return "No configuration for S3 found"

            Validate.value_against_list(
                "s3 config",
                list(s3_config.keys()),
                ["region", "access_key", "secret_key"],
            )

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "region": s3_config.get("region", "ap-south-1"),
                    "access_key": s3_config.get("access_key"),
                    "secret_key": s3_config.get("secret_key"),
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type.lower() == "gcs":
            if not gcs_config:
                return "No configuration for GCS found"

            Validate.value_against_list(
                "gcs config",
                list(gcs_config.keys()),
                [
                    "project_id",
                    "gcp_project_name",
                    "type",
                    "private_key_id",
                    "private_key",
                    "client_email",
                    "client_id",
                    "auth_uri",
                    "token_uri",
                ],
            )

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "project_id": gcs_config.get("project_id"),
                    "gcp_project_name": gcs_config.get("gcp_project_name"),
                    "service_account_json": {
                        "type": gcs_config.get("type"),
                        "project_id": gcs_config.get("project_id"),
                        "private_key_id": gcs_config.get("private_key_id"),
                        "private_key": gcs_config.get("private_key"),
                        "client_email": gcs_config.get("client_email"),
                        "client_id": gcs_config.get("client_id"),
                        "auth_uri": gcs_config.get("auth_uri"),
                        "token_uri": gcs_config.get("token_uri"),
                    },
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type == "gdrive":
            if not gdrive_config:
                return "No configuration for Google Drive found"

            Validate.value_against_list(
                "gdrive config",
                list(gdrive_config.keys()),
                [
                    "project_id",
                    "type",
                    "private_key_id",
                    "private_key",
                    "client_email",
                    "client_id",
                    "auth_uri",
                    "token_uri",
                ],
            )

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "service_account_json": {
                        "type": gdrive_config.get("type"),
                        "project_id": gdrive_config.get("project_id"),
                        "private_key_id": gdrive_config.get("private_key_id"),
                        "private_key": gdrive_config.get("private_key"),
                        "client_email": gdrive_config.get("client_email"),
                        "client_id": gdrive_config.get("client_id"),
                        "auth_uri": gdrive_config.get("auth_uri"),
                        "token_uri": gdrive_config.get("token_uri"),
                    },
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type == "sftp":
            if not sftp_config:
                return "No configuration for Google Drive found"

            Validate.value_against_list(
                "sftp config",
                list(sftp_config.keys()),
                ["hostname", "port", "username", "password"],
            )

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "sftp_json": {
                        "hostname": sftp_config.get("hostname"),
                        "port": sftp_config.get("port"),
                        "username": sftp_config.get("username"),
                        "password": sftp_config.get("password"),
                    },
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type == "dropbox":
            url_data = self.api_client.get(
                f"{DROPBOX_OAUTH}?organization_id={self.organization_id}"
            )
            print(f"Url: {url_data['details']['url']}")
            code = input(f"{url_data['details']['message']}: ")

            if not code:
                return "No authentication code provided."

            payload = {
                "link_service": {
                    "service_name": data_connector_name,
                    "dropbox_json": {"code": code},
                },
                "link_service_type": data_connector_type,
            }

        if data_connector_type == "HuggingFace":
            if not hf_token:
                return "No hf_token provided"
            
            payload = {
                "link_service":{
                    "service_name": data_connector_name,
                    "hf_token": hf_token
                },
                "link_service_type": data_connector_type
            }

        url = build_url(
            CREATE_DATA_CONNECTORS, data_connector_name, None, self.organization_id
        )
        res = self.api_client.post(url, payload)
        return res["details"]

    def test_data_connectors(self, data_connector_name) -> str:
        """Test connection for the data connectors

        :param data_connector_name: str
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id:
            return "No Project Name or Organization id found"
        url = build_url(
            TEST_DATA_CONNECTORS, data_connector_name, None, self.organization_id
        )
        res = self.api_client.post(url)
        return res["details"]

    def delete_data_connectors(self, data_connector_name) -> str:
        """Delete the data connectors

        :param data_connector_name: str
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id:
            return "No Project Name or Organization id found"

        url = build_url(
            DELETE_DATA_CONNECTORS, data_connector_name, None, self.organization_id
        )
        res = self.api_client.post(url)
        return res["details"]

    def list_data_connectors(self) -> str | pd.DataFrame:
        """List the data connectors"""
        url = build_list_data_connector_url(
            LIST_DATA_CONNECTORS, None, self.organization_id
        )
        res = self.api_client.post(url)

        if res["success"]:
            df = pd.DataFrame(res["details"])
            df = df.drop(
                [
                    "_id",
                    "region",
                    "gcp_project_name",
                    "gcp_project_id",
                    "gdrive_file_name",
                    "project_name",
                ],
                axis=1,
                errors="ignore",
            )
            return df

        return res["details"]

    def list_data_connectors_buckets(self, data_connector_name) -> str | List:
        """List the buckets in data connectors

        :param data_connector_name: str
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id:
            return "No Organization id found"

        url = build_url(LIST_BUCKETS, data_connector_name, None, self.organization_id)
        res = self.api_client.get(url)

        if res.get("message", None):
            print(res["message"])
        return res["details"]

    def list_data_connectors_filepath(
        self,
        data_connector_name,
        bucket_name: Optional[str] = None,
        root_folder: Optional[str] = None,
    ) -> str | Dict:
        """List the filepaths in data connectors

        :param data_connector_name: str
        :param bucket_name: str | Required for S3 & GCS
        :param root_folder: str | Root folder of SFTP
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id:
            return "No Organization id found"

        def get_connector() -> str | pd.DataFrame:
            url = build_list_data_connector_url(
                LIST_DATA_CONNECTORS, None, self.organization_id
            )
            res = self.api_client.post(url)

            if res["success"]:
                df = pd.DataFrame(res["details"])
                filtered_df = df.loc[df["link_service_name"] == data_connector_name]
                if filtered_df.empty:
                    return "No data connector found"
                return filtered_df

            return res["details"]

        connectors = get_connector()
        if isinstance(connectors, pd.DataFrame):
            value = connectors.loc[
                connectors["link_service_name"] == data_connector_name,
                "link_service_type",
            ].values[0]
            ds_type = value

            if ds_type == "s3" or ds_type == "gcs":
                if not bucket_name:
                    return "Missing argument bucket_name"

            if ds_type == "sftp":
                if not root_folder:
                    return "Missing argument root_folder"

        if self.organization_id:
            url = f"{LIST_FILEPATHS}?organization_id={self.organization_id}&link_service_name={data_connector_name}&bucket_name={bucket_name}&root_folder={root_folder}"
        res = self.api_client.get(url)

        if res.get("message", None):
            print(res["message"])
        return res["details"]

    def credits(self):
        url = build_list_data_connector_url(
            COMPUTE_CREDIT_URI, None, self.organization_id
        )
        res = self.api_client.get(url)
        return res["details"]
