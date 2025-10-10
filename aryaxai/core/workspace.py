import pandas as pd
from pydantic import BaseModel
from typing import Optional
from aryaxai.client.client import APIClient
from aryaxai.common.enums import UserRole
from aryaxai.common.validation import Validate
from aryaxai.common.xai_uris import (
    AVAILABLE_CUSTOM_SERVERS_URI,
    CREATE_PROJECT_URI,
    GET_WORKSPACES_DETAILS_URI,
    START_CUSTOM_SERVER_URI,
    STOP_CUSTOM_SERVER_URI,
    UPDATE_WORKSPACE_URI,
    GET_NOTIFICATIONS_URI,
    CLEAR_NOTIFICATIONS_URI,
)
from aryaxai.core.project import Project
from aryaxai.core.text import TextProject
from aryaxai.core.agent import AgentProject


class Workspace(BaseModel):
    """Class to work with AryaXAI workspaces"""

    organization_id: Optional[str] = None
    created_by: str
    user_workspace_name: str
    workspace_name: str
    created_at: str

    api_client: APIClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client = kwargs.get("api_client")

    def rename_workspace(self, new_workspace_name: str) -> str:
        """rename the current workspace to new name

        :param new_workspace_name: name for the workspace to be renamed to
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "update_workspace": {
                    "workspace_name": new_workspace_name,
                }
            },
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        self.user_workspace_name = new_workspace_name
        return res.get("details")

    def delete_workspace(self) -> str:
        """deletes the current workspace
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {"delete_workspace": self.user_workspace_name},
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def add_user_to_workspace(self, email: str, role: str) -> str:
        """adds user to current workspace

        :param email: user email
        :param role: user role ["admin", "manager", "user"]
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "add_user_workspace": {
                    "email": email,
                    "role": role,
                },
            },
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def remove_user_from_workspace(self, email: str) -> str:
        """removes user from the current workspace

        :param email: user email
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "remove_user_workspace": email,
            },
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def update_user_access_for_workspace(self, email: str, role: UserRole) -> str:
        """update the user access for the workspace

        :param email: user email
        :param role: new user role ["admin", "user"]
        :return: _description_
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "update_user_workspace": {
                    "email": email,
                    "role": role,
                }
            },
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def projects(self) -> pd.DataFrame:
        """get user projects for this Workspace

        :return: Projects details dataframe
        """
        workspace = self.api_client.get(
            f"{GET_WORKSPACES_DETAILS_URI}?workspace_name={self.workspace_name}"
        )
        projects_df = pd.DataFrame(
            workspace.get("data", {}).get("projects", []),
            columns=[
                "user_project_name",
                "access_type",
                "created_by",
                "created_at",
                "updated_at",
                "instance_type",
                "instance_status",
            ],
        )
        return projects_df

    def project(self, project_name: str) -> Project:
        """Select specific project

        :param project_name: Name of the project
        :return: Project
        """
        workspace = self.api_client.get(
            f"{GET_WORKSPACES_DETAILS_URI}?workspace_name={self.workspace_name}"
        )

        project = next(
            filter(lambda project: project.get("user_project_name") == project_name, workspace.get("data", {}).get("projects", [])),
            None,
        )

        if project is None:
            raise Exception("Project Not Found")
        
        if project.get("metadata",{}).get("modality") == "text": return TextProject(api_client=self.api_client, **project)
        elif project.get("metadata",{}).get("modality") == "agent": return AgentProject(api_client=self.api_client, **project)

        return Project(api_client=self.api_client, **project)

    def create_project(
        self,
        project_name: str,
        modality: str,
        project_type: str,
        project_sub_type: Optional[str] = None,
        server_type: Optional[str] = None,
    ) -> Project:
        """creates new project in the current workspace

        :param project_name: name for the project
        :param modality: modality for the project
            Eg:- tabular, image, text
        :project_type: type for the project
            Eg:- classification, regression
        :return: response
        """
        payload = {
            "project_name": project_name,
            "modality": modality,
            "project_type": project_type,
            "project_sub_type": project_sub_type,
            "workspace_name": self.workspace_name,
        }

        if self.organization_id:
            payload["organization_id"] = self.organization_id

        if server_type:
            custom_servers = self.api_client.get(AVAILABLE_CUSTOM_SERVERS_URI)
            Validate.value_against_list(
                "server_type",
                server_type,
                [server["name"] for server in custom_servers],
            )

            payload["instance_type"] = server_type
            payload["server_config"] = {}

        res = self.api_client.post(CREATE_PROJECT_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])
        
        if modality == "text":
            project = TextProject(api_client=self.api_client, **res["details"])
        elif modality == "agent":
            project = AgentProject(api_client=self.api_client, **res["details"])
        else:
            project = Project(api_client=self.api_client, **res["details"])

        return project

    def get_notifications(self) -> pd.DataFrame:
        """get user workspace notifications

        :return: DataFrame
        """
        url = f"{GET_NOTIFICATIONS_URI}?workspace_name={self.workspace_name}"

        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting workspace notifications.")

        notifications = res["details"]

        if not notifications:
            return "No notifications found."

        return pd.DataFrame(notifications).reindex(
            columns=["project_name", "message", "time"]
        )

    def clear_notifications(self) -> str:
        """clear user workspace notifications

        :raises Exception: _description_
        :return: str
        """
        url = f"{CLEAR_NOTIFICATIONS_URI}?workspace_name={self.workspace_name}"

        res = self.api_client.post(url)

        if not res["success"]:
            raise Exception("Error while clearing workspace notifications.")

        return res["details"]

    def start_server(self) -> str:
        """start dedicated workspace server

        :return: response
        """

        res = self.api_client.post(
            f"{START_CUSTOM_SERVER_URI}?workspace_name={self.workspace_name}"
        )

        if not res["success"]:
            raise Exception(res.get("message"))

        return res["message"]

    def stop_server(self) -> str:
        """stop dedicated workspace server

        :return: response
        """
        res = self.api_client.post(
            f"{STOP_CUSTOM_SERVER_URI}?workspace_name={self.workspace_name}"
        )

        if not res["success"]:
            raise Exception(res.get("message"))

        return res["message"]

    def update_server(self, server_type: str) -> str:
        """update dedicated workspace server
        :param server_type: dedicated instance to run workloads
            for all available instances check xai.available_custom_servers()

        :return: response
        """
        custom_servers = self.api_client.get(AVAILABLE_CUSTOM_SERVERS_URI)
        Validate.value_against_list(
            "server_type",
            server_type,
            [server["name"] for server in custom_servers],
        )

        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "update_workspace": {
                    "workspace_name": self.user_workspace_name,
                    "instance_type": server_type,
                },
                "update_operational_hours": {},
            },
        }

        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Server Updated"

    def __print__(self) -> str:
        return f"Workspace(user_workspace_name='{self.user_workspace_name}', created_by='{self.created_by}', created_at='{self.created_at}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()
