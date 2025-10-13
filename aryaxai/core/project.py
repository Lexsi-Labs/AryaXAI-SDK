from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, List, Optional
from aryaxai.client.client import APIClient
from aryaxai.common.constants import (
    MODEL_TYPES,
    DATA_DRIFT_DASHBOARD_REQUIRED_FIELDS,
    DATA_DRIFT_STAT_TESTS,
    SYNTHETIC_MODELS_DEFAULT_HYPER_PARAMS,
    TARGET_DRIFT_DASHBOARD_REQUIRED_FIELDS,
    TARGET_DRIFT_STAT_TESTS,
    BIAS_MONITORING_DASHBOARD_REQUIRED_FIELDS,
    MODEL_PERF_DASHBOARD_REQUIRED_FIELDS,
)
from aryaxai.common.types import (
    DataConfig,
    ProjectConfig,
    SyntheticDataConfig,
    SyntheticModelHyperParams,
    GCSConfig,
    S3Config,
    GDriveConfig,
    SFTPConfig,
)
from aryaxai.common.utils import parse_datetime, parse_float, poll_events
from aryaxai.common.validation import Validate
from aryaxai.common.monitoring import (
    BiasMonitoringPayload,
    DataDriftPayload,
    ImageDashboardPayload,
    ModelPerformancePayload,
    TargetDriftPayload,
)

import pandas as pd

from aryaxai.common.xai_uris import (
    ALL_DATA_FILE_URI,
    AVAILABLE_BATCH_SERVERS_URI,
    AVAILABLE_CUSTOM_SERVERS_URI,
    AVAILABLE_SYNTHETIC_CUSTOM_SERVERS_URI,
    AVAILABLE_TAGS_URI,
    CASE_INFO_TEXT_URI,
    CASE_INFO_URI,
    CASE_DTREE_URI,
    CASE_LOGS_TEXT_URI,
    CASE_LOGS_URI,
    CLEAR_NOTIFICATIONS_URI,
    CREATE_OBSERVATION_URI,
    CREATE_POLICY_URI,
    CREATE_SYNTHETIC_PROMPT_URI,
    CREATE_TRIGGER_URI,
    DUPLICATE_MONITORS_URI,
    DASHBOARD_LOGS_URI,
    DOWNLOAD_DASHBOARD_LOGS_URI,
    DELETE_CASE_URI,
    DELETE_DATA_FILE_URI,
    DELETE_SYNTHETIC_MODEL_URI,
    DELETE_SYNTHETIC_TAG_URI,
    DOWNLOAD_SYNTHETIC_DATA_URI,
    DOWNLOAD_TAG_DATA_URI,
    DELETE_TRIGGER_URI,
    DUPLICATE_OBSERVATION_URI,
    DUPLICATE_POLICY_URI,
    EXECUTED_TRIGGER_URI,
    FETCH_EVENTS,
    GENERATE_DASHBOARD_URI,
    GET_AVAILABLE_TEXT_MODELS_URI,
    GET_CASES_URI,
    GET_DASHBOARD_SCORE_URI,
    GET_DASHBOARD_URI,
    GET_DATA_DIAGNOSIS_URI,
    GET_DATA_DRIFT_DIAGNOSIS_URI,
    GET_DATA_SUMMARY_URI,
    GET_EXECUTED_TRIGGER_INFO,
    GET_FEATURE_IMPORTANCE_URI,
    GET_LABELS_URI,
    GET_MODEL_TYPES_URI,
    GET_MODELS_URI,
    GET_MONITORS_ALERTS,
    GET_NOTIFICATIONS_URI,
    GET_OBSERVATION_PARAMS_URI,
    GET_OBSERVATIONS_URI,
    GET_POLICIES_URI,
    GET_POLICY_PARAMS_URI,
    GET_PROJECT_CONFIG,
    GET_SYNTHETIC_DATA_TAGS_URI,
    GET_SYNTHETIC_MODEL_DETAILS_URI,
    GET_SYNTHETIC_MODEL_PARAMS_URI,
    GET_SYNTHETIC_MODELS_URI,
    GET_SYNTHETIC_PROMPT_URI,
    GET_VIEWED_CASE_URI,
    IMAGE_DL,
    MODEL_INFERENCES_URI,
    MODEL_PARAMETERS_URI,
    MODEL_PERFORMANCE_DASHBOARD_URI,
    MODEL_SUMMARY_URI,
    PROJECT_OVERVIEW_TEXT_URI,
    REMOVE_MODEL_URI,
    RUN_DATA_DRIFT_DIAGNOSIS_URI,
    RUN_MODEL_ON_DATA_URI,
    SEARCH_CASE_URI,
    START_CUSTOM_SERVER_URI,
    STOP_CUSTOM_SERVER_URI,
    TABULAR_DL,
    TABULAR_ML,
    TAG_DATA_URI,
    TRAIN_MODEL_URI,
    TRAIN_SYNTHETIC_MODEL_URI,
    UPDATE_ACTIVE_MODEL_URI,
    UPDATE_ACTIVE_INFERENCE_MODEL_URI,
    GET_TRIGGERS_URI,
    UPDATE_OBSERVATION_URI,
    UPDATE_POLICY_URI,
    UPDATE_PROJECT_URI,
    UPDATE_SYNTHETIC_PROMPT_URI,
    UPLOAD_DATA_FILE_INFO_URI,
    UPLOAD_DATA_FILE_URI,
    UPLOAD_DATA_URI,
    UPLOAD_DATA_WITH_CHECK_URI,
    UPLOAD_MODEL_URI,
    CREATE_DATA_CONNECTORS,
    LIST_DATA_CONNECTORS,
    DELETE_DATA_CONNECTORS,
    TEST_DATA_CONNECTORS,
    LIST_BUCKETS,
    LIST_FILEPATHS,
    UPLOAD_FILE_DATA_CONNECTORS,
    DROPBOX_OAUTH,
    VALIDATE_POLICY_URI,
)
import json
import io
from aryaxai.core.alert import Alert

from aryaxai.core.case import Case, CaseText
from aryaxai.core.model_summary import ModelSummary

from aryaxai.core.dashboard import DASHBOARD_TYPES, Dashboard
from datetime import datetime
import re
from aryaxai.core.utils import build_url, build_list_data_connector_url
from aryaxai.core.synthetic import SyntheticDataTag, SyntheticModel, SyntheticPrompt


class Project(BaseModel):
    organization_id: Optional[str] = None
    created_by: str
    project_name: str
    user_project_name: str
    user_workspace_name: str
    workspace_name: str
    metadata: dict

    api_client: APIClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client = kwargs.get("api_client")

    def rename_project(self, new_project_name: str) -> str:
        """Renames current project

        :param new_project_name: new name for the project
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "update_project": {
                    "project_name": new_project_name,
                }
            },
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        self.user_project_name = new_project_name
        return res.get("details")

    def delete_project(self) -> str:
        """Deletes current project

        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "delete_project": self.user_project_name,
            },
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def add_user_to_project(self, email: str, role: str) -> str:
        """Adds new user to project

        :param email: user email
        :param role: user role ["admin", "manager", "user"]
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "add_user_project": {
                    "email": email,
                    "role": role,
                },
            },
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def remove_user_from_project(self, email: str) -> str:
        """Removes user from project

        :param email: user email
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {"remove_user_project": email},
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def update_user_access_for_project(self, email: str, role: str) -> str:
        """Updates user access for project

        :param email: user email
        :param role: user role
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "update_user_project": {
                    "email": email,
                    "role": role,
                },
            },
        }
        res = self.api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def start_server(self) -> str:
        """start dedicated project server

        :return: response
        """
        res = self.api_client.post(
            f"{START_CUSTOM_SERVER_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            raise Exception(res.get("message"))

        return res["message"]

    def stop_server(self) -> str:
        """stop dedicated project server

        :return: response
        """
        res = self.api_client.post(
            f"{STOP_CUSTOM_SERVER_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            raise Exception(res.get("message"))

        return res["message"]

    def update_server(self, server_type: str) -> str:
        """update dedicated project server
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
            "project_name": self.project_name,
            "modify_req": {
                "update_project": {
                    "project_name": self.user_project_name,
                    "instance_type": server_type,
                },
                "update_operational_hours": {},
            },
        }

        res = self.api_client.post(UPDATE_PROJECT_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Server Updated"

    def config(self) -> str:
        """returns config for the project

        :return: response
        """
        res = self.api_client.get(
            f"{GET_PROJECT_CONFIG}?project_name={self.project_name}"
        )
        if res.get("details") != "Not Found":
            res["details"].pop("updated_by", None)
            res["details"]["metadata"].pop("path", None)
            res["details"]["metadata"].pop("avaialble_tags", None)

        return res.get("details")

    def available_tags(self) -> str:
        """get available tags for the project

        :return: response
        """
        res = self.api_client.get(
            f"{AVAILABLE_TAGS_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to get available tags.")
            raise Exception(error_details)

        return res["details"]

    def get_labels(self, feature_name: str) -> List[str]:
        """get unique value of feature name

        :param feature_name: feature name
        :return: unique values of feature
        """
        res = self.api_client.get(
            f"{GET_LABELS_URI}?project_name={self.project_name}&feature_name={feature_name}"
        )

        if not res["success"]:
            error_details = res.get(
                "details", f"Failed to get labels for {feature_name}"
            )
            raise Exception(error_details)

        return res["labels"]

    def files(self) -> pd.DataFrame:
        """Lists all files uploaded by user

        :return: user uploaded files dataframe
        """
        files = self.api_client.get(
            f"{ALL_DATA_FILE_URI}?project_name={self.project_name}"
        )

        if not files.get("details"):
            raise Exception("Please upload files first")

        files_df = (
            pd.DataFrame(files["details"])
            .drop(["metadata", "project_name", "version"], axis=1)
            .rename(columns={"filepath": "file_name"})
        )

        files_df = files_df.loc[files_df["status"] == "active"]
        files_df["file_name"] = files_df["file_name"].apply(
            lambda file_path: file_path.split("/")[-1]
        )

        return files_df

    def file_summary(self, file_name: str) -> pd.DataFrame:
        """File Summary

        :param file_name: user uploaded file name
        :return: file summary dataframe
        """
        files = self.api_client.get(
            f"{ALL_DATA_FILE_URI}?project_name={self.project_name}"
        )

        if not files.get("details"):
            raise Exception("Please upload files first")

        file_data = next(
            filter(
                lambda file: file["filepath"].split("/")[-1] == file_name,
                files["details"],
            ),
            None,
        )

        if not file_data:
            raise Exception("File Not Found, please pass valid file name")

        file_metadata = {
            "file_size_mb": file_data["metadata"]["file_size_mb"],
            "columns": file_data["metadata"]["columns"],
            "rows": file_data["metadata"]["rows"],
        }

        print(file_metadata)

        file_summary_df = pd.DataFrame(file_data["metadata"]["details"])

        return file_summary_df

    def delete_file(self, file_name: str) -> str:
        """deletes file for the project

        :param file_name: uploaded file name
        :return: response
        """
        files = self.api_client.get(
            f"{ALL_DATA_FILE_URI}?project_name={self.project_name}"
        )

        if not files.get("details"):
            raise Exception("Please upload files first")

        file_data = next(
            filter(
                lambda file: file["filepath"] == file_name
                or file["filepath"].split("/")[-1] == file_name,
                files["details"],
            ),
            None,
        )

        if not file_data:
            raise Exception("File Not Found, please pass valid file name")

        payload = {
            "project_name": self.project_name,
            "workspace_name": self.workspace_name,
            "path": file_data["filepath"],
        }

        res = self.api_client.post(DELETE_DATA_FILE_URI, payload)
        return res.get("details")

    def update_config(self, config: DataConfig):
        """updates config for the project

        :param config: updated config
                    {
                        "tags": List[str]
                        "feature_exclude": List[str]
                        "feature_encodings": Dict[str, str]   # {"feature_name":"labelencode | countencode"}
                        "drop_duplicate_uid": bool
                    },

        :return: response
        """
        if not config:
            raise Exception("Please upload config")

        project_config = self.config()

        if project_config == "Not Found":
            raise Exception("Config does not exist, please upload files first")

        available_tags = self.available_tags()

        if config.get("tags"):
            Validate.value_against_list(
                "tags", config["tags"], available_tags.get("user_tags")
            )

        all_unique_features = [
            *project_config["metadata"]["feature_exclude"],
            *project_config["metadata"]["feature_include"],
        ]

        if config.get("feature_exclude"):
            Validate.value_against_list(
                "feature_exclude",
                config["feature_exclude"],
                all_unique_features,
            )

        if config.get("feature_encodings"):
            Validate.value_against_list(
                "feature_encodings_feature",
                list(config["feature_encodings"].keys()),
                list(project_config["metadata"]["feature_encodings"].keys()),
            )
            Validate.value_against_list(
                "feature_encodings_feature",
                list(config["feature_encodings"].values()),
                ["labelencode", "countencode"],
            )

        if config.get("feature_exclude") is None:
            feature_exclude = project_config["metadata"]["feature_exclude"]
        else:
            feature_exclude = config.get("feature_exclude", [])

        feature_include = [
            feature for feature in all_unique_features if feature not in feature_exclude
        ]

        feature_encodings = (
            config.get("feature_encodings")
            or project_config["metadata"]["feature_encodings"]
        )

        drop_duplicate_uid = (
            config.get("drop_duplicate_uid")
            or project_config["metadata"]["drop_duplicate_uid"]
        )

        tags = config.get("tags") or project_config["metadata"]["tags"]

        payload = {
            "project_name": self.project_name,
            "project_type": project_config["project_type"],
            "unique_identifier": project_config["unique_identifier"],
            "true_label": project_config["true_label"],
            "pred_label": project_config.get("pred_label"),
            "config_update": True,
            "metadata": {
                "feature_include": feature_include,
                "feature_exclude": feature_exclude,
                "feature_encodings": feature_encodings,
                "drop_duplicate_uid": drop_duplicate_uid,
                "tags": tags,
            },
        }

        print("Config :-")
        print(json.dumps(payload["metadata"], indent=1))

        res = self.api_client.post(TRAIN_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details", "Failed to update config"))

        poll_events(self.api_client, self.project_name, res.get("event_id"))

    def upload_data(
        self,
        data: str | pd.DataFrame,
        tag: str,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        model_architecture: Optional[str] = None,
        model_type: Optional[str] = None,
        config: Optional[ProjectConfig] = None,
    ) -> str:
        """Uploads data for the current project
        :param data: file path | dataframe to be uploaded
        :param tag: tag for data
        :param config: project config
                {
                    "project_type": "",
                    "unique_identifier": "",
                    "true_label": "",
                    "pred_label": "",
                    "feature_exclude": [],
                    "drop_duplicate_uid: "",
                    "handle_errors": False,
                    "handle_data_imbalance": False, # SMOTE sampling
                    "feature_encodings": Dict[str, str]   # {"feature_name":"labelencode | countencode | onehotencode"}
                },
                defaults to None
        :return: response
        """

        def build_upload_data(data):
            if isinstance(data, str):
                file = open(data, "rb")
                return file
            elif isinstance(data, pd.DataFrame):
                csv_buffer = io.BytesIO()
                data.to_csv(csv_buffer, index=False, encoding="utf-8")
                csv_buffer.seek(0)
                file_name = f"{tag}_sdk_{datetime.now().replace(microsecond=0)}.csv"
                file = (file_name, csv_buffer.getvalue())
                return file
            else:
                raise Exception("Invalid Data Type")

        def upload_file_and_return_path(data, data_type, tag=None) -> str:
            files = {"in_file": build_upload_data(data)}
            res = self.api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type={data_type}&tag={tag}",
                files,
            )

            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        project_config = self.config()

        if project_config == "Not Found":
            if self.metadata.get("modality") == "image":
                if (
                    not model
                    or not model_architecture
                    or not model_type
                    or not model_name
                ):
                    raise Exception("Model details is required for Image project type")

                uploaded_path = upload_file_and_return_path(data, "data", tag)

                model_uploaded_path = upload_file_and_return_path(model, "model")

                payload = {
                    "project_name": self.project_name,
                    "project_type": self.metadata.get("project_type"),
                    "metadata": {
                        "path": uploaded_path,
                        "model_name": model_name,
                        "model_path": model_uploaded_path,
                        "model_architecture": model_architecture,
                        "model_type": model_type,
                        "tag": tag,
                        "tags": [tag],
                    },
                }

            if self.metadata.get("modality") == "tabular":
                if not config:
                    config = {
                        "project_type": "",
                        "unique_identifier": "",
                        "true_label": "",
                        "pred_label": "",
                        "feature_exclude": [],
                        "drop_duplicate_uid": False,
                        "handle_errors": False,
                        "handle_data_imbalance": False,
                    }
                    raise Exception(
                        f"Project Config is required, since no config is set for project \n {json.dumps(config,indent=1)}"
                    )

                Validate.check_for_missing_keys(
                    config, ["project_type", "unique_identifier", "true_label"]
                )

                Validate.value_against_list(
                    "project_type", config, ["classification", "regression"]
                )

                uploaded_path = upload_file_and_return_path(data, "data", tag)

                file_info = self.api_client.post(
                    UPLOAD_DATA_FILE_INFO_URI, {"path": uploaded_path}
                )

                column_names = file_info.get("details").get("column_names")

                Validate.value_against_list(
                    "unique_identifier",
                    config["unique_identifier"],
                    column_names,
                    lambda: self.delete_file(uploaded_path),
                )

                if config.get("feature_exclude"):
                    Validate.value_against_list(
                        "feature_exclude",
                        config["feature_exclude"],
                        column_names,
                        lambda: self.delete_file(uploaded_path),
                    )

                feature_exclude = [
                    config["unique_identifier"],
                    config["true_label"],
                    *config.get("feature_exclude", []),
                ]

                feature_include = [
                    feature
                    for feature in column_names
                    if feature not in feature_exclude
                ]

                feature_encodings = config.get("feature_encodings", {})
                if feature_encodings:
                    Validate.value_against_list(
                        "feature_encodings_feature",
                        list(feature_encodings.keys()),
                        column_names,
                    )
                    Validate.value_against_list(
                        "feature_encodings_feature",
                        list(feature_encodings.values()),
                        ["labelencode", "countencode", "onehotencode"],
                    )

                payload = {
                    "project_name": self.project_name,
                    "project_type": config["project_type"],
                    "unique_identifier": config["unique_identifier"],
                    "true_label": config["true_label"],
                    "pred_label": config.get("pred_label"),
                    "metadata": {
                        "path": uploaded_path,
                        "tag": tag,
                        "tags": [tag],
                        "drop_duplicate_uid": config.get("drop_duplicate_uid"),
                        "handle_errors": config.get("handle_errors", False),
                        "feature_exclude": feature_exclude,
                        "feature_include": feature_include,
                        "feature_encodings": feature_encodings,
                        "feature_actual_used": [],
                        "handle_data_imbalance": config.get(
                            "handle_data_imbalance", False
                        ),
                    },
                }

            res = self.api_client.post(UPLOAD_DATA_WITH_CHECK_URI, payload)

            if not res["success"]:
                self.delete_file(uploaded_path)
                raise Exception(res.get("details"))

            poll_events(self.api_client, self.project_name, res["event_id"])

            return res.get("details")

        if project_config != "Not Found" and config:
            raise Exception("Config already exists, please remove config")

        uploaded_path = upload_file_and_return_path(data, "data", tag)

        payload = {
            "path": uploaded_path,
            "tag": tag,
            "type": "data",
            "project_name": self.project_name,
        }
        res = self.api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details")

    def upload_feature_mapping(self, data: str | dict) -> str:
        """uploads feature mapping for the project

        :param data: response
        :return: response
        """

        def build_upload_data():
            if isinstance(data, str):
                file = open(data, "rb")
                return file
            elif isinstance(data, dict):
                json_buffer = io.BytesIO()
                json_str = json.dumps(data, ensure_ascii=False, indent=4)
                json_buffer.write(json_str.encode("utf-8"))
                json_buffer.seek(0)
                file_name = (
                    f"feature_mapping_sdk_{datetime.now().replace(microsecond=0)}.json"
                )
                file = (file_name, json_buffer.getvalue())
                return file
            else:
                raise Exception("Invalid Data Type")

        def upload_file_and_return_path() -> str:
            files = {"in_file": build_upload_data()}
            res = self.api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type=feature_mapping",
                files,
            )

            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        uploaded_path = upload_file_and_return_path()

        payload = {
            "path": uploaded_path,
            "type": "feature_mapping",
            "project_name": self.project_name,
        }
        res = self.api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details", "Feature mapping upload successful")

    def upload_data_description(self, data: str | pd.DataFrame) -> str:
        """uploads data description for the project

        :param data: response
        :return: response
        """

        def build_upload_data():
            if isinstance(data, str):
                file = open(data, "rb")
                return file
            elif isinstance(data, pd.DataFrame):
                csv_buffer = io.BytesIO()
                data.to_csv(csv_buffer, index=False, encoding="utf-8")
                csv_buffer.seek(0)
                file_name = (
                    f"data_description_sdk_{datetime.now().replace(microsecond=0)}.csv"
                )
                file = (file_name, csv_buffer.getvalue())
                return file
            else:
                raise Exception("Invalid Data Type")

        def upload_file_and_return_path() -> str:
            files = {"in_file": build_upload_data()}
            res = self.api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type=data_description",
                files,
            )

            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        uploaded_path = upload_file_and_return_path()

        payload = {
            "path": uploaded_path,
            "type": "data_description",
            "project_name": self.project_name,
        }
        res = self.api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details", "Data description upload successful")

    def upload_feature_mapping_dataconnectors(
        self,
        data_connector_name: str,
        bucket_name: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> str:
        """uploads feature mapping for the project

        :param data_connector_name: name of the data connector
        :param bucket_name: if data connector has buckets # Example: s3/gcs buckets
        :param file_path: filepath from the bucket for the data to read
        :return: response
        """

        def get_connector() -> str | pd.DataFrame:
            url = build_list_data_connector_url(
                LIST_DATA_CONNECTORS, self.project_name, self.organization_id
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
                if not file_path:
                    return "Missing argument file_path"
        else:
            return connectors

        def upload_file_and_return_path() -> str:
            if not self.project_name:
                return "Missing Project Name"
            if self.organization_id:
                res = self.api_client.post(
                    f"{UPLOAD_FILE_DATA_CONNECTORS}?project_name={self.project_name}&organization_id={self.organization_id}&link_service_name={data_connector_name}&data_type=feature_mapping&bucket_name={bucket_name}&file_path={file_path}"
                )
            else:
                res = self.api_client.post(
                    f"{UPLOAD_FILE_DATA_CONNECTORS}?project_name={self.project_name}&link_service_name={data_connector_name}&data_type=feature_mapping&bucket_name={bucket_name}&file_path={file_path}"
                )
            print(res)
            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        uploaded_path = upload_file_and_return_path()

        payload = {
            "path": uploaded_path,
            "type": "feature_mapping",
            "project_name": self.project_name,
        }
        res = self.api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details", "Feature mapping upload successful")

    def upload_data_description_dataconnectors(
        self,
        data_connector_name: str,
        bucket_name: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> str:
        """uploads data description for the project

        :param data_connector_name: name of the data connector
        :param bucket_name: if data connector has buckets # Example: s3/gcs buckets
        :param file_path: filepath from the bucket for the data to read
        :return: response
        """

        def get_connector() -> str | pd.DataFrame:
            url = build_list_data_connector_url(
                LIST_DATA_CONNECTORS, self.project_name, self.organization_id
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
                if not file_path:
                    return "Missing argument file_path"
        else:
            return connectors

        def upload_file_and_return_path() -> str:
            if not self.project_name:
                return "Missing Project Name"
            if self.organization_id:
                res = self.api_client.post(
                    f"{UPLOAD_FILE_DATA_CONNECTORS}?project_name={self.project_name}&organization_id={self.organization_id}&link_service_name={data_connector_name}&data_type=data_description&bucket_name={bucket_name}&file_path={file_path}"
                )
            else:
                res = self.api_client.post(
                    f"{UPLOAD_FILE_DATA_CONNECTORS}?project_name={self.project_name}&link_service_name={data_connector_name}&data_type=data_description&bucket_name={bucket_name}&file_path={file_path}"
                )
            print(res)
            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        uploaded_path = upload_file_and_return_path()

        payload = {
            "path": uploaded_path,
            "type": "data_description",
            "project_name": self.project_name,
        }
        res = self.api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details", "Data description upload successful")

    def upload_model_types(self) -> dict:
        """Model types which can be uploaded using upload_model()

        :return: response
        """
        model_types = self.api_client.get(GET_MODEL_TYPES_URI)

        return model_types

    def upload_model(
        self,
        model_path: str,
        model_architecture: str,
        model_type: str,
        model_name: str,
        model_data_tags: list,
        model_test_tags: Optional[list],
        instance_type: Optional[str] = None,
        explainability_method: Optional[list] = ["shap"],
        feature_list: Optional[list] = None,
    ):
        """Uploads your custom model on AryaXAI

        :param model_path: path of the model
        :param model_architecture: architecture of model ["machine_learning", "deep_learning"]
        :param model_type: type of the model based on the architecture ["Xgboost","Lgboost","CatBoost","Random_forest","Linear_Regression","Logistic_Regression","Gaussian_NaiveBayes","SGD"]
                use upload_model_types() method to get all allowed model_types
        :param model_name: name of the model
        :param model_data_tags: data tags for model
        :param model_test_tags: test tags for model (optional)
        :param instance_type: instance to be used for uploading model (optional)
        :param explainability_method: explainability method to be used while uploading model ["shap", "lime"] (optional)
        :param feature_list: list of features in sequence which are to be passed in the model (optional)
        """

        def upload_file_and_return_path() -> str:
            files = {"in_file": open(model_path, "rb")}
            model_data_tags_str = ",".join(model_data_tags)
            res = self.api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type=model&tag={model_data_tags_str}",
                files,
            )

            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        model_types = self.api_client.get(GET_MODEL_TYPES_URI)
        valid_model_architecture = model_types.get("model_architecture").keys()
        Validate.value_against_list(
            "model_achitecture", model_architecture, valid_model_architecture
        )

        valid_model_types = model_types.get("model_architecture")[model_architecture]
        Validate.value_against_list("model_type", model_type, valid_model_types)

        tags = self.tags()
        Validate.value_against_list("model_data_tags", model_data_tags, tags)

        if model_test_tags:
            Validate.value_against_list("model_test_tags", model_test_tags, tags)

        uploaded_path = upload_file_and_return_path()

        if instance_type:
            custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
            Validate.value_against_list(
                "instance_type",
                instance_type,
                [
                    server["instance_name"]
                    for server in custom_batch_servers.get("details", [])
                ],
            )

        if explainability_method:
            Validate.value_against_list(
                "explainability_method", explainability_method, ["shap", "lime"]
            )

        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "model_architecture": model_architecture,
            "model_type": model_type,
            "model_path": uploaded_path,
            "model_data_tags": model_data_tags,
            "model_test_tags": model_test_tags,
            "explainability_method": explainability_method,
            "feature_list": feature_list,
        }

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(UPLOAD_MODEL_URI, payload)

        if not res.get("success"):
            raise Exception(res.get("details"))

        poll_events(
            self.api_client,
            self.project_name,
            res["event_id"],
            lambda: self.delete_file(uploaded_path),
        )

    def upload_model_dataconnectors(
        self,
        data_connector_name: str,
        model_architecture: str,
        model_type: str,
        model_name: str,
        model_data_tags: list,
        model_test_tags: Optional[list],
        instance_type: Optional[str] = None,
        explainability_method: Optional[list] = ["shap"],
        bucket_name: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        """Uploads your custom model on AryaXAI

        :param data_connector_name: name of the data connector
        :param model_architecture: architecture of model ["machine_learning", "deep_learning"]
        :param model_type: type of the model based on the architecture ["Xgboost","Lgboost","CatBoost","Random_forest","Linear_Regression","Logistic_Regression","Gaussian_NaiveBayes","SGD"]
                use upload_model_types() method to get all allowed model_types
        :param model_name: name of the model
        :param model_data_tags: data tags for model
        :param model_test_tags: test tags for model (optional)
        :param instance_type: instance to be used for uploading model (optional)
        :param explainability_method: explainability method to be used while uploading model ["shap", "lime"] (optional)
        :param bucket_name: if data connector has buckets # Example: s3/gcs buckets
        :param file_path: filepath from the bucket for the data to read
        """

        def get_connector() -> str | pd.DataFrame:
            url = build_list_data_connector_url(
                LIST_DATA_CONNECTORS, self.project_name, self.organization_id
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
                if not file_path:
                    return "Missing argument file_path"
        else:
            return connectors

        def upload_file_and_return_path() -> str:
            if not self.project_name:
                return "Missing Project Name"
            model_data_tags_str = ",".join(model_data_tags)
            if self.organization_id:
                res = self.api_client.post(
                    f"{UPLOAD_FILE_DATA_CONNECTORS}?project_name={self.project_name}&organization_id={self.organization_id}&link_service_name={data_connector_name}&data_type=model&bucket_name={bucket_name}&file_path={file_path}&tag={model_data_tags_str}"
                )
            else:
                res = self.api_client.post(
                    f"{UPLOAD_FILE_DATA_CONNECTORS}?project_name={self.project_name}&link_service_name={data_connector_name}&data_type=model&bucket_name={bucket_name}&file_path={file_path}&tag={model_data_tags_str}"
                )
            print(res)
            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        model_types = self.api_client.get(GET_MODEL_TYPES_URI)
        valid_model_architecture = model_types.get("model_architecture").keys()
        Validate.value_against_list(
            "model_achitecture", model_architecture, valid_model_architecture
        )

        valid_model_types = model_types.get("model_architecture")[model_architecture]
        Validate.value_against_list("model_type", model_type, valid_model_types)

        tags = self.tags()
        Validate.value_against_list("model_data_tags", model_data_tags, tags)

        if model_test_tags:
            Validate.value_against_list("model_test_tags", model_test_tags, tags)

        uploaded_path = upload_file_and_return_path()

        if instance_type:
            custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
            Validate.value_against_list(
                "instance_type",
                instance_type,
                [
                    server["instance_name"]
                    for server in custom_batch_servers.get("details", [])
                ],
            )

        if explainability_method:
            Validate.value_against_list(
                "explainability_method", explainability_method, ["shap", "lime"]
            )

        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "model_architecture": model_architecture,
            "model_type": model_type,
            "model_path": uploaded_path,
            "model_data_tags": model_data_tags,
            "model_test_tags": model_test_tags,
            "explainability_method": explainability_method,
        }

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(UPLOAD_MODEL_URI, payload)

        if not res.get("success"):
            raise Exception(res.get("details"))

        poll_events(
            self.api_client,
            self.project_name,
            res["event_id"],
            lambda: self.delete_file(uploaded_path),
        )

    def upload_docker_compose(
        self, 
        model_provider: str,
        model_name: str,
        model_type: str,
        model_task_type: str,
        hf_token: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        files = {"in_file": open(file_path, "rb")}
        res = self.api_client.file(
            f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&model_provider={model_provider}&model_name={model_name}&model_type={model_type}&model_task_type={model_task_type}&hf_token={hf_token}",
            files=files
        )
        return res

    def data_observations(self, tag: str) -> pd.DataFrame:
        """Data Observations for the project

        :param tag: tag for data ["Training", "Testing", "Validation", "Custom"]
        :return: data observations dataframe
        """
        payload = {"project_name": self.project_name, "refresh": "false"}
        res = self.api_client.post(f"{GET_DATA_SUMMARY_URI}", payload)
        valid_tags = res["data"]["data"].keys()

        if not valid_tags:
            raise Exception("Data summary not available, please upload data first.")

        if tag not in valid_tags:
            raise Exception(f"Not a vaild tag. Pick a valid tag from {valid_tags}")

        data = {
            "Total Data Volume": res["data"]["overview"]["Total Data Volumn"],
            "Unique Features": res["data"]["overview"]["Unique Features"],
        }

        print(data)
        summary = pd.DataFrame(res["data"]["data"][tag])
        return summary

    def data_warnings(self, tag: str) -> pd.DataFrame:
        """Data warnings for the project

        :param tag: tag for data ["Training", "Testing", "Validation", "Custom"]
        :return: data warnings dataframe
        """
        res = self.api_client.get(
            f"{GET_DATA_DIAGNOSIS_URI}?project_name={self.project_name}"
        )
        valid_tags = res["details"].keys()

        if not valid_tags:
            raise Exception("Data warnings not available, please upload data first.")

        Validate.value_against_list("tag", tag, valid_tags)

        data_warnings = pd.DataFrame(res["details"][tag]["alerts"])
        data_warnings[["Tag", "Description"]] = data_warnings[0].str.extract(
            r"\['(.*?)'] (.+?) #"
        )
        data_warnings["Description"] = data_warnings["Description"].str.replace(
            r"[^\w\s]", "", regex=True
        )
        data_warnings = data_warnings[["Description", "Tag"]]

        data = {"Warnings": len(data_warnings)}
        print(data)

        return data_warnings

    def data_drift_diagnosis(
        self,
        baseline_tags: Optional[List[str]] = None,
        current_tags: Optional[List[str]] = None,
        instance_type: Optional[str] = "",
    ) -> pd.DataFrame:
        """Data Drift Diagnosis for the project

        :param tag: tag for data ["Training", "Testing", "Validation", "Custom"]
        :return: data drift diagnosis dataframe
        """

        if baseline_tags and current_tags:
            if instance_type not in [
                "small",
                "xsmall",
                "2xsmall",
                "3xsmall",
                "medium",
                "xmedium",
                "2xmedium",
                "3xmedium",
                "large",
                "xlarge",
                "2xlarge",
                "3xlarge",
            ]:
                return "instance_type is not valid. Valid types are small, xsmall, 2xsmall, 3xsmall, medium, xmedium, 2xmedium, 3xmedium, large, xlarge, 2xlarge, 3xlarge"

            payload = {
                "project_name": self.project_name,
                "baseline_tags": baseline_tags,
                "current_tags": current_tags,
                "instance_type": instance_type,
            }
            res = self.api_client.post(RUN_DATA_DRIFT_DIAGNOSIS_URI, payload)

            if not res["success"]:
                if res.get("details").get("reason"):
                    raise Exception(res.get("details").get("reason"))
                else:
                    raise Exception(res.get("message"))
            poll_events(self.api_client, self.project_name, res["task_id"])

        res = self.api_client.post(
            f"{GET_DATA_DRIFT_DIAGNOSIS_URI}?project_name={self.project_name}"
        )
        data_drift_diagnosis = pd.DataFrame(res["details"]["detailed_report"]).drop(
            ["current_small_hist", "ref_small_hist"], axis=1
        )

        return data_drift_diagnosis

    def get_default_dashboard(self, type: str) -> Dashboard:
        """get default dashboard

        :param type: type of the dashboard
        :return: Dashboard
        """

        res = self.api_client.get(
            f"{GET_DASHBOARD_URI}?type={type}&project_name={self.project_name}"
        )

        if res["success"]:
            auth_token = self.api_client.get_auth_token()
            query_params = f"?project_name={self.project_name}&type={type}&access_token={auth_token}"
            return Dashboard(
                config=res.get("config"),
                raw_data=res.get("details"),
                query_params=query_params,
            )

        raise Exception(
            "Cannot retrieve default dashboard, please create new dashboard"
        )

    def get_data_drift_dashboard(
        self,
        payload: DataDriftPayload = {},
        instance_type: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        """get data drift dashboard

        :param run_in_background: runs in background without waiting for dashboard generation to complete
        :param instance_type: instance type for running on custom server
        :param payload: data drift payload
            {
                "base_line_tag": "",
                "current_tag": "",
                "stat_test_name": "",
                "stat_test_threshold": "",
                "features_to_use": []
                "date_feature": "",
                "baseline_date": { "start_date": "", "end_date": ""},
                "current_date": { "start_date": "", "end_date": ""},
            }
            defaults to None
            key values for payload:
                stat_test_name=
                    chisquare (Chi-Square test):
                        default for categorical features if the number of labels for feature > 2
                        only for categorical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
                    jensenshannon (Jensen-Shannon distance):
                        for numerical and categorical features
                        returns distance
                        default threshold 0.05
                        drift detected when distance >= threshold
                    ks (Kolmogorov–Smirnov (K-S) test):
                        default for numerical features
                        only for numerical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
                    kl_div (Kullback-Leibler divergence):
                        for numerical and categorical features
                        returns divergence
                        default threshold 0.05
                        drift detected when divergence >= threshold,
                    psi (Population Stability Index):
                        for numerical and categorical features
                        returns psi_value
                        default_threshold=0.1
                        drift detected when psi_value >= threshold
                    wasserstein (Wasserstein distance (normed)):
                        only for numerical features
                        returns distance
                        default threshold 0.05
                        drift detected when distance >= threshold
                    z (Ztest):
                        default for categorical features if the number of labels for feature <= 2
                        only for categorical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
        :return: Dashboard
        """
        if not payload:
            return self.get_default_dashboard("data_drift")

        payload["project_name"] = self.project_name

        # validate required fields
        Validate.check_for_missing_keys(payload, DATA_DRIFT_DASHBOARD_REQUIRED_FIELDS)

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        if payload.get("features_to_use"):
            Validate.value_against_list(
                "features_to_use",
                payload.get("features_to_use", []),
                tags_info["alluniquefeatures"],
            )

        Validate.value_against_list(
            "stat_test_name", payload["stat_test_name"], DATA_DRIFT_STAT_TESTS
        )

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "instance_type",
            instance_type,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(f"{GENERATE_DASHBOARD_URI}?type=data_drift", payload)

        if not res["success"]:
            error_details = res.get("details", "Failed to generate dashboard")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("data_drift")

        return "Data Drift dashboard generation initiated"

    def get_target_drift_dashboard(
        self,
        payload: TargetDriftPayload = {},
        instance_type: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        """get target drift dashboard

        :param run_in_background: runs in background without waiting for dashboard generation to complete
        :param instance_type: instance type for running on custom server
        :param payload: target drift payload
                {
                    "base_line_tag": "",
                    "current_tag": "",
                    "stat_test_name": "",
                    "stat_test_threshold": "",
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "model_type": "",
                    "baseline_true_label": "",
                    "current_true_label": ""
                }
                defaults to None
                key values for payload:
                    stat_test_name=
                        chisquare (Chi-Square test):
                        default for categorical features if the number of labels for feature > 2
                        only for categorical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
                    jensenshannon (Jensen-Shannon distance):
                        for numerical and categorical features
                        returns distance
                        default threshold 0.05
                        drift detected when distance >= threshold
                    kl_div (Kullback-Leibler divergence):
                        for numerical and categorical features
                        returns divergence
                        default threshold 0.05
                        drift detected when divergence >= threshold,
                    psi (Population Stability Index):
                        for numerical and categorical features
                        returns psi_value
                        default_threshold=0.1
                        drift detected when psi_value >= threshold
                    z (Ztest):
                        default for categorical features if the number of labels for feature <= 2
                        only for categorical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
        :return: Dashboard
        """
        if not payload:
            return self.get_default_dashboard("target_drift")

        payload["project_name"] = self.project_name

        # validate required fields
        Validate.check_for_missing_keys(payload, TARGET_DRIFT_DASHBOARD_REQUIRED_FIELDS)

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        Validate.value_against_list("model_type", payload["model_type"], MODEL_TYPES)

        Validate.value_against_list(
            "stat_test_name", payload["stat_test_name"], TARGET_DRIFT_STAT_TESTS
        )

        Validate.value_against_list(
            "baseline_true_label",
            [payload["baseline_true_label"]],
            tags_info["alluniquefeatures"],
        )

        Validate.value_against_list(
            "current_true_label",
            [payload["current_true_label"]],
            tags_info["alluniquefeatures"],
        )

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "instance_type",
            instance_type,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=target_drift", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to get dashboard url")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("target_drift")

        return "Target drift dashboard generation initiated"

    def get_bias_monitoring_dashboard(
        self,
        payload: BiasMonitoringPayload = {},
        instance_type: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        """get bias monitoring dashboard

        :param run_in_background: runs in background without waiting for dashboard generation to complete
        :param instance_type: instance type for running on custom server
        :param payload: bias monitoring payload
                {
                    "base_line_tag": "",
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "model_type": "",
                    "baseline_true_label": "",
                    "baseline_pred_label": "",
                    "features_to_use": []
                }
                defaults to None
        :return: Dashboard
        """
        if not payload:
            return self.get_default_dashboard("biasmonitoring")

        payload["project_name"] = self.project_name

        # validate required fields
        Validate.check_for_missing_keys(
            payload, BIAS_MONITORING_DASHBOARD_REQUIRED_FIELDS
        )

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        Validate.value_against_list("model_type", payload["model_type"], MODEL_TYPES)

        Validate.value_against_list(
            "baseline_true_label",
            [payload["baseline_true_label"]],
            tags_info["alluniquefeatures"],
        )

        Validate.value_against_list(
            "baseline_pred_label",
            [payload["baseline_pred_label"]],
            tags_info["alluniquefeatures"],
        )

        if payload.get("features_to_use"):
            Validate.value_against_list(
                "features_to_use",
                payload.get("features_to_use", []),
                tags_info["alluniquefeatures"],
            )

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "instance_type",
            instance_type,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=biasmonitoring", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to get dashboard url")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("biasmonitoring")

        return "Bias monitoring dashboard generation initiated"

    def get_model_performance_dashboard(
        self,
        payload: ModelPerformancePayload = {},
        instance_type: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        """get model performance dashboard

        :param run_in_background: runs in background without waiting for dashboard generation to complete
        :param instance_type: instance type for running on custom server
        :param payload: model performance payload
                {
                    "base_line_tag": "",
                    "current_tag": "",
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "model_type": "",
                    "baseline_true_label": "",
                    "baseline_pred_label": "",
                    "current_true_label": "",
                    "current_pred_label": ""
                }
                defaults to None
        :return: Dashboard
        """
        if not payload:
            return self.get_default_dashboard("performance")

        payload["project_name"] = self.project_name

        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        if self.metadata.get("modality") == "image":
            Validate.check_for_missing_keys(payload, ["base_line_tag", "current_tag"])

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        if self.metadata.get("modality") == "tabular":
            Validate.check_for_missing_keys(
                payload, MODEL_PERF_DASHBOARD_REQUIRED_FIELDS
            )
            Validate.validate_date_feature_val(
                payload, tags_info["alldatetimefeatures"]
            )

            Validate.value_against_list(
                "model_type", payload["model_type"], MODEL_TYPES
            )

            Validate.value_against_list(
                "baseline_true_label",
                [payload["baseline_true_label"]],
                tags_info["alluniquefeatures"],
            )

            Validate.value_against_list(
                "baseline_pred_label",
                [payload["baseline_pred_label"]],
                tags_info["alluniquefeatures"],
            )

            Validate.value_against_list(
                "current_true_label",
                [payload["current_true_label"]],
                tags_info["alluniquefeatures"],
            )

            Validate.value_against_list(
                "current_pred_label",
                [payload["current_pred_label"]],
                tags_info["alluniquefeatures"],
            )

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "instance_type",
            instance_type,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=performance", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to get dashboard url")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("performance")

        return "Model performance dashboard generation initiated"

    def get_image_property_drift_dashboard(
        self,
        payload: ImageDashboardPayload = {},
        instance_type: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        if not payload:
            return self.get_default_dashboard("image_property_drift")

        payload["project_name"] = self.project_name

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "instance_type",
            instance_type,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=image_property_drift", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to generate dashboard")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("image_property_drift")

        return "Image Property Drift dashboard generation initiated"

    def get_label_drift_dashboard(
        self,
        payload: ImageDashboardPayload = {},
        instance_type: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        if not payload:
            return self.get_default_dashboard("label_drift")

        payload["project_name"] = self.project_name

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "instance_type",
            instance_type,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=label_drift", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to generate dashboard")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("label_drift")

        return "Label Drift dashboard generation initiated"

    def get_property_label_correlation_dashboard(
        self,
        payload: ImageDashboardPayload = {},
        instance_type: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        if not payload:
            return self.get_default_dashboard("property_label_correlation")

        payload["project_name"] = self.project_name

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "instance_type",
            instance_type,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=property_label_correlation", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to generate dashboard")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("property_label_correlation")

        return "Property label correlation dashboard generation initiated"

    def get_image_dataset_drift_dashboard(
        self,
        payload: ImageDashboardPayload = {},
        instance_type: Optional[str] = None,
        run_in_background: bool = False,
    ) -> Dashboard:
        if not payload:
            return self.get_default_dashboard("image_dataset_drift")

        payload["project_name"] = self.project_name

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        Validate.value_against_list(
            "instance_type",
            instance_type,
            [
                server["instance_name"]
                for server in custom_batch_servers.get("details", [])
            ],
        )

        if instance_type:
            payload["instance_type"] = instance_type

        res = self.api_client.post(
            f"{GENERATE_DASHBOARD_URI}?type=image_dataset_drift", payload
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to generate dashboard")
            raise Exception(error_details)

        if not run_in_background:
            poll_events(self.api_client, self.project_name, res["task_id"])
            return self.get_default_dashboard("image_dataset_drift")

        return "Image Dataset Drift dashboard generation initiated"

    def get_all_dashboards(self, type: str, page: Optional[int] = 1):
        """get all dashboard

        :param type: type of the dashboard
        :page: page number defaults to 1
        """

        Validate.value_against_list(
            "type",
            type,
            DASHBOARD_TYPES,
        )

        res = self.api_client.get(
            f"{DASHBOARD_LOGS_URI}?project_name={self.project_name}&type={type}&page={page}",
        )
        if not res["success"]:
            raise Exception(res.get("details", "Failed to get all dashboard"))
        res = res.get("details").get("dashboards")

        logs = pd.DataFrame(res)
        logs.drop(
            columns=[
                "max_features",
                "limit_features",
                "baseline_date",
                "current_date",
                "task_id",
                "date_feature",
                "stat_test_threshold",
                "project_name",
                "file_id",
                "updated_at",
                "features_to_use",
            ],
            inplace=True,
            errors="ignore",
        )
        return logs
    
    def get_score(self, dashboard_id, feature_name):
        resp = self.api_client.get(f"{GET_DASHBOARD_SCORE_URI}?project_name={self.project_name}&dashboard_id={dashboard_id}&feature_name={feature_name}")
        resp = resp.get("details").get("dashboards")
        logs = pd.DataFrame(resp)
        logs.drop(
            columns=[
                "max_features",
                "limit_features",
                "baseline_date",
                "current_date",
                "task_id",
                "date_feature",
                "stat_test_threshold",
                "project_name",
                "file_id",
                "updated_at",
                "features_to_use",
            ],
            inplace=True,
            errors="ignore",
        )
        column_drift_results = logs.metadata[0].get("DatasetColumnDriftResults")
        matched_column_info = next((item for item in column_drift_results if item.get("column_name") == feature_name), None)
        return matched_column_info

    def get_dashboard_metadata(self, type: str, dashboard_id: str) -> Dashboard:
        """get dashboard

        :param type: type of the dashboard
        :param dashboard_id: id of dashboard
        :return: Dashboard
        """
        Validate.value_against_list(
            "type",
            type,
            DASHBOARD_TYPES,
        )

        res = self.api_client.get(
            f"{GET_DASHBOARD_URI}?type={type}&project_name={self.project_name}&dashboard_id={dashboard_id}"
        )

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get dashboard"))

        auth_token = self.api_client.get_auth_token()
        query_params = f"?project_name={self.project_name}&type={type}&dashboard_id={dashboard_id}&access_token={auth_token}"

        return res

    def get_dashboard_log_data(self, type: str):
        """get all dashboard

        :param type: type of the dashboard
        """
        Validate.value_against_list(
            "type",
            type,
            DASHBOARD_TYPES,
        )
        self.api_client.refresh_bearer_token()
        auth_token = self.api_client.get_auth_token()
        query_params = (
            f"project_name={self.project_name}&dashboard_type={type}&token={auth_token}"
        )

        uri = f"{DOWNLOAD_DASHBOARD_LOGS_URI}?{query_params}"
        res = self.api_client.base_request("GET", uri)

        if res.status_code != 200:
            raise Exception(
                res.get(
                    "details", f"Error Downloading Dasboard Logs, {res.status_code}"
                )
            )

        try:
            df = pd.read_csv(io.StringIO(res.text))
        except:
            df = pd.DataFrame()

        return df

    def get_dashboard(self, type: str, dashboard_id: str) -> Dashboard:
        """get dashboard

        :param type: type of the dashboard
        :param dashboard_id: id of dashboard
        :return: Dashboard
        """
        Validate.value_against_list(
            "type",
            type,
            DASHBOARD_TYPES,
        )

        res = self.api_client.get(
            f"{GET_DASHBOARD_URI}?type={type}&project_name={self.project_name}&dashboard_id={dashboard_id}"
        )

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get dashboard"))

        auth_token = self.api_client.get_auth_token()
        query_params = f"?project_name={self.project_name}&type={type}&dashboard_id={dashboard_id}&access_token={auth_token}"

        return Dashboard(
            config=res.get("config"),
            raw_data=res.get("details"),
            query_params=query_params,
        )

    def monitoring_triggers(self) -> pd.DataFrame:
        """get monitoring triggers of project

        :return: DataFrame
        """
        url = f"{GET_TRIGGERS_URI}?project_name={self.project_name}"
        res = self.api_client.get(url)

        if not res["success"]:
            return Exception(res.get("details", "Failed to get triggers"))

        monitoring_triggers = res.get("details", [])

        if not monitoring_triggers:
            return "No monitoring triggers found."

        monitoring_triggers = pd.DataFrame(monitoring_triggers)
        monitoring_triggers = monitoring_triggers[
            monitoring_triggers["deleted"] == False
        ]
        monitoring_triggers = monitoring_triggers.drop("project_name", axis=1)

        return monitoring_triggers

    def duplicate_monitoring_triggers(self, trigger_name, new_trigger_name) -> str:
        if trigger_name == new_trigger_name:
            return "Duplicate trigger name can't be same"
        url = f"{DUPLICATE_MONITORS_URI}?project_name={self.project_name}&trigger_name={trigger_name}&new_trigger_name={new_trigger_name}"
        res = self.api_client.post(url)

        if not res["success"]:
            return res.get("details", "Failed to clone triggers")

        return res["details"]

    def create_monitoring_trigger(self, payload: dict) -> str:
        """create monitoring trigger for project

        :param payload: Data Drift Trigger Payload
                {
                    "trigger_type": ""  #["Data Drift", "Target Drift", "Model Performance"]
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "",   #['daily','weekly','monthly','quarterly','yearly']
                    "stat_test_name": "",
                    "stat_test_threshold": 0,
                    "datadrift_features_per": 7,
                    "dataset_drift_percentage": 50,
                    "features_to_use": [],
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "base_line_tag": "",
                    "current_tag": "",
                    "instance_type": ""  #Instance type to used for running trigger
                } OR Target Drift Trigger Payload
                {
                    "trigger_type": ""  #["Data Drift", "Target Drift", "Model Performance"]
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "",   #['daily','weekly','monthly','quarterly','yearly']
                    "model_type": "",
                    "stat_test_name": ""
                    "stat_test_threshold": 0,
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "base_line_tag": "",
                    "current_tag": "",
                    "baseline_true_label": "",
                    "current_true_label": "",
                    "instance_type": ""  #Instance type to used for running trigger
                } OR Model Performance Trigger Payload
                {
                    "trigger_type": ""  #["Data Drift", "Target Drift", "Model Performance"]
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "",   #['daily','weekly','monthly','quarterly','yearly']
                    "model_type": "",
                    "model_performance_metric": "",
                    "model_performance_threshold": "",
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "base_line_tag": "",
                    "baseline_true_label": "",
                    "baseline_pred_label": "",
                    "instance_type": ""  #Instance type to used for running trigger
                }
        :return: response
        """
        payload["project_name"] = self.project_name

        required_payload_keys = [
            "trigger_type",
            "priority",
            "mail_list",
            "frequency",
            "trigger_name",
        ]

        Validate.check_for_missing_keys(payload, required_payload_keys)

        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "create_trigger": payload,
            },
        }
        res = self.api_client.post(CREATE_TRIGGER_URI, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to create trigger"))

        return "Trigger created successfully."

    def delete_monitoring_trigger(self, name: str) -> str:
        """delete monitoring trigger for project

        :param name: trigger name
        :return: str
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "delete_trigger": name,
            },
        }

        res = self.api_client.post(DELETE_TRIGGER_URI, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to delete trigger"))

        return "Monitoring trigger deleted successfully."

    def alerts(self, page_num: int = 1) -> pd.DataFrame:
        """get monitoring alerts of project

        :param page_num: page num, defaults to 1
        :return: alerts DataFrame
        """
        payload = {"page_num": page_num, "project_name": self.project_name}

        res = self.api_client.post(EXECUTED_TRIGGER_URI, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to get alerts"))

        monitoring_alerts = res.get("details", [])

        if not monitoring_alerts:
            return "No monitoring alerts found."

        return pd.DataFrame(monitoring_alerts)

    def get_alert_details(self, id: str) -> Alert:
        """get alert details by id

        :param id: alert or trigger id
        :return: Alert
        """
        payload = {
            "project_name": self.project_name,
            "id": id,
        }
        res = self.api_client.post(GET_EXECUTED_TRIGGER_INFO, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to get trigger details"))

        trigger_info = res["details"][0]

        if not trigger_info["successful"]:
            return Alert(info={}, detailed_report=[], not_used_features=[])

        trigger_info = trigger_info["details"]

        detailed_report = trigger_info.get("detailed_report")
        not_used_features = trigger_info.get("Not_Used_Features")

        trigger_info.pop("detailed_report", None)
        trigger_info.pop("Not_Used_Features", None)

        return Alert(
            info=trigger_info,
            detailed_report=detailed_report,
            not_used_features=not_used_features,
        )
    
    def get_monitors_alerts(self, monitor_id: str, time: int):
        url = f"{GET_MONITORS_ALERTS}?project_name={self.project_name}&monitor_id={monitor_id}&time={time}"
        res = self.api_client.get(url)
        data = pd.DataFrame(res.get("details"))
        return data

    def get_model_performance(self, model_name: str = None) -> Dashboard:
        """
        get model performance dashboard
        """
        auth_token = self.api_client.get_auth_token()
        dashboard_query_params = f"?type=model_performance&project_name={self.project_name}&access_token={auth_token}"
        raw_data_query_params = f"?project_name={self.project_name}"

        if model_name:
            dashboard_query_params = f"{dashboard_query_params}&model_name={model_name}"
            raw_data_query_params = f"{raw_data_query_params}&model_name={model_name}"

        raw_data = self.api_client.get(
            f"{MODEL_PERFORMANCE_DASHBOARD_URI}{raw_data_query_params}"
        )

        return Dashboard(
            config={},
            query_params=dashboard_query_params,
            raw_data=raw_data.get("details"),
        )

    def model_parameters(self) -> dict:
        """Model Parameters

        :return: response
        """

        model_params = self.api_client.get(MODEL_PARAMETERS_URI)

        return model_params

    def train_model(
        self,
        model_type: str,
        data_config: Optional[DataConfig] = None,
        model_config: Optional[dict] = None,
        instance_type: Optional[str] = None,
    ) -> str:
        """Train new model

        :param model_type: type of model
        :param data_config: config for the data
                        {
                            "tags": List[str]
                            "test_tags": List[str]
                            "feature_exclude": List[str]
                            "feature_encodings": Dict[str, str]   # {"feature_name":"labelencode | countencode"}
                            "drop_duplicate_uid": bool
                            "use_optuna": bool # Allow using Optuna Framework for hyperparameter optimization
                            "sample_percentage": float   # Data sample percentage to be used to train
                            "explainability_sample_percentage": float  # Explainability sample percentage to be used
                            "lime_explainability_iterations": int # Lime Explainability iterations to be used
                            "explainability_method": str # List of explainability method ["shap", "lime"]
                            "handle_data_imbalance": bool # Handle data imbalance using SMOTE
                        },
                        defaults to None
        :param model_config: config with hyper parameters for the model, defaults to None
        :param instance_type: instance to be used for model training
        :return: response
        """

        project_config = self.config()

        if project_config == "Not Found":
            raise Exception("Upload files first")

        available_models = self.available_models()

        Validate.value_against_list("model_type", model_type, available_models)

        all_unique_features = [
            *project_config["metadata"]["feature_exclude"],
            *project_config["metadata"]["feature_include"],
        ]

        if instance_type:
            custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
            Validate.value_against_list(
                "instance_type",
                instance_type,
                [
                    server["instance_name"]
                    for server in custom_batch_servers.get("details", [])
                ],
            )

        if data_config:
            if data_config.get("feature_exclude"):
                Validate.value_against_list(
                    "feature_exclude",
                    data_config["feature_exclude"],
                    all_unique_features,
                )

            if data_config.get("tags"):
                available_tags = self.tags()
                Validate.value_against_list("tags", data_config["tags"], available_tags)

            if data_config.get("test_tags"):
                available_tags = self.tags()
                Validate.value_against_list(
                    "test_tags", data_config["test_tags"], available_tags
                )

            if data_config.get("feature_encodings"):
                Validate.value_against_list(
                    "feature_encodings_feature",
                    list(data_config["feature_encodings"].keys()),
                    list(project_config["metadata"]["feature_encodings"].keys()),
                )
                Validate.value_against_list(
                    "feature_encodings_feature",
                    list(data_config["feature_encodings"].values()),
                    ["labelencode", "countencode", "onehotencode"],
                )

            if data_config.get("sample_percentage"):
                if (
                    data_config["sample_percentage"] < 0
                    or data_config["sample_percentage"] > 1
                ):
                    raise Exception(
                        "Data sample percentage is invalid, select between 0 and 1"
                    )

            if data_config.get("explainability_sample_percentage"):
                if (
                    data_config["explainability_sample_percentage"] < 0
                    or data_config["explainability_sample_percentage"] > 1
                ):
                    raise Exception(
                        "Explainability sample percentage is invalid, select between 0 and 1"
                    )

            if data_config.get("lime_explainability_iterations"):
                if (
                    data_config["lime_explainability_iterations"] < 1
                    or data_config["lime_explainability_iterations"] > 10000
                ):
                    raise Exception(
                        "Lime explainability iterations is invalid, select between 1 and 10000"
                    )

            if data_config.get("explainability_method"):
                Validate.value_against_list(
                    "explainability_method",
                    data_config["explainability_method"],
                    ["shap", "lime"],
                )

        if model_config:
            model_params = self.api_client.get(MODEL_PARAMETERS_URI)
            model_name = f"{model_type}_{project_config['project_type']}".lower()
            model_parameters = model_params.get(model_name)

            if model_parameters:
                for model_config_param in model_config.keys():
                    model_param = model_parameters.get(model_config_param)
                    model_config_param_value = model_config[model_config_param]

                    if not model_param:
                        # raise Exception(
                        #     f"Invalid model config for {model_type} \n {json.dumps(model_parameters)}"
                        # )
                        continue

                    if model_param["type"] == "select":
                        Validate.value_against_list(
                            model_config_param,
                            model_config_param_value,
                            model_param["value"],
                        )
                    elif model_param["type"] == "input":
                        if model_config_param_value > model_param["max"]:
                            raise Exception(
                                f"{model_config_param} value cannot be greater than {model_param['max']}"
                            )
                        if model_config_param_value < model_param["min"]:
                            raise Exception(
                                f"{model_config_param} value cannot be less than {model_param['min']}"
                            )

        data_conf = data_config or {}

        feature_exclude = [
            *data_conf.get("feature_exclude", []),
        ]

        feature_include = [
            feature for feature in all_unique_features if feature not in feature_exclude
        ]

        feature_encodings = (
            data_conf.get("feature_encodings")
            or project_config["metadata"]["feature_encodings"]
        )

        drop_duplicate_uid = (
            data_conf.get("drop_duplicate_uid")
            or project_config["metadata"]["drop_duplicate_uid"]
        )

        explainability_method = (
            data_conf.get("explainability_method")
            or project_config["metadata"]["explainability_method"]
        )

        tags = data_conf.get("tags") or project_config["metadata"]["tags"]
        test_tags = (
            data_conf.get("test_tags") or project_config["metadata"]["test_tags"] or []
        )
        use_optuna = (
            data_conf.get("use_optuna")
            or project_config["metadata"]["use_optuna"]
            or False
        )
        handle_data_imbalance = (
            data_conf.get("handle_data_imbalance")
            or project_config["metadata"]["handle_data_imbalance"]
            or False
        )

        payload = {
            "project_name": self.project_name,
            "project_type": project_config["project_type"],
            "unique_identifier": project_config["unique_identifier"],
            "true_label": project_config["true_label"],
            "metadata": {
                "model_name": model_type,
                "model_parameters": model_config,
                "feature_include": feature_include,
                "feature_exclude": feature_exclude,
                "feature_encodings": feature_encodings,
                "drop_duplicate_uid": drop_duplicate_uid,
                "tags": tags,
                "test_tags": test_tags,
                "use_optuna": use_optuna,
                "explainability_method": explainability_method,
                "handle_data_imbalance": handle_data_imbalance,
            },
            "sample_percentage": data_conf.get("sample_percentage"),
            "explainability_sample_percentage": data_conf.get(
                "explainability_sample_percentage"
            ),
            "lime_explainability_iterations": data_conf.get(
                "lime_explainability_iterations"
            ),
        }

        if instance_type:
            payload["instance_type"] = instance_type

        print("Config :-")
        print(json.dumps(payload["metadata"], indent=1))

        res = self.api_client.post(TRAIN_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        print("\nTraining Initiated")
        poll_events(self.api_client, self.project_name, res["event_id"])

        return "Model Trained Successfully"

    def models(self) -> pd.DataFrame:
        """Models trained for the project

        :return: Dataframe with details of all models
        """
        res = self.api_client.get(f"{GET_MODELS_URI}?project_name={self.project_name}")

        if not res["success"]:
            raise Exception(res["details"])

        staged_models = res["details"]["staged"]

        staged_models_df = pd.DataFrame(staged_models)

        return staged_models_df

    def active_model(self) -> pd.DataFrame:
        """Current Active Model for project

        :return: current active model dataframe
        """
        staged_models_df = self.models()
        active_model = staged_models_df[staged_models_df["status"] == "active"]
        return active_model

    def available_models(self) -> List[str]:
        """Returns all models which can be trained on platform

        :return: list of all models
        """
        res = self.api_client.get(f"{GET_MODELS_URI}?project_name={self.project_name}")

        if not res["success"]:
            raise Exception(res["details"])

        available_models = list(
            map(lambda data: data["model_name"], res["details"]["available"])
        )

        return available_models

    def activate_model(self, model_name: str) -> str:
        """Sets the model to active for the project

        :param model_name: name of the model
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
        }
        res = self.api_client.post(UPDATE_ACTIVE_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")

    def update_inference_model_status(self, model_name: str, activate: bool) -> str:
        """Sets the model to active for inferencing

        :param model_name: name of the model
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "activate": activate,
        }

        res = self.api_client.post(UPDATE_ACTIVE_INFERENCE_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")

    def remove_model(self, model_name: str) -> str:
        """Removes the trained model for the project

        :param model_name: name of the model
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
        }
        res = self.api_client.post(REMOVE_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")

    def model_inference(
        self,
        tag: str,
        model_name: Optional[str] = None,
        instance_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run model inference on data

        :param tag: data tag for running inference
        :param model_name: name of the model, defaults to active model for the project
        :return: model inference dataframe
        """

        available_tags = self.tags()
        if tag not in available_tags:
            raise Exception(
                f"{tag} tag is not valid, select valid tag from :\n{available_tags}"
            )

        models = self.models()

        available_models = models["model_name"].to_list()

        if model_name:
            Validate.value_against_list("model_name", model_name, available_models)

        model = (
            model_name
            or models.loc[models["status"] == "active"]["model_name"].values[0]
        )

        if instance_type:
            custom_batch_servers = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
            Validate.value_against_list(
                "instance_type",
                instance_type,
                [
                    server["instance_name"]
                    for server in custom_batch_servers.get("details", [])
                ],
            )

        run_model_payload = {
            "project_name": self.project_name,
            "model_name": model,
            "tags": tag,
            "instance_type": instance_type,
        }

        run_model_res = self.api_client.post(RUN_MODEL_ON_DATA_URI, run_model_payload)

        if not run_model_res["success"]:
            raise Exception(run_model_res["details"])

        poll_events(
            api_client=self.api_client,
            project_name=self.project_name,
            event_id=run_model_res["event_id"],
        )

        auth_token = self.api_client.get_auth_token()

        uri = f"{DOWNLOAD_TAG_DATA_URI}?project_name={self.project_name}&tag={tag}_{model}_Inference&token={auth_token}"

        tag_data = self.api_client.base_request("GET", uri)

        tag_data_df = pd.read_csv(io.StringIO(tag_data.text))

        return tag_data_df

    def model_inferences(self) -> pd.DataFrame:
        """All model inferences

        :return: model inferences dataframe
        """

        res = self.api_client.get(
            f"{MODEL_INFERENCES_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            raise Exception(res.get("details"))

        model_inference_df = pd.DataFrame(res["details"]["inference_details"])

        return model_inference_df

    def model_summary(self, model_name: Optional[str] = None) -> ModelSummary:
        """Model Summary

        :param model_name: name of the model, defaults to active model for project
        :return: model summary
        """
        if self.metadata.get("modality") == "text":
            res = self.api_client.post(
                f"{PROJECT_OVERVIEW_TEXT_URI}?project_name={self.project_name}"
            )
            return res.get("details")
        else:
            res = self.api_client.get(
                f"{MODEL_SUMMARY_URI}?project_name={self.project_name}"
                + (f"&model_name={model_name}" if model_name else "")
            )

        if not res["success"]:
            raise Exception(res["details"])

        return ModelSummary(api_client=self.api_client, **res.get("details"))

    def tags(self) -> List[str]:
        """Available User Tags for Project

        :return: list of tags
        """
        available_tags = self.available_tags()

        tags = available_tags.get("user_tags")

        return tags

    def all_tags(self) -> List[str]:
        """Available All Tags for Project

        :return: list of tags
        """
        available_tags = self.available_tags()

        tags = available_tags.get("alltags")

        return tags

    def tag_data(self, tag: str, page: Optional[int] = 1) -> pd.DataFrame:
        """Tag Data

        :return: tag data dataframe
        """
        tags = self.all_tags()

        Validate.value_against_list("tag", tag, tags)

        payload = {"page": page, "project_name": self.project_name, "tag": tag}
        res = self.api_client.post(TAG_DATA_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details", "Tag data Not Found"))

        tag_data_df = pd.DataFrame(res["details"]["data"])

        return tag_data_df

    def get_tag_data(
        self,
        tag: str,
    ) -> pd.DataFrame:
        """Run model inference on data

        :param tag: data tag for downloading
        :return: dataframe
        """

        tags = self.available_tags()
        available_tags = tags["alltags"]
        if tag not in available_tags:
            raise Exception(
                f"{tag} tag is not valid, select valid tag from :\n{available_tags}"
            )

        auth_token = self.api_client.get_auth_token()

        uri = f"{DOWNLOAD_TAG_DATA_URI}?project_name={self.project_name}&tag={tag}&token={auth_token}"

        tag_data = self.api_client.base_request("GET", uri)

        tag_data_df = pd.read_csv(io.StringIO(tag_data.text))

        return tag_data_df

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
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"
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
                f"{DROPBOX_OAUTH}?project_name={self.project_name}"
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
            CREATE_DATA_CONNECTORS,
            data_connector_name,
            self.project_name,
            self.organization_id,
        )
        res = self.api_client.post(url, payload)
        return res["details"]

    def test_data_connectors(self, data_connector_name) -> str:
        """Test connection for the data connectors

        :param data_connector_name: str
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"
        url = build_url(
            TEST_DATA_CONNECTORS,
            data_connector_name,
            self.project_name,
            self.organization_id,
        )
        res = self.api_client.post(url)
        return res["details"]

    def delete_data_connectors(self, data_connector_name) -> str:
        """Delete the data connectors

        :param data_connector_name: str
        """
        if not data_connector_name:
            return "Missing argument data_connector_name"
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"

        url = build_url(
            DELETE_DATA_CONNECTORS,
            data_connector_name,
            self.project_name,
            self.organization_id,
        )
        res = self.api_client.post(url)
        return res["details"]

    def list_data_connectors(self) -> str | pd.DataFrame:
        """List the data connectors"""
        url = build_list_data_connector_url(
            LIST_DATA_CONNECTORS, self.project_name, self.organization_id
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
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"

        url = build_url(
            LIST_BUCKETS, data_connector_name, self.project_name, self.organization_id
        )
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
        if not self.organization_id and not self.project_name:
            return "No Project Name or Organization id found"

        def get_connector() -> str | pd.DataFrame:
            url = build_list_data_connector_url(
                LIST_DATA_CONNECTORS, self.project_name, self.organization_id
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

        if self.project_name:
            url = f"{LIST_FILEPATHS}?project_name={self.project_name}&link_service_name={data_connector_name}&bucket_name={bucket_name}&root_folder={root_folder}"
        elif self.organization_id:
            url = f"{LIST_FILEPATHS}?organization_id={self.organization_id}&link_service_name={data_connector_name}&bucket_name={bucket_name}&root_folder={root_folder}"
        res = self.api_client.get(url)

        if res.get("message", None):
            print(res["message"])
        return res["details"]

    def upload_data_dataconnectors(
        self,
        data_connector_name: str,
        tag: str,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        model_architecture: Optional[str] = None,
        model_type: Optional[str] = None,
        bucket_name: Optional[str] = None,
        file_path: Optional[str] = None,
        config: Optional[ProjectConfig] = None,
    ) -> str:
        """Uploads data for the current project with data connectors
        :param data_connector_name: name of the data connector
        :param tag: tag for data
        :param bucket_name: if data connector has buckets # Example: s3/gcs buckets
        :param file_path: filepath from the bucket for the data to read
        :param config: project config
                {
                    "project_type": "",
                    "unique_identifier": "",
                    "true_label": "",
                    "pred_label": "",
                    "feature_exclude": [],
                    "drop_duplicate_uid: "",
                    "handle_errors": False,
                    "feature_encodings": Dict[str, str]   # {"feature_name":"labelencode | countencode | onehotencode"}
                },
                defaults to None
        :return: response
        """
        print("Preparing Data Upload")

        def get_connector() -> str | pd.DataFrame:
            url = build_list_data_connector_url(
                LIST_DATA_CONNECTORS, self.project_name, self.organization_id
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
                if not file_path:
                    return "Missing argument file_path"
        else:
            return connectors

        def upload_file_and_return_path(file_path, data_type, tag=None) -> str:
            if not self.project_name:
                return "Missing Project Name"
            query_params = f"project_name={self.project_name}&link_service_name={data_connector_name}&data_type={data_type}&tag={tag}&bucket_name={bucket_name}&file_path={file_path}"
            if self.organization_id:
                query_params += f"&organization_id={self.organization_id}"
            res = self.api_client.post(f"{UPLOAD_FILE_DATA_CONNECTORS}?{query_params}")
            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        project_config = self.config()

        if project_config == "Not Found":
            if self.metadata.get("modality") == "image":
                if (
                    not model_path
                    or not model_architecture
                    or not model_type
                    or not model_name
                ):
                    raise Exception("Model details is required for Image project type")

                uploaded_path = upload_file_and_return_path(file_path, "data", tag)

                model_uploaded_path = upload_file_and_return_path(model_path, "model")

                payload = {
                    "project_name": self.project_name,
                    "project_type": self.metadata.get("project_type"),
                    "metadata": {
                        "path": uploaded_path,
                        "model_name": model_name,
                        "model_path": model_uploaded_path,
                        "model_architecture": model_architecture,
                        "model_type": model_type,
                        "tag": tag,
                        "tags": [tag],
                    },
                }

            if self.metadata.get("modality") == "tabular":
                if not config:
                    config = {
                        "project_type": "",
                        "unique_identifier": "",
                        "true_label": "",
                        "pred_label": "",
                        "feature_exclude": [],
                        "drop_duplicate_uid": False,
                        "handle_errors": False,
                    }
                    raise Exception(
                        f"Project Config is required, since no config is set for project \n {json.dumps(config,indent=1)}"
                    )

                Validate.check_for_missing_keys(
                    config, ["project_type", "unique_identifier", "true_label"]
                )

                Validate.value_against_list(
                    "project_type", config, ["classification", "regression"]
                )

                uploaded_path = upload_file_and_return_path(file_path, "data", tag)

                file_info = self.api_client.post(
                    UPLOAD_DATA_FILE_INFO_URI, {"path": uploaded_path}
                )

                column_names = file_info.get("details").get("column_names")

                Validate.value_against_list(
                    "unique_identifier",
                    config["unique_identifier"],
                    column_names,
                    lambda: self.delete_file(uploaded_path),
                )

                if config.get("feature_exclude"):
                    Validate.value_against_list(
                        "feature_exclude",
                        config["feature_exclude"],
                        column_names,
                        lambda: self.delete_file(uploaded_path),
                    )

                feature_exclude = [
                    config["unique_identifier"],
                    config["true_label"],
                    *config.get("feature_exclude", []),
                ]

                feature_include = [
                    feature
                    for feature in column_names
                    if feature not in feature_exclude
                ]

                feature_encodings = config.get("feature_encodings", {})
                if feature_encodings:
                    Validate.value_against_list(
                        "feature_encodings_feature",
                        list(feature_encodings.keys()),
                        column_names,
                    )
                    Validate.value_against_list(
                        "feature_encodings_feature",
                        list(feature_encodings.values()),
                        ["labelencode", "countencode", "onehotencode"],
                    )

                payload = {
                    "project_name": self.project_name,
                    "project_type": config["project_type"],
                    "unique_identifier": config["unique_identifier"],
                    "true_label": config["true_label"],
                    "pred_label": config.get("pred_label"),
                    "metadata": {
                        "path": uploaded_path,
                        "tag": tag,
                        "tags": [tag],
                        "drop_duplicate_uid": config.get("drop_duplicate_uid"),
                        "handle_errors": config.get("handle_errors", False),
                        "feature_exclude": feature_exclude,
                        "feature_include": feature_include,
                        "feature_encodings": feature_encodings,
                        "feature_actual_used": [],
                    },
                }

            res = self.api_client.post(UPLOAD_DATA_WITH_CHECK_URI, payload)

            if not res["success"]:
                self.delete_file(uploaded_path)
                raise Exception(res.get("details"))

            poll_events(self.api_client, self.project_name, res["event_id"])

            return res.get("details")

        if project_config != "Not Found" and config:
            raise Exception("Config already exists, please remove config")

        uploaded_path = upload_file_and_return_path(file_path, "data", tag)

        payload = {
            "path": uploaded_path,
            "tag": tag,
            "type": "data",
            "project_name": self.project_name,
        }
        res = self.api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details")

    def cases(
        self,
        unique_identifier: Optional[str] = None,
        tag: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: Optional[int] = 1
    ) -> pd.DataFrame:
        """Cases for the Project

        :param unique_identifier: unique identifer of the case for filtering, defaults to None
        :param tag: data tag for filtering, defaults to None
        :param start_date: start date for filtering, defaults to None
        :param end_date: end data for filtering, defaults to None
        :return: casse details dataframe
        """

        def get_cases():
            payload = {
                "project_name": self.project_name,
                "page_num": page,
            }
            res = self.api_client.post(GET_CASES_URI, payload)
            return res

        def search_cases():
            payload = {
                "project_name": self.project_name,
                "unique_identifier": unique_identifier,
                "start_date": start_date,
                "end_date": end_date,
                "tag": tag,
                "page_num": page,
            }
            res = self.api_client.post(SEARCH_CASE_URI, payload)
            return res

        cases = (
            search_cases()
            if unique_identifier or tag or start_date or end_date
            else get_cases()
        )

        if not cases["success"]:
            raise Exception("No cases found")

        cases_df = pd.DataFrame(cases.get("details"))

        return cases_df

    def case_info(
        self,
        unique_identifer: str,
        case_id: Optional[str] = None,
        tag: Optional[str] = None,
        model_name: Optional[str] = None,
        instance_type: Optional[str] = None,
        xai: Optional[list] = [],
        risk_policies: Optional[bool] = False
    ) -> Case:
        """Case Info

        :param unique_identifer: unique identifer of case
        :param case_id: case id, defaults to None
        :param tag: case tag, defaults to None
        :param model_name: trained model name, defaults to None
        :param instance_type: instance to be used for case
                Eg:- nova-0.5, nova-1, nova-1.5
        :param components: various components to be generated with predictions
                Eg:- ['feature_importance', 'similar_cases', 'policies']
        :return: Case object with details
        """
        payload = {
            "project_name": self.project_name,
            "case_id": case_id,
            "unique_identifier": unique_identifer,
            "tag": tag,
            "model_name": model_name,
            "instance_type": instance_type,
            "risk_policies": risk_policies,
            "xai": xai
        }
        if self.metadata.get("modality") == "text":
            res = self.api_client.post(CASE_INFO_TEXT_URI, payload)
            return CaseText(**res["details"])
        else:
            res = self.api_client.post(CASE_INFO_URI, payload)
        if not res["success"]:
            raise Exception(res["details"])

        if self.metadata.get("modality") == "tabular" and "dtree" in xai:
            prediction_path_payload = {
                "project_name": self.project_name,
                "unique_identifier": unique_identifer,
                "case_id": case_id,
                "model_name": res["details"]["model_name"],
                "data_id": res["details"]["data_id"],
                "instance_type": instance_type,
            }

            dtree_res = self.api_client.post(CASE_DTREE_URI, prediction_path_payload)
            if dtree_res["success"]:
                res["details"]["case_prediction_svg"] = dtree_res["details"][
                    "case_prediction_svg"
                ]
                res["details"]["case_prediction_path"] = dtree_res["details"][
                    "case_prediction_path"
                ]
                res["details"]["audit_trail"]["cost"]["xai_dtree"] = dtree_res["details"]["cost_dtree"]
                res["details"]["audit_trail"]["time"]["xai_dtree"] = dtree_res["details"]["time_dtree"]
                res["details"]["audit_trail"]["compute_type"]["xai_dtree"] = dtree_res["details"]["compute_type"]
        res["details"]["project_name"] = self.project_name
        res["details"]["api_client"] = self.api_client
        case = Case(**res["details"])

        return case

    def delete_cases(
        self,
        unique_identifier: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> str:
        """Delete Cases

        :param unique_identifier: unique identifier of case, defaults to None
        :param start_date: start date of case, defaults to None
        :param end_date: end date of case, defaults to None
        :param tag: tag of case, defaults to None
        :return: response
        """
        if tag:
            all_tags = self.all_tags()
            Validate.value_against_list("tag", tag, all_tags)

        paylod = {
            "project_name": self.project_name,
            "unique_identifier": [unique_identifier],
            "start_date": start_date,
            "end_date": end_date,
            "tag": tag,
        }

        res = self.api_client.post(DELETE_CASE_URI, paylod)

        if not res["success"]:
            raise Exception(res["details"])

        return res["details"]

    def case_logs(self, page: Optional[int] = 1) -> pd.DataFrame:
        """Get already viewed case logs

        :param page: page number, defaults to 1
        :return: Case object with details
        """

        res = self.api_client.get(
            f"{CASE_LOGS_URI}?project_name={self.project_name}&page={page}"
        )

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get case logs"))

        case_logs_df = pd.DataFrame(
            res["details"]["logs"],
            columns=[
                "case_log_id",
                "case_id",
                "unique_identifier",
                "tag",
                "model_name",
                "time_taken",
                "created_at",
            ],
        )
        case_logs_df["case_log_id"] = case_logs_df["case_id"].astype(str)
        case_logs_df.drop(columns=["case_id"], inplace=True)

        return case_logs_df

    def get_viewed_case(self, case_id: str) -> Case:
        """Get already viewed case

        :param case_id: case id
        :return: Case object with details
        """

        res = self.api_client.get(
            f"{GET_VIEWED_CASE_URI}?project_name={self.project_name}&case_id={case_id}"
        )

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get viewed case"))

        data = {**res["details"], **res["details"].get("result", {})}
        data["api_client"] = self.api_client
        if self.metadata.get("modality") != "text": case = Case(**data)
        if self.metadata.get("modality") == "text": case = CaseText(**data)
        return case

    def get_notifications(self) -> pd.DataFrame:
        """get user project notifications

        :return: DataFrame
        """
        url = f"{GET_NOTIFICATIONS_URI}?project_name={self.project_name}"

        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting project notifications.")

        notifications = [
            notification
            for notification in res["details"]
            if notification.get("project_name", None)
        ]

        if not notifications:
            return "No notifications found."

        return pd.DataFrame(notifications).reindex(columns=["message", "time"])

    def clear_notifications(self) -> str:
        """clear user project notifications

        :raises Exception: _description_
        :return: response
        """
        url = f"{CLEAR_NOTIFICATIONS_URI}?project_name={self.project_name}"

        res = self.api_client.post(url)

        if not res["success"]:
            raise Exception("Error while clearing project notifications.")

        return res["details"]

    def observations(self) -> pd.DataFrame:
        """Observations

        :return: observation details dataframe
        """
        res = self.api_client.get(
            f"{GET_OBSERVATIONS_URI}?project_name={self.project_name}"
        )

        observation_df = pd.DataFrame(res.get("details"))

        if observation_df.empty:
            return observation_df

        observation_df = observation_df[
            observation_df["status"].isin(["active", "inactive"])
        ]

        if observation_df.empty:
            return observation_df

        observation_df["expression"] = observation_df["metadata"].apply(
            lambda metadata: generate_expression(metadata["expression"])
        )

        observation_df = observation_df.drop(
            columns=[
                "project_name",
                "configuration",
                "metadata",
                "updated_by",
                "created_by",
                "updated_keys",
            ],
            errors="ignore",
        )
        observation_df = observation_df.reindex(
            [
                "observation_id",
                "observation_name",
                "status",
                "statement",
                "linked_features",
                "expression",
                "created_at",
                "updated_at",
            ],
            axis=1,
        )
        observation_df.reset_index(inplace=True, drop=True)
        return observation_df

    def observation_trail(self) -> pd.DataFrame:
        """Observation Trail

        :return: observation trail details dataframe
        """
        res = self.api_client.get(
            f"{GET_OBSERVATIONS_URI}?project_name={self.project_name}"
        )

        if not res.get("details"):
            raise Exception("No observations found")

        observation_df = pd.DataFrame(res.get("details"))
        observation_df = observation_df[
            observation_df["status"].isin(["updated", "deleted"])
        ]
        if observation_df.empty:
            raise Exception("No observation trail found")
        observation_df = observation_df.rename(
            columns={
                "statement": "old_statement",
                "linked_features": "old_linked_features",
            }
        )
        observation_df["updated_keys"].replace(float("nan"), None, inplace=True)
        observation_df["updated_statement"] = observation_df["updated_keys"].apply(
            lambda data: data.get("statement") if data else None
        )

        observation_df["updated_linked_features"] = observation_df[
            "updated_keys"
        ].apply(lambda data: data.get("linked_features") if data else None)

        observation_df["old_expression"] = observation_df["metadata"].apply(
            lambda metadata: generate_expression(metadata["expression"])
        )

        observation_df["updated_expression"] = observation_df["updated_keys"].apply(
            lambda data: (
                generate_expression(data.get("metadata", {}).get("expression"))
                if data
                else None
            )
        )

        observation_df = observation_df.drop(
            columns=[
                "project_name",
                "configuration",
                "metadata",
                "updated_by",
                "created_by",
                "updated_keys",
            ],
            errors="ignore",
        )

        observation_df = observation_df.reindex(
            [
                "observation_id",
                "observation_name",
                "status",
                "old_statement",
                "updated_statement",
                "old_linked_features",
                "updated_linked_features",
                "old_expression",
                "updated_expression",
                "created_at",
                "updated_at",
            ],
            axis=1,
        )

        observation_df.reset_index(inplace=True, drop=True)
        return observation_df
    
    def duplicate_observation(self, observation_name, new_observation_name) -> str:
        if observation_name == new_observation_name:
            return "Duplicate observation name can't be same"
        url = f"{DUPLICATE_OBSERVATION_URI}?project_name={self.project_name}&observation_name={observation_name}&new_observation_name={new_observation_name}"
        res = self.api_client.post(url)

        if not res["success"]:
            return res.get("details", "Failed to clone triggers")

        return res["details"]

    def create_observation(
        self,
        observation_name: str,
        expression: str,
        statement: str,
        linked_features: List[str],
        priority: Optional[int] = 5
    ) -> str:
        """Creates New Observation

        :param observation_name: name of observation
        :param expression: expression of observation
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :param statement: statement of observation
            Eg: The building type is {BldgType}
                the content inside the curly brackets represents the feature name
        :param linked_features: linked features of observation
        :return: response
        """
        observation_params = self.api_client.get(
            f"{GET_OBSERVATION_PARAMS_URI}?project_name={self.project_name}"
        )

        Validate.string("expression", expression)

        Validate.string("statement", statement)

        Validate.value_against_list(
            "linked_feature",
            linked_features,
            list(observation_params["details"]["features"].keys()),
        )
        configuration, expression = build_expression(expression)

        validate_configuration(configuration, observation_params["details"], self.project_name, self.api_client, True)

        payload = {
            "project_name": self.project_name,
            "observation_name": observation_name,
            "status": "active",
            "configuration": configuration,
            "metadata": {"expression": expression},
            "statement": [statement],
            "linked_features": linked_features,
            "priority": priority
        }

        res = self.api_client.post(CREATE_OBSERVATION_URI, payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return "Observation Created"

    def update_observation(
        self,
        observation_id: str,
        observation_name: str,
        status: Optional[str] = None,
        expression: Optional[str] = None,
        statement: Optional[str] = None,
        linked_features: Optional[List[str]] = None,
    ) -> str:
        """Updates Observation

        :param observation_id: id of observation
        :param observation_name: name of observation
        :param status: status of observation ["active","inactive"]
        :param expression: new expression for observation, defaults to None
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :param statement: new statement for observation, defaults to None
            Eg: The building type is {BldgType}
                the content inside the curly brackets represents the feature name
        :param linked_features: new linked features for observation, defaults to None
        :return: response
        """
        if not status and not expression and not statement and not linked_features:
            raise Exception("update parameters for observation not passed")

        payload = {
            "project_name": self.project_name,
            "observation_id": observation_id,
            "observation_name": observation_name,
            "update_keys": {},
        }

        observation_params = self.api_client.get(
            f"{GET_OBSERVATION_PARAMS_URI}?project_name={self.project_name}"
        )

        if expression:
            Validate.string("expression", expression)
            configuration, expression = build_expression(expression)
            validate_configuration(configuration, observation_params["details"], self.project_name, self.api_client)
            payload["update_keys"]["configuration"] = configuration
            payload["update_keys"]["metadata"] = {"expression": expression}

        if linked_features:
            Validate.value_against_list(
                "linked_feature",
                linked_features,
                list(observation_params["details"]["features"].keys()),
            )
            payload["update_keys"]["linked_features"] = linked_features

        if statement:
            Validate.string("statement", statement)
            payload["update_keys"]["statement"] = [statement]

        if status:
            Validate.value_against_list("status", status, ["active", "inactive"])
            payload["update_keys"]["status"] = status

        res = self.api_client.post(UPDATE_OBSERVATION_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Observation Updated"

    def delete_observation(
        self,
        observation_id: str,
        observation_name: str,
    ) -> str:
        """Deletes Observation

        :param observation_id: id of observation
        :param observation_name: name of observation
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "observation_id": observation_id,
            "observation_name": observation_name,
            "delete": True,
            "update_keys": {},
        }

        res = self.api_client.post(UPDATE_OBSERVATION_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Observation Deleted"

    def policies(self) -> pd.DataFrame:
        """Policies

        :return: policy details dataframe
        """
        res = self.api_client.get(
            f"{GET_POLICIES_URI}?project_name={self.project_name}"
        )

        policy_df = pd.DataFrame(res.get("details"))

        if policy_df.empty:
            return policy_df

        policy_df = policy_df[policy_df["status"].isin(["active", "inactive"])]

        if policy_df.empty:
            return policy_df

        policy_df["expression"] = policy_df["metadata"].apply(
            lambda metadata: generate_expression(metadata["expression"])
        )

        policy_df = policy_df.drop(
            columns=[
                "project_name",
                "configuration",
                "metadata",
                "linked_features",
                "updated_by",
                "created_by",
                "updated_keys",
            ],
            errors="ignore",
        )
        policy_df = policy_df.reindex(
            [
                "policy_id",
                "policy_name",
                "status",
                "statement",
                "decision",
                "expression",
                "created_at",
                "updated_at",
            ],
            axis=1,
        )
        policy_df.reset_index(inplace=True, drop=True)
        return policy_df

    def policy_trail(self) -> pd.DataFrame:
        """Policy Trail

        :return: observation details dataframe
        """
        res = self.api_client.get(
            f"{GET_POLICIES_URI}?project_name={self.project_name}"
        )

        if not res.get("details"):
            raise Exception("No policies found")

        policy_df = pd.DataFrame(res.get("details"))
        policy_df = policy_df[policy_df["status"].isin(["updated", "deleted"])]
        if policy_df.empty:
            raise Exception("No policy trail found")
        policy_df = policy_df.rename(
            columns={
                "statement": "old_statement",
                "decision": "old_decision",
            }
        )

        policy_df["updated_keys"].replace(float("nan"), None, inplace=True)

        policy_df["updated_statement"] = policy_df["updated_keys"].apply(
            lambda data: data.get("statement") if data else None
        )

        policy_df["updated_decision"] = policy_df["updated_keys"].apply(
            lambda data: data.get("decision") if data else None
        )

        policy_df["old_expression"] = policy_df["metadata"].apply(
            lambda metadata: generate_expression(metadata["expression"])
        )

        policy_df["updated_expression"] = policy_df["updated_keys"].apply(
            lambda data: (
                generate_expression(data.get("metadata", {}).get("expression"))
                if data
                else None
            )
        )

        policy_df = policy_df.drop(
            columns=[
                "project_name",
                "configuration",
                "metadata",
                "updated_by",
                "created_by",
                "updated_keys",
                "linked_features",
            ],
            errors="ignore",
        )
        policy_df = policy_df.reindex(
            [
                "policy_id",
                "policy_name",
                "status",
                "old_statement",
                "updated_statement",
                "old_decision",
                "updated_decision",
                "old_expression",
                "updated_expression",
                "created_at",
                "updated_at",
            ],
            axis=1,
        )
        policy_df.reset_index(inplace=True, drop=True)

        return policy_df
    
    def duplicate_policy(self, policy_name, new_policy_name) -> str:
        if policy_name == new_policy_name:
            return "Duplicate observation name can't be same"
        url = f"{DUPLICATE_POLICY_URI}?project_name={self.project_name}&policy_name={policy_name}&new_policy_name={new_policy_name}"
        res = self.api_client.post(url)

        if not res["success"]:
            return res.get("details", "Failed to clone Policy")

        return res["details"]

    def create_policy(
        self,
        policy_name: str,
        expression: str,
        statement: str,
        decision: str,
        input: Optional[str] = None,
        models: Optional[list] = [],
        priority: Optional[int] = 5
    ) -> str:
        """Creates New Policy

        :param policy_name: name of policy
        :param expression: expression of policy
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :param statement: statement of policy
            Eg: The building type is {BldgType}
                the content inside the curly brackets represents the feature name
        :param decision: decision of policy
        :param input: custom input for the decision if input selected for decision of policy
        :param models: List of trained model names - The policy will only execute for the selected model. In case of empty list will execute for all models
        :return: response
        """
        configuration, expression = build_expression(expression)

        policy_params = self.api_client.get(
            f"{GET_POLICY_PARAMS_URI}?project_name={self.project_name}"
        )

        validate_configuration(configuration, policy_params["details"], self.project_name, self.api_client)

        Validate.value_against_list(
            "decision", decision, list(policy_params["details"]["decision"].values())[0]
        )

        if decision == "input":
            Validate.string("Decision input", input)

        payload = {
            "project_name": self.project_name,
            "policy_name": policy_name,
            "status": "active",
            "configuration": configuration,
            "metadata": {"expression": expression},
            "statement": [statement],
            "decision": input if decision == "input" else decision,
            "models": models,
            "priority": priority
        }

        res = self.api_client.post(CREATE_POLICY_URI, payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return "Policy Created"

    def update_policy(
        self,
        policy_id: str,
        policy_name: str,
        status: Optional[str] = None,
        expression: Optional[str] = None,
        statement: Optional[str] = None,
        decision: Optional[str] = None,
        input: Optional[str] = None,
    ) -> str:
        """Updates Policy

        :param policy_id: id of policy
        :param policy_name: name of policy
        :param status: status of policy ["active","inactive"]
        :param expression: new expression for policy, defaults to None
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :param statement: new statment for policy, defaults to None
            Eg: The building type is {BldgType}
                the content inside the curly brackets represents the feature name
        :param decision: new decision for policy, defaults to None
        :param input: custom input for the decision if input selected for decision of policy
        :return: response
        """
        if not status and not expression and not statement and not decision:
            raise Exception("update parameters for policy not passed")

        payload = {
            "project_name": self.project_name,
            "policy_id": policy_id,
            "policy_name": policy_name,
            "update_keys": {},
        }

        policy_params = self.api_client.get(
            f"{GET_POLICY_PARAMS_URI}?project_name={self.project_name}"
        )

        if expression:
            Validate.string("expression", expression)
            configuration, expression = build_expression(expression)
            validate_configuration(configuration, policy_params["details"], self.project_name, self.api_client)
            payload["update_keys"]["configuration"] = configuration
            payload["update_keys"]["metadata"] = {"expression": expression}

        if statement:
            Validate.string("statement", statement)
            payload["update_keys"]["statement"] = [statement]

        if status:
            Validate.value_against_list("status", status, ["active", "inactive"])
            payload["update_keys"]["status"] = status

        if decision:
            Validate.value_against_list(
                "decision",
                decision,
                list(policy_params["details"]["decision"].values())[0],
            )
            if decision == "input":
                Validate.string("Decision input", input)
            payload["update_keys"]["decision"] = (
                input if decision == "input" else decision
            )

        res = self.api_client.post(UPDATE_POLICY_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Policy Updated"

    def delete_policy(
        self,
        policy_id: str,
        policy_name: str,
    ) -> str:
        """Deletes Policy

        :param policy_id: id of policy
        :param policy_name: name of policy
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "policy_id": policy_id,
            "policy_name": policy_name,
            "delete": True,
            "update_keys": {},
        }

        res = self.api_client.post(UPDATE_POLICY_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Policy Deleted"

    def get_synthetic_model_params(self) -> dict:
        """get hyper parameters of synthetic models

        :return: hyper params
        """
        return self.api_client.get(GET_SYNTHETIC_MODEL_PARAMS_URI)

    def train_synthetic_model(
        self,
        model_name: str,
        data_config: Optional[SyntheticDataConfig] = {},
        hyper_params: Optional[SyntheticModelHyperParams] = {},
        instance_type: Optional[str] = "shared",
    ):
        """Train synthetic model

        :param model_name: model name ['GPT2', 'CTGAN']
        :param data_config: config for the data
            {
                "tags": List[str]
                "feature_exclude": List[str]
                "feature_include": List[str]
                "drop_duplicate_uid": bool
            },
            defaults to {}
        :param hyper_params: hyper parameters for the model. check param type and value range below,
            For GPT2 (Generative Pretrained Transformer) model - Works well on high dimensional tabular data,
            {
                "batch_size": int [1, 500] defaults to 100
                "early_stopping_patience": int [1, 100], defaults to 10
                "early_stopping_threshold": float [0.1, 100], defaults to 0.0001
                "epochs": int [1, 150], defaults to 100
                "model_type": "tabular",
                "random_state": int [1, 150], defaults to 1
                "tabular_config": "GPT2Config",
                "train_size": float [0, 0.9] defaults to 0.8
            }
            For CTGAN (Conditional Tabular GANs) model - Balances between training computation and dimensionality,
            {
                "epochs": int, [1, 150] defaults to 100
                "test_ratio": float [0, 1] defaults to 0.2
            }
            defaults to {}
        :param instance_type: type of instance to run training
            for all available instances check xai.available_synthetic_custom_servers()
            defaults to shared

        :return: response
        """

        project_config = self.config()

        if project_config == "Not Found":
            raise Exception("Upload files first")

        project_config = project_config["metadata"]

        if instance_type != "shared":
            available_servers = self.api_client.get(
                AVAILABLE_SYNTHETIC_CUSTOM_SERVERS_URI
            )["details"]
            servers = list(
                map(lambda instance: instance["instance_name"], available_servers)
            )
            Validate.value_against_list("instance_type", instance_type, servers)

        all_models_param = self.get_synthetic_model_params()

        try:
            model_params = all_models_param[model_name]
        except KeyError as e:
            availabel_models = list(all_models_param.keys())
            Validate.value_against_list("model", [model_name], availabel_models)

        # validate and prepare data config
        data_config["model_name"] = model_name

        available_tags = self.tags()
        tags = data_config.get("tags", available_tags)

        Validate.value_against_list("tag", tags, available_tags)

        feature_exclude = data_config.get(
            "feature_exclude", project_config["feature_exclude"]
        )

        Validate.value_against_list(
            "feature_exclude", feature_exclude, project_config["avaialble_options"]
        )

        feature_include = data_config.get(
            "feature_include", project_config["feature_include"]
        )

        Validate.value_against_list(
            "feature_include",
            feature_include,
            project_config["avaialble_options"],
        )

        drop_duplicate_uid = data_config.get(
            "drop_duplicate_uid", project_config["drop_duplicate_uid"]
        )

        SYNTHETIC_MODELS_DEFAULT_HYPER_PARAMS[model_name].update(hyper_params)
        hyper_params = SYNTHETIC_MODELS_DEFAULT_HYPER_PARAMS[model_name]

        # validate model hyper parameters
        for key, value in hyper_params.items():
            model_param = model_params.get(key, None)

            if model_param:
                if model_param["type"] == "input":
                    if model_param["value"] == "int":
                        if not isinstance(value, int):
                            raise Exception(f"{key} value should be integer")
                    elif model_param["value"] == "float":
                        if not isinstance(value, float):
                            raise Exception(f"{key} value should be float")

                        if value < model_param["min"] or value > model_param["max"]:
                            raise Exception(
                                f"{key} value should be between {model_param['min']} and {model_param['max']}"
                            )
                    elif model_param["type"] == "select":
                        Validate.value_against_list(
                            "value", [value], model_param["value"]
                        )

        print(f"Using data config: {json.dumps(data_config, indent=4)}")
        print(f"Using hyper params: {json.dumps(hyper_params, indent=4)}")

        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "instance_type": instance_type,
            "metadata": {
                "model_name": model_name,
                "tags": tags,
                "feature_exclude": feature_exclude,
                "feature_include": feature_include,
                "feature_actual_used": [],
                "drop_duplicate_uid": drop_duplicate_uid,
                "model_parameters": hyper_params,
            },
        }

        res = self.api_client.post(TRAIN_SYNTHETIC_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        print("Training initiated...")
        poll_events(self.api_client, self.project_name, res["event_id"])

    def remove_synthetic_model(self, model_name: str) -> str:
        """deletes synthetic model

        :param model_name: model name
        :raises ValueError: _description_
        :raises Exception: _description_
        :return: response message
        """
        models_df = self.synthetic_models()
        valid_models = models_df["model_name"].tolist()

        if model_name not in valid_models:
            raise ValueError(
                f"{model_name} is not valid. Pick a valid value from {valid_models}"
            )

        payload = {"project_name": self.project_name, "model_name": model_name}

        res = self.api_client.post(DELETE_SYNTHETIC_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return f"{model_name} deleted successfully."

    def synthetic_models(self) -> pd.DataFrame:
        """get synthetic models for the project

        :return: synthetic models
        """
        url = f"{GET_SYNTHETIC_MODELS_URI}?project_name={self.project_name}"

        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting synthetics models.")

        models = res["details"]

        models_df = pd.DataFrame(models)

        return models_df

    def synthetic_model(self, model_name: str) -> SyntheticModel:
        """get synthetic model details

        :param model_name: model name
        :raises Exception: _description_
        :return: _description_
        """
        models_df = self.synthetic_models()
        valid_models = models_df["model_name"].tolist()

        if model_name not in valid_models:
            raise ValueError(
                f"{model_name} is not valid. Pick a valid value from {valid_models}"
            )

        url = f"{GET_SYNTHETIC_MODEL_DETAILS_URI}?project_name={self.project_name}&model_name={model_name}"

        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception(res["details"])

        model_details = res["details"][0]

        metadata = model_details["metadata"]
        data_quality = model_details["results"]

        del model_details["metadata"]
        del model_details["results"]

        synthetic_model = SyntheticModel(
            **model_details,
            **data_quality,
            metadata=metadata,
            api_client=self.api_client,
            project=self,
        )

        return synthetic_model

    def synthetic_tags(self) -> pd.DataFrame:
        """get synthetic data tags of the model
        :raises Exception: _description_
        :return: list of tags
        """
        url = f"{GET_SYNTHETIC_DATA_TAGS_URI}?project_name={self.project_name}"

        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting synthetics data tags.")

        data_tags = res["details"]

        for data_tag in data_tags:
            del data_tag["metadata"]
            del data_tag["plot_data"]

        return pd.DataFrame(data_tags)

    def synthetic_tag(self, tag: str) -> SyntheticDataTag:
        """get synthetic data tag by tag name
        :param tag: tag name
        :raises Exception: _description_
        :return: tag
        """
        url = f"{GET_SYNTHETIC_DATA_TAGS_URI}?project_name={self.project_name}"

        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting synthetics data tags.")

        data_tags = res["details"]

        syn_data_tags = [
            SyntheticDataTag(
                **data_tag,
                api_client=self.api_client,
                project_name=self.project_name,
                project=self,
            )
            for data_tag in data_tags
        ]

        data_tag = next(
            (syn_data_tag for syn_data_tag in syn_data_tags if syn_data_tag.tag == tag),
            None,
        )

        if not data_tag:
            valid_tags = [syn_data_tag.tag for syn_data_tag in syn_data_tags]
            raise Exception(f"{tag} is invalid. Pick a valid value from {valid_tags}")

        return data_tag

    def synthetic_tag_datapoints(self, tag: str) -> pd.DataFrame:
        """get synthetic tag datapoints

        :param tag: tag name
        :raises Exception: _description_
        :return: datapoints
        """
        all_tags = self.all_tags()

        Validate.value_against_list(
            "tag",
            tag,
            all_tags,
        )

        res = self.api_client.base_request(
            "GET",
            f"{DOWNLOAD_SYNTHETIC_DATA_URI}?project_name={self.project_name}&tag={tag}&token={self.api_client.get_auth_token()}",
        )

        synthetic_data = pd.read_csv(io.StringIO(res.content.decode("utf-8")))

        return synthetic_data

    def remove_synthetic_tag(self, tag: str) -> str:
        """delete synthetic data tag

        :raises Exception: _description_
        :return: response messsage
        """
        all_tags = self.all_tags()

        Validate.value_against_list(
            "tag",
            tag,
            all_tags,
        )

        payload = {
            "project_name": self.project_name,
            "tag": tag,
        }

        res = self.api_client.post(DELETE_SYNTHETIC_TAG_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return f"{tag} deleted successfully."

    def get_observation_params(self) -> dict:
        """get observation parameters for the project (used in validating synthetic prompt)"""
        url = f"{GET_OBSERVATION_PARAMS_URI}?project_name={self.project_name}"

        res = self.api_client.get(url)

        return res["details"]

    def create_synthetic_prompt(self, name: str, expression: str) -> str:
        """create synthetic prompt for the project

        :param name: prompt name
        :param expression: expression of policy
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :raises Exception: _description_
        :return: response message
        """
        name = name.strip()

        if not name:
            raise Exception("name is required")

        configuration, expression = build_expression(expression)

        prompt_params = self.get_observation_params()
        validate_configuration(configuration, prompt_params, self.project_name, self.api_client)

        payload = {
            "prompt_name": name,
            "project_name": self.project_name,
            "configuration": configuration,
            "metadata": {"expression": expression},
        }

        res = self.api_client.post(CREATE_SYNTHETIC_PROMPT_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return "Synthetic prompt created successfully."

    def update_synthetic_prompt(self, prompt_id: str, status: str) -> str:
        """update synthetic prompt

        :param prompt_id: prompt id
        :param activate: True or False
        :raises Exception: _description_
        :raises Exception: _description_
        :return: response message
        """
        if status not in ["active", "inactive"]:
            raise Exception(
                "Invalid status value. Pick a valid value from ['active', 'inactive']."
            )

        payload = {
            "delete": False,
            "project_name": self.project_name,
            "prompt_id": prompt_id,
            "update_keys": {"status": status},
        }

        res = self.api_client.post(UPDATE_SYNTHETIC_PROMPT_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return "Synthetic prompt updated successfully."

    def synthetic_prompts(self) -> pd.DataFrame:
        """get synthetic prompts for the project

        :raises Exception: _description_
        :return: _description_
        """
        url = f"{GET_SYNTHETIC_PROMPT_URI}?project_name={self.project_name}"

        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception(res["details"])

        prompts = res["details"]

        return pd.DataFrame(prompts).reindex(
            columns=["prompt_id", "prompt_name", "status", "created_at", "updated_at"]
        )

    def synthetic_prompt(self, prompt_id: str) -> SyntheticPrompt:
        """get synthetic prompt by prompt id

        :raises Exception: _description_
        :return: _description_
        """
        url = f"{GET_SYNTHETIC_PROMPT_URI}?project_name={self.project_name}"
        res = self.api_client.get(url)

        if not res["success"]:
            raise Exception(res["details"])

        prompts = res["details"]

        curr_prompt = next(
            (prompt for prompt in prompts if prompt["prompt_id"] == prompt_id), None
        )

        if not curr_prompt:
            raise Exception(f"Invalid prompt_id")

        return SyntheticPrompt(**curr_prompt, api_client=self.api_client, project=self)

    def evals_tabular(self, model_name: str, tag: Optional[str] = ""):
        """get evals for ml tabular model

        :param model_name: model name
        :return: evals
        """
        url = f"{TABULAR_ML}?model_name={model_name}&project_name={self.project_name}&tag={tag}"
        res = self.api_client.post(url)
        if not res["success"]:
            raise Exception(res["message"])

        return pd.DataFrame(res["comparison_metrics"])

    def evals_dl_tabular(self, model_name: str):
        """get evals for ml tabular model

        :param model_name: model name
        :return: evals
        """
        url = f"{TABULAR_DL}?model_name={model_name}&project_name={self.project_name}"
        res = self.api_client.post(url)
        if not res["success"]:
            raise Exception(res["message"])

        return pd.DataFrame(res["comparison_metrics"])

    def evals_dl_image(self, model_name: str, unique_identifier: str):
        """get evals for ml tabular model
        :param model_name: model name
        :return: evals
        """
        url = f"{IMAGE_DL}?model_name={model_name}&project_name={self.project_name}&unique_identifier={unique_identifier}"
        res = self.api_client.post(url)
        if not res["success"]:
            raise Exception(res["message"])

        return res["attributions"]
    
    def get_feature_importance(self, model_name: str, feature_name: str, xai_method: str):
        url = f"{GET_FEATURE_IMPORTANCE_URI}?project_name={self.project_name}&model_name={model_name}&feature_name={feature_name}&xai_method={xai_method}"
        res = self.api_client.get(url)
        if not res["success"]:
            raise Exception(res["message"])
        return res.get("feature_importance", "")

    def events(
        self,
        event_id: Optional[str] = None,
        event_names: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
    ) -> List[Dict]:
        """get info about events

        :return: event details
        """
        payload = {
            "project_name": self.project_name,
            "event_id": event_id,
            "event_names": event_names,
            "status": status,
        }

        res = self.api_client.post(FETCH_EVENTS, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")

    def __print__(self) -> str:
        return f"Project(user_project_name='{self.user_project_name}', created_by='{self.created_by}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()


def generate_expression(expression):
    if not expression:
        return None
    generated_expression = ""
    for item in expression:
        if isinstance(item, str):
            generated_expression += " " + item
        else:
            generated_expression += (
                f" {item['column']} {item['expression']} {item['value']}"
            )
    return generated_expression


def build_expression(expression_string):
    condition_operators = {
        "!==": "_NOTEQ",
        "==": "_ISEQ",
        ">": "_GRT",
        "<": "_LST",
    }
    logical_operators = {"and": "_AND", "or": "_OR"}

    metadata_expression = []
    configuration = []
    string_to_be_parsed = expression_string

    matches = re.findall(r"(\w+)\s*([!=<>]+)\s*(\w+)", expression_string)

    total_opening_parentheses = re.findall(r"\(", expression_string)
    total_closing_parentheses = re.findall(r"\)", expression_string)

    if len(total_opening_parentheses) != len(total_closing_parentheses):
        raise Exception("Invalid expression, check parentheses")

    for i, match in enumerate(matches):
        column, expression, value = match
        if expression not in condition_operators.keys():
            raise Exception(f"Not a valid condition operator in {match}")

        opening_parentheses = re.findall(r"\(", string_to_be_parsed.split(column, 1)[0])
        if opening_parentheses:
            metadata_expression.extend(opening_parentheses)
            configuration.extend(opening_parentheses)

        metadata_expression.append(
            {
                "column": column,
                "value": value,
                "expression": expression,
            }
        )
        configuration.append(
            {
                "column": column,
                "value": value,
                "expression": condition_operators[expression],
            }
        )

        string_to_be_parsed = string_to_be_parsed.split(value, 1)[1]
        between_conditions_split = string_to_be_parsed.split(
            matches[i + 1][0] if i < len(matches) - 1 else None, 1
        )
        closing_parentheses = re.findall(
            r"\)",
            between_conditions_split[0] if len(between_conditions_split) > 0 else "",
        )
        if closing_parentheses:
            metadata_expression.extend(closing_parentheses)
            configuration.extend(closing_parentheses)

        if i < len(matches) - 1:
            between_conditions = between_conditions_split[0].strip()
            between_conditions = between_conditions.replace(")", "").replace("(", "")
            logical_operator = re.search(r"and|or", between_conditions)
            if not logical_operator:
                raise Exception(f"{between_conditions} is not valid logical operator")
            log_operator = logical_operator.group()
            log_operator_split = list(
                filter(
                    lambda op: op != "" and op != " ",
                    between_conditions.split(log_operator, 1),
                )
            )
            if len(log_operator_split) > 0:
                raise Exception(f"{between_conditions} is not valid logical operator")
            metadata_expression.append(log_operator)
            configuration.append(logical_operators[log_operator])

    return configuration, metadata_expression


def validate_configuration(configuration, params, project_name="", api_client=APIClient(), observations=False):
    for expression in configuration:
        if isinstance(expression, str):
            if expression not in ["(", ")", *params.get("logical_operators")]:
                raise Exception(f"{expression} not a valid logical operator")

        if isinstance(expression, dict):
            # validate column name
            Validate.value_against_list(
                "feature",
                expression.get("column"),
                list(params.get("features", {}).keys()),
            )

            # validate operator
            Validate.value_against_list(
                "condition_operator",
                expression.get("expression"),
                params.get("condition_operators"),
            )

            # validate value(s)
            expression_value = expression.get("value")
            valid_feature_values = params.get("features").get(expression.get("column"))
            if observations and isinstance(valid_feature_values, list):
                condition_operators = {
                        "_NOTEQ": "!==",
                        "_ISEQ": "==",
                        "_GRT": ">",
                        "_LST": "<",
                    }
                res = api_client.get(f"{VALIDATE_POLICY_URI}?project_name={project_name}&column1_name={expression.get('column')}&column2_name={expression.get('value')}&operation={condition_operators[expression.get('expression')]}")
                if not res.get("success"):
                    raise Exception(res.get("message"))
            if isinstance(valid_feature_values, str):
            #     if valid_feature_values == "input" and not parse_float(
            #         expression_value
            #     ):
            #         raise Exception(
            #             f"Invalid value comparison with {expression_value} for {expression.get('column')}"
            #         )
                if valid_feature_values == "datetime" and not parse_datetime(
                    expression_value
                ):
                    raise Exception(
                        f"Invalid value comparison with {expression_value} for {expression.get('column')}"
                    )

                else:
                    condition_operators = {
                        "_NOTEQ": "!==",
                        "_ISEQ": "==",
                        "_GRT": ">",
                        "_LST": "<",
                    }
                    res = api_client.get(f"{VALIDATE_POLICY_URI}?project_name={project_name}&column1_name={expression.get('column')}&column2_name={expression.get('value')}&operation={condition_operators[expression.get('expression')]}")
                    if not res.get("success"):
                        raise Exception(res.get("message"))
