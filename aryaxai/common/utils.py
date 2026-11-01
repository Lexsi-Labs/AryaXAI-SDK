from datetime import datetime
from typing import Callable, Optional
from aryaxai.client.client import APIClient
from IPython.display import display, HTML

from aryaxai.common.xai_uris import POLL_EVENTS


def parse_float(s):
    """parse float from string, return None if not possible

    :param s: string to parse
    :return: float or None
    """
    try:
        return float(s)
    except ValueError:
        return None


def parse_datetime(s, format="%Y-%m-%d %H:%M:%S"):
    """Parse datetime from string, return None if not possible

    :param s: string to parse
    :param format: format string for datetime parsing
    :return: datetime or None
    """
    try:
        return datetime.strptime(s, format)
    except ValueError:
        return None


def pretty_date(date: str) -> str:
    """return date in format dd-mm-YYYY HH:MM:SS

    :param date: str datetime
    :return: pretty datetime
    """
    try:
        datetime_obj = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        try:
            datetime_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                datetime_obj = datetime.fromisoformat(date)
            except ValueError:
                print("Date format invalid.")
                return date

    return datetime_obj.strftime("%d-%m-%Y %H:%M:%S")


def poll_events(
    api_client: APIClient,
    project_name: str,
    event_id: str,
    handle_failed_event: Optional[Callable] = None,
    progress_message: str = "progress",
):
    last_message = ""
    log_length = 0
    progress = 0

    for event in api_client.stream(
        f"{POLL_EVENTS}?project_name={project_name}&event_id={event_id}"
    ):
        details = event.get("details")

        if not event.get("success"):
            raise Exception(details)
        if details.get("event_logs"):
            print(details.get("event_logs")[log_length:])
            log_length = len(details.get("event_logs"))
        if details.get("message") != last_message:
            last_message = details.get("message")
            print(f"{details.get('message')}")
        if details.get("progress"):
            if details.get("progress") != progress:
                progress = details.get("progress")
                print(f"{progress_message}: {progress}%")
            # display(HTML(f"<progress style='width:100%' value='{progress}' max='100'></progress>"))
        if details.get("status") == "failed":
            if handle_failed_event:
                handle_failed_event()
            raise Exception(details.get("message"))
