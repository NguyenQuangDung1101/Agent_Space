from datetime import datetime
from typing import Any


def run(arguments: dict[str, Any] | None = None) -> dict[str, str]:
    now = datetime.now().astimezone()

    return {
        "datetime": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": now.tzname() or "local",
    }


if __name__ == "__main__":
    print(run())