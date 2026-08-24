import pandas as pd
from fastapi.testclient import TestClient

import main


def test_upload_replaces_stale_session_cookie(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        [
            {
                "Case ID": "1",
                "Start Timestamp": "2026/01/01 00:00:00.000",
                "Complete Timestamp": "2026/01/01 00:01:00.000",
                "Activity": "Start",
                "Resource": "Alice",
                "Role": "Operator",
            }
        ]
    )
    monkeypatch.setattr(main, "FILES_DIR", str(tmp_path))
    monkeypatch.setattr(main, "process_file", lambda _: frame)
    monkeypatch.setattr(main, "describe_df", lambda _: ("", [], [], []))
    main.sessions.clear()

    with TestClient(main.app) as client:
        client.cookies.set(
            "session_id",
            "stale-session",
            domain="testserver.local",
            path="/",
        )

        response = client.post(
            "/upload?panel_id=panel-1",
            files={"file": ("event.csv", b"ignored", "text/csv")},
        )

        assert response.status_code == 200
        session_cookie_headers = [
            value
            for value in response.headers.get_list("set-cookie")
            if value.startswith("session_id=")
        ]
        assert len(session_cookie_headers) == 1
        assert "Max-Age=0" not in session_cookie_headers[0]
        new_session_id = client.cookies.get("session_id")
        assert new_session_id is not None
        assert new_session_id != "stale-session"
        assert new_session_id in main.sessions
        assert "panel-1" in main.sessions[new_session_id]["panels"]

    main.sessions.clear()
