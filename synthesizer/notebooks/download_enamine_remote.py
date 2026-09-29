import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE_URL = "https://enamine.net"
REAL_DB_URL = f"{BASE_URL}/compound-collections/real-compounds/real-database"

payload = json.loads(__PAYLOAD_JSON__)


def extract_real_downloads(html_text: str) -> list[dict[str, str]]:
    matches = re.findall(
        r"""href="/component/download/\?view=file&amp;f=(\d+)"[^>]*reference: '([^']+)'""",
        html_text,
    )
    downloads: list[dict[str, str]] = []
    seen: set[str] = set()
    for file_id, reference in matches:
        if reference.startswith("REAL HAC") and file_id not in seen:
            seen.add(file_id)
            downloads.append({"id": file_id, "reference": reference})
    if not downloads:
        raise RuntimeError("Could not extract REAL download links from the current Enamine page.")
    return downloads


def extract_login_form(html_text: str) -> tuple[str, dict[str, str]]:
    action_match = re.search(r'<form action="([^"]*task=user\.login[^"]*)"', html_text)
    if action_match is None:
        raise RuntimeError("Could not find the Enamine login form action.")
    hidden_inputs = dict(
        re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', html_text)
    )
    if "return" not in hidden_inputs:
        raise RuntimeError("Could not find the Enamine login return token.")
    return action_match.group(1), hidden_inputs


def filename_from_response(response: requests.Response, fallback_name: str) -> str:
    content_disposition = response.headers.get("content-disposition", "")
    filename_match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if filename_match:
        return filename_match.group(1)
    final_url_name = response.url.rstrip("/").rsplit("/", 1)[-1]
    if final_url_name and "." in final_url_name:
        return final_url_name
    return fallback_name


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    real_page = session.get(REAL_DB_URL, timeout=60)
    real_page.raise_for_status()
    downloads = extract_real_downloads(real_page.text)

    first_download_url = f"{BASE_URL}/component/download/?task=file.download&f={downloads[0]['id']}"
    login_page = session.get(first_download_url, timeout=60)
    login_page.raise_for_status()
    if "task=user.login" not in login_page.text:
        raise RuntimeError("Expected a login page before download, but Enamine returned a different page.")

    login_action, hidden_inputs = extract_login_form(login_page.text)
    login_response = session.post(
        urljoin(BASE_URL, login_action),
        data={
            "username": payload["username"],
            "password": payload["password"],
            "remember": "yes",
            **hidden_inputs,
        },
        allow_redirects=False,
        timeout=60,
    )

    if login_response.status_code not in (302, 303):
        raise RuntimeError(f"Login did not redirect as expected; status was {login_response.status_code}.")

    probe = session.get(first_download_url, allow_redirects=False, timeout=60)
    if probe.status_code in (302, 303) and "/login-page" in probe.headers.get("location", ""):
        raise RuntimeError("Login appears to have failed; download URL still redirects to the login page.")

    target_dir = Path(payload["target_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in downloads:
        file_id = item["id"]
        reference = item["reference"].replace(" ", "_").replace(",", "")
        download_url = f"{BASE_URL}/component/download/?task=file.download&f={file_id}"
        fallback_name = f"{reference}.bin"

        with session.get(download_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", ""):
                snippet = response.text[:200].replace("\n", " ")
                raise RuntimeError(f"Expected a file for {file_id}, but received HTML: {snippet}")

            filename = filename_from_response(response, fallback_name)
            destination = target_dir / filename
            if destination.exists() and destination.stat().st_size > 0:
                print(f"Skipping existing file: {destination}", flush=True)
                continue

            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        print(f"Downloaded {destination.name} -> {destination}", flush=True)

    print(f"Completed download into {target_dir}", flush=True)


if __name__ == "__main__":
    main()
