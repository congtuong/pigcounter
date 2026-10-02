"""Download a recorded Dahua interval through the vendor NetSDK Python package."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_local_time(value: str) -> datetime:
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"time must use YYYY-MM-DD HH:MM:SS: {value}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Dahua NVR footage to a DAV file using NetSDK."
    )
    parser.add_argument("--host", default=os.getenv("DAHUA_HOST"), required=not os.getenv("DAHUA_HOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DAHUA_NETSDK_PORT", "37777")))
    parser.add_argument("--username", default=os.getenv("DAHUA_USERNAME", "admin"))
    parser.add_argument(
        "--password-env",
        default="DAHUA_PASSWORD",
        help="environment variable containing the device password (default: DAHUA_PASSWORD)",
    )
    parser.add_argument("--channel", type=int, required=True, help="1-based recorder channel")
    parser.add_argument("--start", type=parse_local_time, required=True, help="device-local start time")
    parser.add_argument("--end", type=parse_local_time, required=True, help="device-local end time")
    parser.add_argument("--output", type=Path, required=True, help="destination .dav file")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sdk_imports():
    try:
        from ctypes import c_int, sizeof

        from NetSDK.NetSDK import NetClient
        from NetSDK.SDK_Callback import fTimeDownLoadPosCallBack
        from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE, EM_QUERY_RECORD_TYPE, EM_USEDEV_MODE
        from NetSDK.SDK_Struct import (
            NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
            NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
            NET_TIME,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Dahua NetSDK Python package is missing. Install the Windows x64 NetSDK "
            "Python wheel and its DLLs into this Python environment."
        ) from exc
    return (
        c_int,
        sizeof,
        NetClient,
        fTimeDownLoadPosCallBack,
        EM_LOGIN_SPAC_CAP_TYPE,
        EM_QUERY_RECORD_TYPE,
        EM_USEDEV_MODE,
        NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
        NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
        NET_TIME,
    )


def _sdk_time(value: datetime, net_time_type):
    result = net_time_type()
    result.dwYear = value.year
    result.dwMonth = value.month
    result.dwDay = value.day
    result.dwHour = value.hour
    result.dwMinute = value.minute
    result.dwSecond = value.second
    return result


def download_recording(args: argparse.Namespace) -> dict[str, object]:
    if os.name != "nt":
        raise RuntimeError("The configured Dahua NetSDK wheel is Windows-only.")
    if args.start >= args.end:
        raise ValueError("--start must be earlier than --end")
    if args.channel < 1:
        raise ValueError("--channel is 1-based and must be at least 1")
    if args.port < 1 or args.port > 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be positive")
    if args.output.suffix.lower() != ".dav":
        raise ValueError("--output must have a .dav extension")

    password = os.getenv(args.password_env)
    if not password:
        raise RuntimeError(f"Set the {args.password_env} environment variable first.")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.stem}.partial.dav")
    if partial.exists():
        partial.unlink()

    (
        c_int,
        sizeof,
        NetClient,
        progress_callback_type,
        login_type,
        record_type,
        device_mode,
        login_input_type,
        login_output_type,
        net_time_type,
    ) = _sdk_imports()

    NetClient()
    if not NetClient.InitEx():
        raise RuntimeError("Dahua NetSDK initialization failed.")

    login_id = 0
    download_id = 0
    completed = threading.Event()
    progress = {"total": 0, "received": 0}

    @progress_callback_type
    def on_download_progress(_handle, total_size, received_size, _index, _record_info, _user_data):
        total = int(total_size)
        received = int(received_size)
        progress["total"] = total
        progress["received"] = received
        if received == 0xFFFFFFFF or (total > 0 and received >= total):
            completed.set()

    try:
        login_input = login_input_type()
        login_input.dwSize = sizeof(login_input)
        login_input.szIP = args.host.encode("ascii")
        login_input.nPort = args.port
        login_input.szUserName = args.username.encode("ascii")
        login_input.szPassword = password.encode("ascii")
        login_input.emSpecCap = int(login_type.TCP)
        login_output = login_output_type()
        login_output.dwSize = sizeof(login_output)

        login_id, device_info, login_error = NetClient.LoginWithHighLevelSecurity(
            login_input, login_output
        )
        if not login_id:
            raise RuntimeError(f"Dahua NetSDK login failed: {login_error or 'unknown error'}")

        channel_count = int(device_info.nChanNum)
        if args.channel > channel_count:
            raise ValueError(
                f"channel {args.channel} is outside this recorder's {channel_count} channels"
            )

        # Dahua SDK channel IDs are zero-based; user-facing channel numbers are one-based.
        channel_id = args.channel - 1
        stream_mode_ok = NetClient.SetDeviceMode(
            login_id, int(device_mode.RECORD_STREAM_TYPE), c_int(0)
        )
        if not stream_mode_ok:
            raise RuntimeError("Dahua NetSDK could not set the recording stream search mode.")

        start_time = _sdk_time(args.start, net_time_type)
        end_time = _sdk_time(args.end, net_time_type)
        search_ok, file_count, _files = NetClient.QueryRecordFile(
            login_id,
            channel_id,
            int(record_type.ALL),
            start_time,
            end_time,
            "",
            10000,
            True,
        )
        if not search_ok or file_count < 1:
            raise RuntimeError("Dahua NetSDK found no recordings in the requested interval.")

        download_id = NetClient.DownloadByTimeEx(
            login_id,
            channel_id,
            int(record_type.ALL),
            start_time,
            end_time,
            str(partial),
            on_download_progress,
            0,
            None,
            0,
        )
        if not download_id:
            raise RuntimeError(f"Dahua NetSDK could not start the download: {NetClient.GetLastErrorMessage()}")
        if not completed.wait(args.timeout_seconds):
            raise TimeoutError(
                f"Dahua NetSDK download did not finish within {args.timeout_seconds} seconds."
            )
    finally:
        if download_id:
            NetClient.StopDownload(download_id)
        if login_id:
            NetClient.Logout(login_id)
        NetClient.Cleanup()

    if not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError("Dahua NetSDK reported completion but produced an empty file.")
    if output.exists():
        output.unlink()
    partial.replace(output)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"Downloaded {output}, but OpenCV is not installed to verify decoding.") from exc

    capture = cv2.VideoCapture(str(output))
    try:
        readable, frame = capture.read()
        metadata = {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "height": int(frame.shape[0]) if readable else 0,
            "width": int(frame.shape[1]) if readable else 0,
        }
    finally:
        capture.release()
    if not readable:
        raise RuntimeError(f"Downloaded {output}, but OpenCV could not decode its first frame.")

    return {
        "channel": args.channel,
        "downloaded_bytes": output.stat().st_size,
        "end": args.end.strftime(TIME_FORMAT),
        "output": str(output),
        "record_files_found": file_count,
        "start": args.start.strftime(TIME_FORMAT),
        "video": metadata,
    }


def main() -> int:
    args = parse_args()
    try:
        result = download_recording(args)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
