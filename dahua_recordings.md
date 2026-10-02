# Download Dahua recordings with NetSDK

`scripts/download_dahua_recording.py` logs in to a Dahua recorder with the vendor
NetSDK Python package, searches the requested device-local time interval, and
downloads the recording as a `.dav` file. The script checks that OpenCV can
decode the first frame before reporting success.

Dahua documents and distributes NetSDK through its [developer portal](https://depp.dahuasecurity.com/knowledgeBase)
and [software download center](https://www.dahuasecurity.com/download-center/softwares?menu_id=472).
The Python binding and its native DLLs must be installed in the Python environment
used for the backend; keep the wheel and DLLs out of Git. The working build was
verified on Windows x64 with Python 3.12. The NetSDK import is optional for the
rest of the application. Confirm the package's provenance and compatibility
with the target recorder before deploying it to another host.

Set credentials in the shell rather than putting them in a command or config:

```powershell
$env:DAHUA_HOST = "<recorder-host>"
$env:DAHUA_NETSDK_PORT = "37777"
$env:DAHUA_USERNAME = "<sdk-user>"
$env:DAHUA_PASSWORD = "<password>"
```

From the `backend_end` directory, request a short interval. Channel numbering
is 1-based in the CLI; the adapter maps it to the SDK's 0-based channel ID.
Times must use the recorder's local clock:

```powershell
& .\.venv\Scripts\python.exe .\scripts\download_dahua_recording.py `
  --channel 1 `
  --start "2026-10-01 20:30:00" `
  --end "2026-10-01 20:31:00" `
  --output ".\runtime\recordings\camera-001-20261001-203000.dav"
```

The SDK port is separate from the web and RTSP ports. On the recorder checked
during development, the web UI used port 55331, RTSP used port 554, and NetSDK
used port 37777. Allow the SDK port from the downloader host in the recorder
and firewall configuration.

## Cross-file-interval download behavior

`QueryRecordFile` reports the recorder's source recording segments that overlap
the requested interval. `DownloadByTimeEx` takes one output path and, on the
recorder tested here, writes all matching source segments in that interval into
one `.dav` output. Thus `record_files_found` can be greater than one while the
downloaded output is still a single file.

Verified on recorder channel 1 for **2026-09-30 12:50–13:10** (device-local
time), which crosses the recorder's 13:00 source-file boundary:

- `QueryRecordFile` found 2 source recording files.
- `DownloadByTimeEx` produced one `.dav` file (319,257,348 bytes).
- OpenCV reported 25 FPS and 30,000 frames, or 1,200 seconds (20 minutes).
- The downloaded video opened and decoded at least its first frame at 1920×1080.

This confirms cross-file-interval merging for the tested recorder/firmware. It
is not a guarantee for every Dahua model, firmware, or recording type. Test the
target device before relying on it; keep `record_files_found` as the source
segment count and treat the requested output path as one downloaded artifact.

## Validation limits

The helper waits for the SDK download-complete callback, rejects empty output,
and verifies that OpenCV can decode the first frame. That does not prove every
frame or timestamp in the file is continuous. The 20-minute test separately
reported 30,000 frames at 25 FPS, but an archive may still contain gaps from
recording/storage interruptions. For stronger acceptance checks, decode the
whole file and compare its observed duration/frame count with the requested
interval and the recorder's segment metadata.

A scheduled downloader should run after each recording interval has closed,
store a manifest for retry/deduplication, and then pass completed files to the
existing video-file tracking path.
