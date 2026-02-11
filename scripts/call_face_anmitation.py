import subprocess
import uuid
import sys
import argparse
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

def _format_exit_code(code: int) -> str:
    if code == 4294967295:
        return "-1"
    return str(code)

def _run_capture_text(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

def _list_wsl_distros() -> List[str]:
    try:
        result = _run_capture_text(["wsl.exe", "-l", "-q"])
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

def _build_wsl_prefix(preferred_distro: Optional[str]) -> Tuple[List[str], Optional[str], Optional[str]]:
    if preferred_distro:
        distros = _list_wsl_distros()
        if preferred_distro in distros:
            return ["wsl.exe", "-d", preferred_distro, "--"], preferred_distro, None
        if distros:
            return ["wsl.exe", "--"], None, f"WSL distro not found: {preferred_distro}. Using default distro."
        return ["wsl.exe", "--"], None, f"WSL distro not found: {preferred_distro}. WSL distros list is empty."
    return ["wsl.exe", "--"], None, None

def _looks_like_windows_path(path: str) -> bool:
    p = (path or "").strip()
    if not p:
        return False
    if len(p) >= 2 and p[1] == ":":
        return True
    if p.startswith("\\\\") or p.startswith("\\"):
        return True
    return False

def _manual_windows_to_wsl_mnt(path: str) -> str:
    raw = (path or "").strip()
    if not raw:
        return raw
    normalized = raw.replace("/", "\\")
    if len(normalized) >= 3 and normalized[1] == ":" and (normalized[2] == "\\" or normalized[2] == "/"):
        drive = normalized[0].lower()
        rest = normalized[2:].lstrip("\\").replace("\\", "/")
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    if len(normalized) >= 2 and normalized[1] == ":":
        drive = normalized[0].lower()
        rest = normalized[2:].lstrip("\\").replace("\\", "/")
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    return raw

def _to_wsl_path(path: str, *, preferred_distro: Optional[str]) -> str:
    raw = (path or "").strip()
    if not _looks_like_windows_path(raw):
        return raw
    prefix, chosen, _ = _build_wsl_prefix(preferred_distro)
    cmd = prefix + ["wslpath", "-a", "-u", raw]
    result = _run_capture_text(cmd)
    if result.returncode != 0:
        return _manual_windows_to_wsl_mnt(raw)
    converted = (result.stdout or "").strip()
    return converted or _manual_windows_to_wsl_mnt(raw)

# 写入临时文件并返回 Windows 磁盘完整路径
def save_face_anim_uploads(
    *,
    task_dir: str,
    video_bytes: Union[bytes, bytearray, memoryview],
    audio_bytes: Union[bytes, bytearray, memoryview],
    video_filename: str = "input_video.mp4",
    audio_filename: str = "input_audio.wav",
) -> Tuple[str, str]:
    resolved_task_dir = os.path.abspath(str(task_dir or "").strip() or ".")
    os.makedirs(resolved_task_dir, exist_ok=True)

    resolved_video_filename = str(video_filename or "input_video.mp4").strip() or "input_video.mp4"
    resolved_audio_filename = str(audio_filename or "input_audio.wav").strip() or "input_audio.wav"

    video_path = os.path.join(resolved_task_dir, resolved_video_filename)
    audio_path = os.path.join(resolved_task_dir, resolved_audio_filename)

    with open(video_path, "wb") as f:
        f.write(bytes(video_bytes))
    with open(audio_path, "wb") as f:
        f.write(bytes(audio_bytes))

    return video_path, audio_path

# 调用 WSL bash 脚本
def run_face_anim_wsl(
    *,
    script_path: str,
    speaker_id: str,
    audio_path: str,
    task_id: Optional[str] = None,
    distro: Optional[str] = None,
    script_args: Optional[List[str]] = None,
    dry_run: bool = False,
    timeout_s: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    resolved_task_id = (task_id or "").strip() or str(uuid.uuid4())
    wsl_prefix, chosen_distro, warn = _build_wsl_prefix(distro)

    resolved_script_path = _to_wsl_path(script_path, preferred_distro=chosen_distro or distro)
    resolved_audio_path = _to_wsl_path(audio_path, preferred_distro=chosen_distro or distro)

    resolved_args: List[str]
    if script_args is not None:
        resolved_args = [str(x) for x in script_args]
    else:
        resolved_args = [str(speaker_id), str(resolved_audio_path), str(resolved_task_id)]

    normalized_args: List[str] = []
    for arg in resolved_args:
        normalized_args.append(_to_wsl_path(str(arg), preferred_distro=chosen_distro or distro))

    cmd = wsl_prefix + ["bash", resolved_script_path] + normalized_args

    if verbose:
        if warn:
            sys.stderr.write(warn + "\n")
        sys.stdout.write(f"Executing command: {' '.join(cmd)}\n")

    if dry_run:
        return {
            "task_id": resolved_task_id,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "command": cmd,
            "used_distro": chosen_distro,
            "warning": warn,
        }

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_s,
        )
        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""

        if verbose:
            sys.stdout.write("--- WSL STDOUT ---\n")
            if stdout_text:
                sys.stdout.write(stdout_text + ("\n" if not stdout_text.endswith("\n") else ""))
            if stderr_text:
                sys.stderr.write("--- WSL STDERR ---\n")
                sys.stderr.write(stderr_text + ("\n" if not stderr_text.endswith("\n") else ""))
            sys.stdout.write(f"--- Return Code: {_format_exit_code(result.returncode)} ---\n")
            sys.stdout.write("Execution successful!\n" if result.returncode == 0 else "Execution failed!\n")

        return {
            "task_id": resolved_task_id,
            "returncode": int(result.returncode),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "command": cmd,
            "used_distro": chosen_distro,
            "warning": warn,
        }
    except FileNotFoundError:
        msg = "Error: wsl.exe not found. Please run this on Windows with WSL installed."
        if verbose:
            sys.stderr.write(msg + "\n")
        return {
            "task_id": resolved_task_id,
            "returncode": -1,
            "stdout": "",
            "stderr": msg,
            "command": cmd,
            "used_distro": chosen_distro,
            "warning": warn,
        }
    except subprocess.TimeoutExpired:
        msg = f"Error: WSL command timed out after {timeout_s}s."
        if verbose:
            sys.stderr.write(msg + "\n")
        return {
            "task_id": resolved_task_id,
            "returncode": -1,
            "stdout": "",
            "stderr": msg,
            "command": cmd,
            "used_distro": chosen_distro,
            "warning": warn,
        }
    except Exception as e:
        msg = f"An error occurred: {e}"
        if verbose:
            sys.stderr.write(msg + "\n")
        return {
            "task_id": resolved_task_id,
            "returncode": -1,
            "stdout": "",
            "stderr": msg,
            "command": cmd,
            "used_distro": chosen_distro,
            "warning": warn,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker-id", default="m001_trump")
    parser.add_argument("--audio-path", default="/home/sdk/input.wav")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--distro", default=os.environ.get("WSL_DISTRO_NAME", ""))
    parser.add_argument("--script-path", default="/mnt/d/Downloads/psfa-2022-main/scripts/test_gpu.sh")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_face_anim_wsl(
        script_path=args.script_path,
        speaker_id=args.speaker_id,
        audio_path=args.audio_path,
        task_id=(args.task_id or None),
        distro=(args.distro or None),
        dry_run=bool(args.dry_run),
        verbose=True,
    )

    sys.stdout.write(
        "Final Result -> "
        f"Task ID: {result.get('task_id')}, Return Code: {_format_exit_code(int(result.get('returncode', -1)))}\n"
    )
