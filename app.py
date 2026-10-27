from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
STREAMLIT_USER_DIR = ROOT / ".streamlit-user"
STREAMLIT_USER_DIR.mkdir(parents=True, exist_ok=True)
if os.name == "nt":
    os.environ["USERPROFILE"] = str(STREAMLIT_USER_DIR)
else:
    os.environ["HOME"] = str(STREAMLIT_USER_DIR)

import streamlit as st

from src.preprocess import build_model_input
from src.model import MODEL_LABELS, MODEL_NAMES, MODEL_TRAINING_DEFAULTS
from src.tiff_io import read_multilayer_tiff, robust_normalize


DATA_ROOT = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"
RUNS_DIR = ROOT / "runs"
HORMONES = ("insulin", "glucagon")
LABELS = ("high", "low")


st.set_page_config(
    page_title="Secretion Classifier",
    page_icon="",
    layout="wide",
)


def tiff_files(folder: Path) -> list[Path]:
    return sorted(list(folder.rglob("*.tif")) + list(folder.rglob("*.tiff")))


def training_folder_summary(input_folder: Path) -> tuple[list[dict[str, object]], int]:
    """Summarize exactly the high/low folders understood by make_manifest.py."""
    rows: list[dict[str, object]] = []
    hormone_dirs = [
        hormone for hormone in HORMONES if (input_folder / hormone).is_dir()
    ]
    groups: list[tuple[str, str, Path]] = []
    if hormone_dirs:
        groups = [
            (hormone, label, input_folder / hormone / label)
            for hormone in hormone_dirs
            for label in LABELS
        ]
    else:
        groups = [
            ("-", label, input_folder / label)
            for label in LABELS
        ]

    total = 0
    for hormone, label, folder in groups:
        files = (
            sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
            if folder.is_dir()
            else []
        )
        total += len(files)
        rows.append(
            {
                "hormone": hormone,
                "label": label,
                "folder": str(folder),
                "folder_status": "OK" if folder.is_dir() else "MISSING",
                "tiff_count": len(files),
            }
        )
    return rows, total


def input_folder_browser(key: str, default: Path = DATA_ROOT) -> Path:
    """Browse folders on the machine running Streamlit and return the current one."""
    folder_key = f"{key}_current_folder"
    path_key = f"{key}_path"
    initial = default.expanduser().resolve()
    if not initial.is_dir():
        initial = DATA_ROOT if DATA_ROOT.is_dir() else ROOT

    if folder_key not in st.session_state:
        st.session_state[folder_key] = str(initial)
    current = Path(st.session_state[folder_key])
    if not current.is_dir():
        current = initial
        st.session_state[folder_key] = str(current)

    st.markdown("#### 학습 데이터 입력 폴더")
    st.caption(f"현재 폴더: `{current}`")

    try:
        subfolders = sorted(
            (path for path in current.iterdir() if path.is_dir()),
            key=lambda path: path.name.lower(),
        )
        browse_error = None
    except (OSError, PermissionError) as exc:
        subfolders = []
        browse_error = str(exc)

    selected = st.selectbox(
        "하위 폴더",
        subfolders,
        format_func=lambda path: path.name,
        placeholder="하위 폴더가 없습니다",
        index=None,
        key=f"{key}_subfolder",
    )
    back_col, open_col, use_col = st.columns([1, 1, 2])
    if back_col.button("⬆ 상위 폴더", key=f"{key}_parent", disabled=current.parent == current):
        st.session_state[folder_key] = str(current.parent)
        st.session_state[path_key] = str(current.parent)
        st.rerun()
    if open_col.button("열기", key=f"{key}_open", disabled=selected is None):
        st.session_state[folder_key] = str(selected)
        st.session_state[path_key] = str(selected)
        st.rerun()
    use_col.success("이 폴더가 학습 입력으로 선택되었습니다.")

    if path_key not in st.session_state:
        st.session_state[path_key] = str(current)
    entered_path = st.text_input(
        "경로로 이동",
        key=path_key,
        help="드라이브 경로 또는 네트워크 폴더 경로를 직접 입력할 수도 있습니다.",
    )
    if st.button("입력한 경로 열기", key=f"{key}_go"):
        candidate = Path(entered_path.strip()).expanduser()
        if candidate.is_dir():
            st.session_state[folder_key] = str(candidate.resolve())
            st.session_state[path_key] = str(candidate.resolve())
            st.rerun()
        else:
            st.error("존재하거나 접근 가능한 폴더 경로를 입력해 주세요.")
    if browse_error:
        st.warning(f"이 폴더의 하위 항목을 읽을 수 없습니다: {browse_error}")

    return current


def roi_preview(folder: Path, key: str) -> int:
    files = tiff_files(folder) if folder.is_dir() else []
    if not files:
        st.warning(f"No TIFF files found under {folder}")
        return int(st.session_state.get(f"{key}_roi_size", 128))

    sample = st.selectbox(
        "Preview image",
        files,
        format_func=lambda path: str(path.relative_to(folder)),
        key=f"{key}_sample",
    )
    stack = read_multilayer_tiff(sample)
    max_roi = int(min(stack.shape[-2:]))
    min_roi = min(16, max_roi)
    default_roi = min(128, max_roi)
    slider_key = f"{key}_roi_size"
    saved_roi = st.session_state.get(slider_key)
    if saved_roi is not None and not min_roi <= int(saved_roi) <= max_roi:
        del st.session_state[slider_key]
    roi_size = st.slider(
        "Maximum ROI side length (original pixels)",
        min_value=min_roi,
        max_value=max_roi,
        value=default_roi,
        step=1,
        key=slider_key,
    )
    model_input, masks = build_model_input(stack, roi_size=roi_size)
    center_row, center_col = masks["nucleus_center"]
    top, left, bottom, right = masks["roi_bounds"]
    effective_roi_size = int(masks["effective_roi_size"])

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(robust_normalize(stack[0]), cmap="gray")
    axes[0].add_patch(
        Rectangle(
            (left, top), effective_roi_size, effective_roi_size,
            fill=False, edgecolor="lime", linewidth=2
        )
    )
    axes[0].scatter([center_col], [center_row], c="red", s=25)
    axes[0].set_title("BF + selected ROI")
    axes[1].imshow(model_input[1], cmap="gray")
    axes[1].contour(masks["primary_nucleus"], levels=[0.5], colors="red", linewidths=1)
    axes[1].set_title("Nucleus ROI / primary nucleus")
    axes[2].imshow(model_input[0], cmap="gray")
    axes[2].set_title("Model BF input")
    axes[3].imshow(model_input[2], cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("Nucleus-guided cell mask")
    axes[4].imshow(model_input[3], cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("Nucleus-localized BF")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption(
        f"Detected nucleus center: ({center_col:.1f}, {center_row:.1f}); "
        f"maximum ROI={roi_size}px, automatic ROI={effective_roi_size}px; "
        f"ROI bounds: left={left}, top={top}, right={right}, bottom={bottom}"
    )
    cell_fraction = float(masks["cell"].mean())
    edge_contact = bool(
        masks["cell"][0].any() or masks["cell"][-1].any()
        or masks["cell"][:, 0].any() or masks["cell"][:, -1].any()
    )
    c1, c2 = st.columns(2)
    c1.metric("Cell area in final ROI", f"{cell_fraction:.1%}")
    c2.metric("Cell touches ROI edge", "Yes" if edge_contact else "No")
    return roi_size


def ensure_data_folders() -> None:
    for hormone in HORMONES:
        for label in LABELS:
            (DATA_ROOT / hormone / label).mkdir(parents=True, exist_ok=True)


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def format_elapsed(seconds: int) -> str:
    hours, remainder = divmod(max(seconds, 0), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_training_with_progress(
    args: list[str], total_epochs: int
) -> subprocess.CompletedProcess[str]:
    """Run training while showing epoch progress, elapsed time, ETA, and logs."""
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    env["PYTHONUNBUFFERED"] = "1"

    st.subheader("Training progress")
    progress = st.progress(0.0, text=f"Preparing training · 0/{total_epochs} epochs")
    status_col, elapsed_col, eta_col = st.columns(3)
    epoch_metric = status_col.empty()
    elapsed_metric = elapsed_col.empty()
    eta_metric = eta_col.empty()
    log_area = st.empty()

    process = subprocess.Popen(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    started = time.monotonic()
    recent_lines: deque[str] = deque(maxlen=25)
    all_lines: list[str] = []
    current_epoch = 0
    eta_text = "Calculating..."

    assert process.stdout is not None
    for raw_line in process.stdout:
        for fragment in raw_line.replace("\r", "\n").splitlines():
            line = fragment.strip()
            if not line:
                continue
            all_lines.append(line)
            recent_lines.append(line)

            batch_match = re.search(
                r"(?:epoch\s+|Batch progress epoch\s+)(\d+)/(\d+):.*?(\d+(?:\.\d+)?)%",
                line,
                re.I,
            )
            done_match = re.search(r"Epoch\s+(\d+)/(\d+)\s+done", line)
            if batch_match:
                epoch_no = int(batch_match.group(1))
                batch_fraction = float(batch_match.group(3)) / 100.0
                current_epoch = epoch_no
                overall = ((epoch_no - 1) + batch_fraction) / max(total_epochs, 1)
                progress.progress(
                    min(max(overall, 0.0), 1.0),
                    text=f"Training epoch {epoch_no}/{total_epochs} · batch {batch_fraction:.0%}",
                )
            elif done_match:
                current_epoch = int(done_match.group(1))
                overall = current_epoch / max(total_epochs, 1)
                progress.progress(
                    min(overall, 1.0),
                    text=f"Completed epoch {current_epoch}/{total_epochs}",
                )

            remaining_match = re.search(r"remaining=([^|]+)", line)
            if remaining_match:
                eta_text = remaining_match.group(1).strip()

            elapsed = int(time.monotonic() - started)
            epoch_metric.metric("Current epoch", f"{current_epoch}/{total_epochs}")
            elapsed_metric.metric("Elapsed", format_elapsed(elapsed))
            eta_metric.metric("Estimated remaining", eta_text)
            log_area.code("\n".join(recent_lines), language="text")

    return_code = process.wait()
    elapsed_metric.metric("Elapsed", format_elapsed(int(time.monotonic() - started)))
    if return_code == 0:
        progress.progress(1.0, text=f"Training completed · {total_epochs}/{total_epochs} epochs")
        epoch_metric.metric("Current epoch", f"{total_epochs}/{total_epochs}")
        eta_metric.metric("Estimated remaining", "Finished")
    else:
        progress.progress(
            min(current_epoch / max(total_epochs, 1), 1.0),
            text=f"Training stopped with exit code {return_code}",
        )

    return subprocess.CompletedProcess(
        args=args,
        returncode=return_code,
        stdout="\n".join(all_lines[-100:]),
        stderr="",
    )


def start_background_training(args: list[str]) -> int:
    """Start training independently of the Streamlit browser session."""
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    env["PYTHONUNBUFFERED"] = "1"
    popen_kwargs = {
        "cwd": ROOT,
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(args, **popen_kwargs)
    return process.pid


def run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([path for path in RUNS_DIR.iterdir() if path.is_dir()])


def prediction_checkpoint_options(target: str) -> list[dict[str, object]]:
    """Return completed checkpoints with lightweight metadata for UI selection."""
    options = []
    for run_dir in run_dirs():
        checkpoint = run_dir / "best_model.pt"
        if not checkpoint.exists() or (run_dir / "training.lock").exists():
            continue
        config = load_json(run_dir / "run_config.json")
        run_target = str(config.get("target", ""))
        if not run_target:
            if "insulin" in run_dir.name.lower():
                run_target = "insulin"
            elif "glucagon" in run_dir.name.lower():
                run_target = "glucagon"
        if run_target != target:
            continue
        model_name = str(config.get("model", "resnet18"))
        metrics = load_json(run_dir / "test_metrics.json")
        options.append({
            "path": checkpoint,
            "run": run_dir.name,
            "model": MODEL_LABELS.get(model_name, model_name),
            "auroc": metrics.get("auroc"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "complete": bool(metrics),
            "modified": checkpoint.stat().st_mtime,
        })
    return sorted(
        options,
        key=lambda item: (
            item["complete"],
            float(item["auroc"]) if item["auroc"] is not None else -1.0,
            item["modified"],
        ),
        reverse=True,
    )


def checkpoint_option_label(option: dict[str, object]) -> str:
    auroc = option["auroc"]
    score = f"AUROC {float(auroc):.3f}" if auroc is not None else "not evaluated"
    return f"{option['run']} · {option['model']} · {score}"


def completed_epochs(run_dir: Path) -> int:
    checkpoint_path = run_dir / "latest_checkpoint.pt"
    if checkpoint_path.exists():
        import torch
        return int(torch.load(checkpoint_path, map_location="cpu").get("epoch", 0))
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return 0
    matches = re.findall(
        r"Epoch\s+(\d+)/\d+\s+done",
        log_path.read_text(encoding="utf-8", errors="ignore"),
    )
    return max((int(value) for value in matches), default=0)


def read_log_tail(path: Path, max_bytes: int = 262_144) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        log_file.seek(max(size - max_bytes, 0))
        return log_file.read().decode("utf-8", errors="ignore")


@st.fragment(run_every=5)
def background_training_status() -> None:
    st.subheader("Background training status")
    candidates = [
        path for path in run_dirs()
        if (path / "train.log").exists() or (path / "training.lock").exists()
    ]
    if not candidates:
        st.caption("No background training runs are available.")
        return

    candidates.sort(
        key=lambda path: (
            (path / "training.lock").exists(),
            (path / "train.log").stat().st_mtime if (path / "train.log").exists() else 0,
        ),
        reverse=True,
    )
    names = [path.name for path in candidates]
    active_names = [
        path.name for path in candidates if (path / "training.lock").exists()
    ]
    widget_key = "background_status_run"
    if active_names and st.session_state.get(widget_key) not in active_names:
        st.session_state[widget_key] = active_names[0]
    elif st.session_state.get(widget_key) not in names:
        st.session_state[widget_key] = names[0]
    selected_name = st.selectbox("Run to monitor", names, key=widget_key)
    run_dir = RUNS_DIR / selected_name
    log_path = run_dir / "train.log"
    log_text = read_log_tail(log_path)

    batch_matches = re.findall(
        r"Batch progress epoch\s+(\d+)/(\d+):\s+(\d+)/(\d+)\s+\(([\d.]+)%\)",
        log_text,
    )
    done_matches = re.findall(r"Epoch\s+(\d+)/(\d+)\s+done", log_text)
    current_epoch = total_epochs = current_batch = total_batches = 0
    batch_percent = 0.0
    if batch_matches:
        epoch_text, total_text, batch_text, batches_text, percent_text = batch_matches[-1]
        current_epoch, total_epochs = int(epoch_text), int(total_text)
        current_batch, total_batches = int(batch_text), int(batches_text)
        batch_percent = float(percent_text) / 100.0
    elif done_matches:
        epoch_text, total_text = done_matches[-1]
        current_epoch, total_epochs = int(epoch_text), int(total_text)
        batch_percent = 1.0

    is_running = (run_dir / "training.lock").exists()
    is_complete = (run_dir / "test_metrics.json").exists()
    if is_running:
        status = "Running"
    elif is_complete:
        status = "Completed"
    else:
        status = "Stopped / waiting to resume"

    overall = 0.0
    if total_epochs:
        overall = ((max(current_epoch, 1) - 1) + batch_percent) / total_epochs
        if is_complete:
            overall = 1.0
    st.progress(
        min(max(overall, 0.0), 1.0),
        text=f"{status} · {overall:.1%} overall",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status)
    c2.metric("Epoch", f"{current_epoch}/{total_epochs}" if total_epochs else "Preparing")
    c3.metric(
        "Batch",
        f"{current_batch}/{total_batches}" if total_batches else "—",
    )
    checkpoint_epoch = 0
    history_path = run_dir / "history.csv"
    if history_path.exists():
        try:
            history = pd.read_csv(history_path, usecols=["epoch"])
            if not history.empty:
                checkpoint_epoch = int(history["epoch"].iloc[-1])
        except (OSError, ValueError, pd.errors.EmptyDataError):
            pass
    c4.metric("Saved checkpoint", f"Epoch {checkpoint_epoch}" if checkpoint_epoch else "None")

    if log_path.exists():
        seconds_since_update = max(0, int(time.time() - log_path.stat().st_mtime))
        st.caption(
            f"Auto-refreshes every 5 seconds · log updated "
            f"{format_elapsed(seconds_since_update)} ago · {log_path.relative_to(ROOT)}"
        )
    useful_lines = [
        line.strip()
        for line in log_text.replace("\r", "\n").splitlines()
        if line.strip() and (
            "Batch progress" in line
            or re.search(r"Epoch\s+\d+/\d+\s+done", line)
            or line.startswith(("Started", "Resumed", "Saved", "Finished", "Device", "GPU", "test"))
        )
    ]
    if useful_lines:
        st.code("\n".join(useful_lines[-20:]), language="text")


def run_selectbox(label: str, key: str) -> Path | None:
    runs = run_dirs()
    if not runs:
        st.info("아직 학습 결과 폴더가 없습니다.")
        return None
    names = [path.name for path in runs]
    shared_key = "selected_run_name"
    selected_run = st.session_state.get(shared_key)
    if selected_run not in names:
        selected_run = names[0]
        st.session_state[shared_key] = selected_run

    # Page-local widgets share the last selected run through one session value.
    st.session_state[key] = selected_run

    def remember_selection() -> None:
        st.session_state[shared_key] = st.session_state[key]

    selected = st.selectbox(
        label,
        names,
        key=key,
        on_change=remember_selection,
    )
    return RUNS_DIR / selected


def show_command_result(result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode == 0:
        st.success("완료되었습니다.")
    else:
        st.error(f"실패했습니다. exit code: {result.returncode}")
    if result.stdout:
        st.code(result.stdout, language="text")
    if result.stderr:
        st.code(result.stderr, language="text")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def data_page() -> None:
    st.header("Data")
    ensure_data_folders()

    input_folder = input_folder_browser("data_input")
    rows, total_tiffs = training_folder_summary(input_folder)
    st.subheader("선택한 학습 데이터 요약")
    path_col, count_col = st.columns([3, 1])
    path_col.metric("현재 적용된 입력 폴더", str(input_folder))
    count_col.metric("인식된 TIFF", f"{total_tiffs:,}개")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    missing = [row for row in rows if row["folder_status"] == "MISSING"]
    empty = [row for row in rows if row["folder_status"] == "OK" and row["tiff_count"] == 0]
    if missing:
        st.error("필요한 high/low 폴더가 없습니다. 위 표에서 MISSING 경로를 확인해 주세요.")
    elif empty:
        st.warning("일부 high/low 폴더에 TIFF 파일이 없습니다.")
    elif total_tiffs:
        st.success(f"선택한 폴더에서 학습용 TIFF {total_tiffs:,}개를 정상적으로 인식했습니다.")

    st.subheader("Preview and manifest")
    roi_size = roi_preview(input_folder, "data_preview")
    st.session_state["training_input_folder"] = str(input_folder)
    st.session_state["training_roi_size"] = roi_size
    c1, c2, c3 = st.columns(3)
    val_fraction = c1.number_input("Validation fraction", 0.01, 0.8, 1 / 6, 0.01)
    test_fraction = c2.number_input("Test fraction", 0.01, 0.8, 0.15, 0.01)
    seed = c3.number_input("Seed", 0, 999999, 17, 1)

    if st.button("Create manifest", type="primary"):
        result = run_command(
            [
                sys.executable,
                "scripts/make_manifest.py",
                "--data-root",
                str(input_folder),
                "--out",
                str(MANIFEST.relative_to(ROOT)),
                "--val-fraction",
                str(val_fraction),
                "--test-fraction",
                str(test_fraction),
                "--seed",
                str(seed),
            ]
        )
        show_command_result(result)

    if MANIFEST.exists():
        df = pd.read_csv(MANIFEST)
        st.caption(str(MANIFEST.relative_to(ROOT)))
        st.dataframe(df.head(200), use_container_width=True, hide_index=True)


def training_page() -> None:
    st.header("Training")
    if not MANIFEST.exists():
        st.warning("먼저 Data 화면에서 manifest를 만들어 주세요.")
        st.info("Choose an input folder below; a manifest will be created when training starts.")

    st.subheader("Input and ROI")
    input_folder = input_folder_browser(
        "training_input",
        Path(st.session_state.get("training_input_folder", DATA_ROOT)),
    )
    roi_size = roi_preview(input_folder, "training_preview")

    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target", ("both",) + HORMONES)
    model_choice = c2.selectbox(
        "Model",
        ("compare_all",) + MODEL_NAMES,
        format_func=lambda value: (
            "Compare all 4 models" if value == "compare_all" else MODEL_LABELS[value]
        ),
    )
    pretrained = c3.checkbox("Use ImageNet pretrained weights", value=True)
    compare_all = model_choice == "compare_all"
    if compare_all:
        run_count = len(MODEL_NAMES) * (2 if target == "both" else 1)
        st.info(
            f"동일한 설정으로 4개 모델을 순차 학습합니다. 총 {run_count}개 run이 생성됩니다."
        )

    st.subheader("New training run")
    suggested_key = f"suggested_run_name_{target}"
    if suggested_key not in st.session_state:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = "secretion" if target == "both" else target
        st.session_state[suggested_key] = f"{prefix}_roi_{timestamp}"
    output_name = st.text_input(
        "New run folder name",
        value=st.session_state[suggested_key],
        help="A new folder will be created under runs/. Use a unique name for every training.",
        key=f"run_folder_name_{target}",
    ).strip()
    invalid_name = (
        not output_name
        or output_name in {".", ".."}
        or bool(re.search(r'[<>:"/\\|?*]', output_name))
    )
    output_dir = RUNS_DIR / output_name if output_name else RUNS_DIR
    if compare_all and output_name:
        output_dirs = [
            RUNS_DIR / (
                f"{output_name}_{model_name}_{hormone}"
                if target == "both" else f"{output_name}_{model_name}"
            )
            for model_name in MODEL_NAMES
            for hormone in (("insulin", "glucagon") if target == "both" else (target,))
        ]
    elif target == "both" and output_name:
        output_dirs = [
            RUNS_DIR / f"{output_name}_insulin", RUNS_DIR / f"{output_name}_glucagon"
        ]
    else:
        output_dirs = [output_dir]
    run_exists = any(path.exists() for path in output_dirs)
    if invalid_name:
        st.error("Enter a folder name without / \\ : * ? \" < > | characters.")
    elif run_exists:
        st.error("One or more output folders already exist. Enter a different run name.")
    else:
        destinations = ", ".join(str(path.relative_to(ROOT)) for path in output_dirs)
        st.success(f"Training results will be saved to {destinations}")

    st.subheader("Model-specific training schedule")
    schedule_mode = st.radio(
        "Schedule",
        ("Automatic recommended", "Custom per model"),
        horizontal=True,
        help="Automatic mode applies a different maximum epoch, patience, learning rate, and batch size to each architecture.",
    )
    automatic_schedule = schedule_mode == "Automatic recommended"
    scheduled_models = MODEL_NAMES if compare_all else (model_choice,)
    model_schedules: dict[str, dict[str, float | int]] = {}
    for model_name in scheduled_models:
        defaults = MODEL_TRAINING_DEFAULTS[model_name]
        if automatic_schedule:
            model_schedules[model_name] = dict(defaults)
        else:
            st.markdown(f"**{MODEL_LABELS[model_name]}**")
            e_col, p_col, lr_col, b_col = st.columns(4)
            model_schedules[model_name] = {
                "max_epochs": int(e_col.number_input(
                    "Maximum epochs", 1, 1000, int(defaults["max_epochs"]), 1,
                    key=f"schedule_epochs_{model_name}",
                )),
                "patience": int(p_col.number_input(
                    "Early stopping patience", 0, 100, int(defaults["patience"]), 1,
                    key=f"schedule_patience_{model_name}",
                )),
                "learning_rate": float(lr_col.number_input(
                    "Learning rate", 0.000001, 1.0, float(defaults["learning_rate"]),
                    format="%.6f", key=f"schedule_lr_{model_name}",
                )),
                "batch_size": int(b_col.number_input(
                    "Batch size", 1, 512, int(defaults["batch_size"]), 1,
                    key=f"schedule_batch_{model_name}",
                )),
            }
    schedule_rows = [
        {
            "model": MODEL_LABELS[name],
            "max_epochs": values["max_epochs"],
            "patience": values["patience"],
            "learning_rate": values["learning_rate"],
            "batch_size": values["batch_size"],
        }
        for name, values in model_schedules.items()
    ]
    st.dataframe(pd.DataFrame(schedule_rows), width="stretch", hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    image_size = c1.number_input("Image size", 32, 1024, 224, 16)
    preview_images = c2.number_input("Preview images", 0, 500, 50, 1)
    weight_decay = c3.number_input("Weight decay", 0.0, 1.0, 0.0001, format="%.6f")
    num_workers = c4.number_input("Workers", 0, 32, 4, 1)
    min_delta = st.number_input(
        "Early stopping minimum AUROC improvement",
        0.0, 0.1, 0.001, 0.001, format="%.3f",
        help="An AUROC increase smaller than this value does not reset patience.",
    )

    script = "train_compare.py" if compare_all else (
        "train_both.py" if target == "both" else "train.py"
    )
    cmd = [sys.executable, script, "--manifest", str(MANIFEST.relative_to(ROOT))]
    if compare_all:
        cmd.extend([
            "--output-prefix", str(output_dir.relative_to(ROOT)),
            "--target", target,
        ])
    elif target == "both":
        cmd.extend(["--output-prefix", str(output_dir.relative_to(ROOT))])
    else:
        cmd.extend(["--target", target, "--output-dir", str(output_dir.relative_to(ROOT))])
    cmd.extend([
        "--image-size",
        str(image_size),
        "--roi-size",
        str(roi_size),
        "--weight-decay",
        str(weight_decay),
        "--num-workers",
        str(num_workers),
        "--preview-images",
        str(preview_images),
        "--early-stopping-min-delta",
        str(min_delta),
    ])
    if compare_all:
        for name, values in model_schedules.items():
            option = name.replace("_", "-")
            cmd.extend([
                f"--{option}-epochs", str(values["max_epochs"]),
                f"--{option}-patience", str(values["patience"]),
                f"--{option}-lr", str(values["learning_rate"]),
                f"--{option}-batch-size", str(values["batch_size"]),
            ])
    else:
        values = model_schedules[model_choice]
        cmd.extend([
            "--model", model_choice,
            "--epochs", str(values["max_epochs"]),
            "--early-stopping-patience", str(values["patience"]),
            "--lr", str(values["learning_rate"]),
            "--batch-size", str(values["batch_size"]),
        ])
    if pretrained:
        cmd.append("--pretrained")

    st.code(" ".join(cmd), language="powershell")
    if st.button(
        "Start training",
        type="primary",
        disabled=invalid_name or run_exists,
    ):
        existing_outputs = [path for directory in output_dirs for path in (
            directory / "train.log",
            directory / "best_model.pt",
            directory / "history.csv",
            directory / "training.lock",
        )]
        if any(path.exists() for path in existing_outputs):
            st.error(
                f"Run folder '{output_name}' is already in use or contains a previous run. "
                "Choose a new Run folder name to prevent concurrent training and overwritten logs."
            )
            st.stop()
        manifest_result = run_command(
            [
                sys.executable,
                "scripts/make_manifest.py",
                "--data-root",
                str(input_folder),
                "--out",
                str(MANIFEST),
            ]
        )
        if manifest_result.returncode != 0:
            st.error("Could not create a manifest from the selected input folder.")
            show_command_result(manifest_result)
            st.stop()
        for directory in output_dirs:
            directory.mkdir(parents=True, exist_ok=False)
        with st.spinner("학습 중입니다. 데이터가 많으면 시간이 걸립니다."):
            pid = start_background_training(cmd)
        st.success(
            f"Training started in the background (PID {pid}). "
            "You may close or refresh this page without stopping it."
        )
        if compare_all:
            st.info(
                "네 모델이 순차 실행됩니다. 아래 Background training status에서 "
                "현재 실행 중인 run과 각 모델의 로그를 확인할 수 있습니다."
            )
        elif target == "both":
            st.info(
                f"Insulin runs first, followed by glucagon. Logs: "
                f"runs/{output_name}_insulin/train.log and "
                f"runs/{output_name}_glucagon/train.log"
            )
        else:
            st.info(f"Progress is recorded in runs/{output_name}/train.log")

    st.subheader("Resume an interrupted run")
    resumable_runs = [
        path for path in run_dirs()
        if ((path / "latest_checkpoint.pt").exists() or (path / "best_model.pt").exists())
        and not (path / "training.lock").exists()
        and not (path / "test_metrics.json").exists()
    ]
    if resumable_runs:
        resume_dir = st.selectbox(
            "Run checkpoint",
            resumable_runs,
            format_func=lambda path: path.name,
            key="resume_training_run",
        )
        import torch
        checkpoint_path = resume_dir / "latest_checkpoint.pt"
        if not checkpoint_path.exists():
            checkpoint_path = resume_dir / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {})
        completed_epoch = completed_epochs(resume_dir)
        resume_epochs = st.number_input(
            "Total epochs after resume",
            min_value=completed_epoch + 1,
            max_value=1000,
            value=max(int(checkpoint_args.get("epochs", completed_epoch + 1)), completed_epoch + 1),
            step=1,
        )
        st.caption(
            f"Completed through epoch {completed_epoch}. Resume starts at epoch "
            f"{completed_epoch + 1}."
        )
        if st.button("Resume training", type="primary"):
            resume_cmd = [
                sys.executable, "train.py",
                "--manifest", str(checkpoint_args["manifest"]),
                "--target", str(checkpoint_args["target"]),
                "--output-dir", str(resume_dir.relative_to(ROOT)),
                "--epochs", str(int(resume_epochs)),
                "--batch-size", str(checkpoint_args["batch_size"]),
                "--image-size", str(checkpoint_args["image_size"]),
                "--roi-size", str(checkpoint_args["roi_size"]),
                "--lr", str(checkpoint_args["lr"]),
                "--weight-decay", str(checkpoint_args["weight_decay"]),
                "--num-workers", str(checkpoint_args["num_workers"]),
                "--preview-images", str(checkpoint_args["preview_images"]),
                "--threshold-mode", str(checkpoint_args["threshold_mode"]),
                "--min-high-recall", str(checkpoint_args["min_high_recall"]),
                "--model", str(checkpoint_args.get("model", "resnet18")),
                "--early-stopping-patience", str(
                    checkpoint_args.get("early_stopping_patience", 0)
                ),
                "--early-stopping-min-delta", str(
                    checkpoint_args.get("early_stopping_min_delta", 0.001)
                ),
                "--resume",
            ]
            if checkpoint_args.get("pretrained"):
                resume_cmd.append("--pretrained")
            pid = start_background_training(resume_cmd)
            st.success(f"Training resumed in the background (PID {pid}).")
    else:
        st.caption("No stopped run with a resume checkpoint is available yet.")

    background_training_status()


def evaluation_page() -> None:
    st.header("Evaluation")
    comparison_rows = []
    for candidate in run_dirs():
        metrics = load_json(candidate / "test_metrics.json")
        checkpoint_args = load_json(candidate / "run_config.json")
        training_summary = load_json(candidate / "training_summary.json")
        checkpoint_path = candidate / "best_model.pt"
        if not metrics:
            continue
        checkpoint = {}
        if not checkpoint_args and checkpoint_path.exists():
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                checkpoint_args = checkpoint.get("args", {})
            except Exception:
                checkpoint_args = {}
        comparison_rows.append({
            "run": candidate.name,
            "target": checkpoint_args.get("target", checkpoint.get("target", "-")),
            "model": MODEL_LABELS.get(
                checkpoint_args.get("model", "resnet18"),
                checkpoint_args.get("model", "resnet18"),
            ),
            "best_epoch": training_summary.get("best_epoch", "-"),
            "completed_epoch": training_summary.get("completed_epoch", "-"),
            "stop_reason": training_summary.get("stop_reason", "legacy_run"),
            **metrics,
        })
    if comparison_rows:
        st.subheader("Model comparison")
        comparison_df = pd.DataFrame(comparison_rows)
        sort_columns = [column for column in ("target", "auroc") if column in comparison_df]
        if sort_columns:
            comparison_df = comparison_df.sort_values(
                sort_columns, ascending=[True, False][:len(sort_columns)]
            )
        st.dataframe(comparison_df, width="stretch", hide_index=True)

    run_dir = run_selectbox("Run", "eval_run")
    if run_dir is None:
        return

    metrics = load_json(run_dir / "test_metrics.json")
    if metrics:
        cols = st.columns(len(metrics))
        for col, (key, value) in zip(cols, metrics.items()):
            if isinstance(value, float):
                col.metric(key, f"{value:.4f}")
            else:
                col.metric(key, value)

    image_names = [
        "training_curves.png",
        "test_confusion_matrix.png",
        "test_roc_curve.png",
        "test_precision_recall_curve.png",
    ]
    images = [run_dir / name for name in image_names if (run_dir / name).exists()]
    if images:
        cols = st.columns(2)
        for index, image_path in enumerate(images):
            cols[index % 2].image(str(image_path), caption=image_path.name, use_container_width=True)

    pred_path = run_dir / "test_predictions.csv"
    if pred_path.exists():
        st.subheader("Test predictions")
        st.dataframe(pd.read_csv(pred_path), use_container_width=True, hide_index=True)


def qc_preview_page() -> None:
    st.header("QC / Model Focus Preview")
    run_dir = run_selectbox("Run", "qc_run")
    if run_dir is None:
        return

    examples_dir = run_dir / "preprocess_attention_examples"
    examples_csv = examples_dir / "examples.csv"
    if not examples_csv.exists():
        st.warning("이 run에는 모델 집중 영역 예시가 없습니다. 학습 시 --preview-images 값을 1 이상으로 주세요.")
        return

    df = pd.read_csv(examples_csv)
    filter_mode = st.radio(
        "Filter",
        ["All", "Correct only", "Wrong only"],
        horizontal=True,
    )
    if filter_mode == "Correct only":
        df = df[df["label"] == df["predicted_label"]]
    elif filter_mode == "Wrong only":
        df = df[df["label"] != df["predicted_label"]]

    st.dataframe(df, use_container_width=True, hide_index=True)
    cols = st.columns(2)
    for index, row in df.iterrows():
        image_path = Path(row["example_image"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        if image_path.exists():
            caption = (
                f"{Path(row['path']).name} | true={row['label']} "
                f"pred={row['predicted_label']} prob_high={row['prob_high']:.3f}"
            )
            cols[index % 2].image(str(image_path), caption=caption, use_container_width=True)


def prediction_page() -> None:
    st.header("Prediction")
    st.subheader("1. Prediction input")
    prediction_default = ROOT / "data" / "new_tiffs"
    prediction_default.mkdir(parents=True, exist_ok=True)
    input_mode = st.radio(
        "Input type",
        ("High / Low folders", "Unlabeled folder", "Single TIFF"),
        horizontal=True,
    )
    input_args: list[str] = []
    input_path = ""
    if input_mode == "High / Low folders":
        st.caption(
            "Insulin과 Glucagon 각각에 대해 실제 High/Low 이미지 폴더를 선택하세요. "
            "각 이미지는 해당 hormone 모델로만 평가됩니다."
        )
        input_count = 0
        labeled_counts: dict[str, int] = {}
        target_tabs = st.tabs(["Insulin test set", "Glucagon test set"])
        for tab, target in zip(target_tabs, HORMONES):
            with tab:
                high_col, low_col = st.columns(2)
                for column, label in zip((high_col, low_col), LABELS):
                    default_folder = prediction_default / target / label
                    default_folder.mkdir(parents=True, exist_ok=True)
                    with column:
                        st.markdown(f"##### {target.capitalize()} {label.capitalize()}")
                        folder = input_folder_browser(
                            f"prediction_{target}_{label}", default_folder
                        )
                        files = tiff_files(folder) if folder.is_dir() else []
                        count_key = f"{target}_{label}"
                        labeled_counts[count_key] = len(files)
                        input_count += len(files)
                        st.metric(
                            f"{target.capitalize()} {label.capitalize()} TIFFs",
                            f"{len(files):,}",
                        )
                        if files:
                            option = f"--{target}-{label}-input"
                            input_args.extend([option, str(folder)])
        input_path = "4 hormone/label folders"
        summary_rows = [
            {
                "target": target,
                "high_tiffs": labeled_counts.get(f"{target}_high", 0),
                "low_tiffs": labeled_counts.get(f"{target}_low", 0),
                "total": (
                    labeled_counts.get(f"{target}_high", 0)
                    + labeled_counts.get(f"{target}_low", 0)
                ),
            }
            for target in HORMONES
        ]
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
        st.metric("Total labeled TIFF files", f"{input_count:,}")
    else:
        prediction_folder = input_folder_browser(
            "prediction_input",
            prediction_default,
        )
        available_tiffs = (
            tiff_files(prediction_folder) if prediction_folder.is_dir() else []
        )
        if input_mode == "Single TIFF":
            selected_tiff = st.selectbox(
                "TIFF file",
                available_tiffs,
                format_func=lambda path: str(path.relative_to(prediction_folder)),
                index=None,
                placeholder="Select a TIFF file",
            )
            input_path = str(selected_tiff) if selected_tiff else ""
            input_count = 1 if selected_tiff else 0
        else:
            input_path = str(prediction_folder)
            input_count = len(available_tiffs)
        if input_path:
            input_args.extend(["--input", input_path])
        input_col, count_col = st.columns([3, 1])
        input_col.metric("Selected input", input_path or "No TIFF selected")
        count_col.metric("TIFF files", f"{input_count:,}")
    if input_count == 0:
        st.warning("선택한 입력에서 TIFF 파일을 찾지 못했습니다.")

    st.subheader("2. Model checkpoints")
    checkpoint_paths: dict[str, str] = {}
    checkpoint_cols = st.columns(2)
    for column, target in zip(checkpoint_cols, HORMONES):
        with column:
            enabled = st.checkbox(
                f"Use {target} model", value=True, key=f"prediction_use_{target}"
            )
            options = prediction_checkpoint_options(target)
            selected = st.selectbox(
                f"{target.capitalize()} checkpoint",
                options,
                format_func=checkpoint_option_label,
                index=0 if options else None,
                placeholder=f"No {target} checkpoint found",
                disabled=not enabled,
                key=f"prediction_checkpoint_{target}",
            )
            if enabled and selected:
                checkpoint_paths[target] = str(selected["path"])
                accuracy = selected["balanced_accuracy"]
                score_text = (
                    f"Balanced accuracy: {float(accuracy):.3f}"
                    if accuracy is not None else "Not evaluated"
                )
                st.caption(f"Model: {selected['model']} · {score_text}")
                st.code(str(selected["path"]), language="text")

    with st.expander("Advanced: enter checkpoint paths manually"):
        for target in HORMONES:
            manual = st.text_input(
                f"Manual {target} checkpoint path",
                value="",
                key=f"prediction_manual_{target}",
            ).strip()
            if manual:
                checkpoint_paths[target] = manual

    st.subheader("3. Output and run")
    output_csv = st.text_input("Output CSV", "predictions_both.csv")
    insulin_ckpt = checkpoint_paths.get("insulin", "")
    glucagon_ckpt = checkpoint_paths.get("glucagon", "")

    cmd = [
        sys.executable,
        "predict.py",
        *input_args,
        "--out", output_csv,
    ]
    if insulin_ckpt:
        cmd.extend(["--insulin-checkpoint", insulin_ckpt])
    if glucagon_ckpt:
        cmd.extend(["--glucagon-checkpoint", glucagon_ckpt])

    invalid_checkpoints = [
        path for path in (insulin_ckpt, glucagon_ckpt)
        if path and not Path(path).is_file()
    ]
    missing_target_checkpoints = [
        target for target in HORMONES
        if any(arg.startswith(f"--{target}-") for arg in input_args)
        and target not in checkpoint_paths
    ]
    can_run = (
        bool(input_args) and input_count > 0
        and bool(checkpoint_paths) and not invalid_checkpoints
        and not missing_target_checkpoints
    )
    if invalid_checkpoints:
        st.error("존재하지 않는 체크포인트: " + ", ".join(invalid_checkpoints))
    if not checkpoint_paths:
        st.warning("Insulin 또는 glucagon 체크포인트를 하나 이상 선택해 주세요.")
    if missing_target_checkpoints:
        st.error(
            "선택한 테스트 폴더에 필요한 체크포인트가 없습니다: "
            + ", ".join(missing_target_checkpoints)
        )
    st.code(" ".join(cmd), language="powershell")
    if st.button("Run prediction", type="primary", disabled=not can_run):
        result = run_command(cmd)
        show_command_result(result)

    out_path = ROOT / output_csv
    if out_path.exists():
        try:
            prediction_df = pd.read_csv(out_path)
        except pd.errors.EmptyDataError:
            st.info("기존 출력 CSV가 비어 있습니다. Prediction을 다시 실행해 주세요.")
        else:
            if "true_label" in prediction_df.columns:
                st.subheader("Prediction accuracy on labeled images")
                result_columns = st.columns(2)
                for column, target in zip(result_columns, HORMONES):
                    predicted_column = f"{target}_predicted_label"
                    if predicted_column not in prediction_df.columns:
                        continue
                    with column:
                        target_df = (
                            prediction_df[
                                prediction_df["target"].astype(str).str.lower() == target
                            ]
                            if "target" in prediction_df.columns else prediction_df
                        )
                        if target_df.empty:
                            st.caption("No labeled images for this target.")
                            continue
                        true_labels = target_df["true_label"].astype(str).str.lower()
                        predicted_labels = (
                            target_df[predicted_column].astype(str).str.lower()
                        )
                        accuracy = float((true_labels == predicted_labels).mean())
                        high_rows = true_labels == "high"
                        low_rows = true_labels == "low"
                        high_recall = float(
                            (predicted_labels[high_rows] == "high").mean()
                        ) if high_rows.any() else float("nan")
                        low_recall = float(
                            (predicted_labels[low_rows] == "low").mean()
                        ) if low_rows.any() else float("nan")
                        st.markdown(f"#### {target.capitalize()}")
                        metric_cols = st.columns(3)
                        metric_cols[0].metric("Accuracy", f"{accuracy:.1%}")
                        metric_cols[1].metric(
                            "High recall",
                            f"{high_recall:.1%}" if pd.notna(high_recall) else "N/A",
                        )
                        metric_cols[2].metric(
                            "Low recall",
                            f"{low_recall:.1%}" if pd.notna(low_recall) else "N/A",
                        )
                        confusion = pd.crosstab(
                            true_labels,
                            predicted_labels,
                            rownames=["True label"],
                            colnames=["Predicted label"],
                        ).reindex(index=LABELS, columns=LABELS, fill_value=0)
                        st.dataframe(confusion, width="stretch")
            st.dataframe(prediction_df, width="stretch", hide_index=True)


def settings_page() -> None:
    st.header("Settings")
    st.write("Project root")
    st.code(str(ROOT), language="text")
    st.write("Python")
    st.code(sys.executable, language="text")


def main() -> None:
    st.sidebar.title("Secretion Classifier")
    page = st.sidebar.radio(
        "Navigation",
        ["Data", "Training", "Evaluation", "QC / Preview", "Prediction", "Settings"],
    )

    if page == "Data":
        data_page()
    elif page == "Training":
        training_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "QC / Preview":
        qc_preview_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Settings":
        settings_page()


if __name__ == "__main__":
    main()
