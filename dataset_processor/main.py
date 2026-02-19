# f:\python\Musetalk\dataset_processor\main.py
import argparse
import datetime
import os
import sys
from typing import Any, Dict


def _bootstrap_paths() -> None:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(pkg_dir)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    if repo_root not in sys.path:
        sys.path.insert(1, repo_root)
    os.chdir(repo_root)


def _set_windows_dpi_awareness() -> None:
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        return


class ConsoleProgress:
    def __init__(self) -> None:
        self._steps: Dict[str, Dict[str, Any]] = {}

    def add_step(self, name: str, status: str, percent: int) -> None:
        self._steps[name] = {"status": status, "percent": percent}
        print(f"[STEP] {name}: {status} {percent}%")

    def update_step(self, name: str, status: str, percent: int) -> None:
        self._steps[name] = {"status": status, "percent": percent}
        print(f"[STEP] {name}: {status} {percent}%")

    def set_current_task(self, text: str) -> None:
        print(f"[TASK] {text}")

    def set_total_progress(self, percent: int) -> None:
        print(f"[TOTAL] {percent}%")


def _console_log(level: str, message: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {level}: {message}")


def _load_config(config_path: str) -> Dict[str, Any]:
    from utils.config_manager import ConfigManager

    return ConfigManager(config_path).load()


def _run_hdtf(cfg: Dict[str, Any], run_integrity: bool, run_preprocess: bool, run_quality: bool) -> bool:
    from core.hdtf_processor import HDTFProcessor

    source_dir = cfg.get("source_dir") or cfg.get("data_dir")
    output_dir = cfg.get("output_dir") or os.path.join(str(source_dir), "processed")
    steps = cfg.get("steps", {})
    batch_size = int(cfg.get("batch_size", 8))

    if not source_dir:
        _console_log("ERROR", "HDTF source_dir 未配置")
        return False

    processor = HDTFProcessor(ConsoleProgress(), _console_log, cfg)
    ok = True
    if run_integrity:
        processor.check_integrity(str(source_dir))
    if run_preprocess:
        ok = processor.preprocess(str(source_dir), str(output_dir), steps, batch_size) and ok
    if run_quality:
        issues = processor.validate_quality(os.path.join(str(output_dir), "meta"))
        issue_count = sum(len(v) for v in issues.values())
        _console_log("INFO", f"HDTF quality issues: {issue_count}")
    return ok


def _run_vfhq(cfg: Dict[str, Any], run_integrity: bool, run_preprocess: bool, run_quality: bool) -> bool:
    from core.vfhq_processor import VFHQProcessor

    base_path = cfg.get("base_path")
    output_dir = cfg.get("output_dir") or os.path.join(str(base_path), "processed")
    steps = cfg.get("steps", {})
    batch_size = int(cfg.get("batch_size", 4))
    strategy = str(cfg.get("strategy", "complete_only"))

    if not base_path:
        _console_log("ERROR", "VFHQ base_path 未配置")
        return False

    processor = VFHQProcessor(ConsoleProgress(), _console_log, cfg)
    ok = True
    if run_integrity:
        processor.check_integrity(str(base_path))
    if run_preprocess:
        ok = processor.preprocess(str(base_path), str(output_dir), steps, batch_size, strategy) and ok
    if run_quality:
        issues = processor.validate_quality(os.path.join(str(output_dir), "meta"))
        issue_count = sum(len(v) for v in issues.values())
        _console_log("INFO", f"VFHQ quality issues: {issue_count}")
    return ok


def _run_batch(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config)
    except Exception as exc:
        _console_log("ERROR", f"加载配置失败: {exc}")
        return 2

    if args.hdtf_source:
        cfg.setdefault("hdtf", {})["source_dir"] = args.hdtf_source
    if args.hdtf_output:
        cfg.setdefault("hdtf", {})["output_dir"] = args.hdtf_output
    if args.vfhq_base:
        cfg.setdefault("vfhq", {})["base_path"] = args.vfhq_base
    if args.vfhq_output:
        cfg.setdefault("vfhq", {})["output_dir"] = args.vfhq_output

    hdtf_steps = cfg.setdefault("hdtf", {}).setdefault("steps", {})
    if args.hdtf_disable_convert:
        hdtf_steps["convert"] = False
    if args.hdtf_disable_segment:
        hdtf_steps["segment"] = False
    if args.hdtf_disable_audio:
        hdtf_steps["audio"] = False
    if args.hdtf_disable_analyze:
        hdtf_steps["analyze"] = False
    if args.hdtf_disable_metadata:
        hdtf_steps["metadata"] = False
    if args.hdtf_disable_splits:
        hdtf_steps["splits"] = False

    vfhq_steps = cfg.setdefault("vfhq", {}).setdefault("steps", {})
    if args.vfhq_disable_convert_to_video:
        vfhq_steps["convert_to_video"] = False

    run_integrity = not args.skip_integrity
    run_preprocess = not args.skip_preprocess
    run_quality = not args.skip_quality

    targets = ["hdtf", "vfhq"] if args.dataset == "all" else [args.dataset]
    overall_ok = True

    for target in targets:
        _console_log("INFO", f"开始处理: {target.upper()}")
        if target == "hdtf":
            overall_ok = _run_hdtf(cfg.get("hdtf", {}), run_integrity, run_preprocess, run_quality) and overall_ok
        elif target == "vfhq":
            overall_ok = _run_vfhq(cfg.get("vfhq", {}), run_integrity, run_preprocess, run_quality) and overall_ok
        _console_log("INFO", f"结束处理: {target.upper()}")

    return 0 if overall_ok else 1


def _run_gui() -> None:
    _set_windows_dpi_awareness()
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

    import tkinter as tk
    from gui.main_window import MainWindow

    root = tk.Tk()
    root.title("MuseTalk 数据集处理工具")
    root.geometry("1100x780")
    root.minsize(980, 700)
    app = MainWindow(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuseTalk dataset processor")
    parser.add_argument("--batch", action="store_true", help="Run in non-GUI batch mode")
    parser.add_argument("--config", default="./dataset_processor/config/default_config.yaml", help="Config yaml path")
    parser.add_argument("--dataset", choices=["all", "hdtf", "vfhq"], default="all", help="Target dataset")
    parser.add_argument("--hdtf-source", default="", help="Override HDTF source_dir")
    parser.add_argument("--hdtf-output", default="", help="Override HDTF output_dir")
    parser.add_argument("--hdtf-disable-convert", action="store_true", help="Disable HDTF convert step")
    parser.add_argument("--hdtf-disable-segment", action="store_true", help="Disable HDTF segment step")
    parser.add_argument("--hdtf-disable-audio", action="store_true", help="Disable HDTF audio step")
    parser.add_argument("--hdtf-disable-analyze", action="store_true", help="Disable HDTF analyze step")
    parser.add_argument("--hdtf-disable-metadata", action="store_true", help="Disable HDTF metadata step")
    parser.add_argument("--hdtf-disable-splits", action="store_true", help="Disable HDTF splits hint step")
    parser.add_argument("--vfhq-base", default="", help="Override VFHQ base_path")
    parser.add_argument("--vfhq-output", default="", help="Override VFHQ output_dir")
    parser.add_argument("--vfhq-disable-convert-to-video", action="store_true", help="Disable VFHQ convert_to_video step")
    parser.add_argument("--skip-integrity", action="store_true", help="Skip integrity check")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocess")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality validation")
    return parser


def main() -> None:
    _bootstrap_paths()
    args = _build_arg_parser().parse_args()
    if args.batch:
        raise SystemExit(_run_batch(args))
    _run_gui()


if __name__ == "__main__":
    main()
