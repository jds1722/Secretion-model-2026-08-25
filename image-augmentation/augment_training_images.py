from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


IMAGE_EXTENSIONS = {".tif", ".tiff"}
OME_PIXEL_KEYS = (
    "PhysicalSizeX", "PhysicalSizeXUnit", "PhysicalSizeY", "PhysicalSizeYUnit",
    "PhysicalSizeZ", "PhysicalSizeZUnit", "TimeIncrement", "TimeIncrementUnit",
)


@dataclass(frozen=True)
class Augmentation:
    suffix: str
    quarter_turns: int = 0
    flip_lr: bool = False
    angle_degrees: float = 0.0
    shift_y: int = 0
    shift_x: int = 0


@dataclass(frozen=True)
class AugmentationOptions:
    include_quarter_turns: bool = True
    include_horizontal_flips: bool = True
    small_angles: tuple[float, ...] = ()
    shift_pixels: int = 0
    fill_mode: str = "border_median"
    fill_value: float = 0


@dataclass(frozen=True)
class AugmentResult:
    source_path: str
    augmentation: str
    destination_path: str
    status: str
    note: str = ""
    axes: str = ""
    mask_source_path: str = ""
    mask_destination_path: str = ""


@dataclass
class TiffPayload:
    data: np.ndarray
    axes: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
    write_options: dict[str, Any] = field(default_factory=dict)


def build_augmentations(options: AugmentationOptions) -> list[Augmentation]:
    augmentations = [Augmentation("orig")]
    if options.include_quarter_turns:
        augmentations.extend(
            Augmentation(f"rot{degrees}", quarter_turns=turns)
            for turns, degrees in ((1, 90), (2, 180), (3, 270))
        )
    if options.include_horizontal_flips:
        bases = list(augmentations)
        augmentations.extend(
            Augmentation(
                f"{base.suffix}_flip_lr",
                quarter_turns=base.quarter_turns,
                flip_lr=True,
            )
            for base in bases
        )
    for angle in options.small_angles:
        if not math.isfinite(angle) or angle == 0:
            continue
        label = f"{angle:+g}".replace("+", "p").replace("-", "m").replace(".", "p")
        augmentations.append(Augmentation(f"rot_{label}deg", angle_degrees=angle))
    if options.shift_pixels:
        distance = abs(int(options.shift_pixels))
        augmentations.extend(
            (
                Augmentation(f"shift_up_{distance}px", shift_y=-distance),
                Augmentation(f"shift_down_{distance}px", shift_y=distance),
                Augmentation(f"shift_left_{distance}px", shift_x=-distance),
                Augmentation(f"shift_right_{distance}px", shift_x=distance),
            )
        )
    return augmentations


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_paths(input_path: Path, output_dir: Path, mask_input_path: Path | None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_dir() and output_dir.resolve() == input_path.resolve():
        raise ValueError("Output folder must not be the same as the input folder.")
    if mask_input_path is not None:
        if not mask_input_path.exists():
            raise FileNotFoundError(f"Mask path does not exist: {mask_input_path}")
        if input_path.is_file() != mask_input_path.is_file():
            raise ValueError("Image and mask inputs must both be files or both be folders.")


def iter_tiff_files(
    input_path: Path,
    recursive: bool,
    excluded_directories: tuple[Path, ...] = (),
) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Input file is not a TIF/TIFF image: {input_path}")
        return [input_path]
    pattern = "**/*" if recursive else "*"
    excluded = tuple(directory.resolve() for directory in excluded_directories)
    return sorted(
        path for path in input_path.glob(pattern)
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(_is_within(path, directory) for directory in excluded)
        )
    )


def _tag_value(page: tifffile.TiffPage, name: str) -> Any | None:
    tag = page.tags.get(name)
    return None if tag is None else tag.value


def _first_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _extract_ome_metadata(ome_xml: str | None) -> dict[str, Any]:
    if not ome_xml:
        return {}
    try:
        parsed = tifffile.xml2dict(ome_xml)
        image = _first_mapping(_first_mapping(parsed.get("OME", {})).get("Image", {}))
        pixels = _first_mapping(image.get("Pixels", {}))
        metadata = {key: pixels[key] for key in OME_PIXEL_KEYS if key in pixels}
        channels = pixels.get("Channel", [])
        if isinstance(channels, dict):
            channels = [channels]
        names = [channel.get("Name") for channel in channels if isinstance(channel, dict)]
        if names and all(name is not None for name in names):
            metadata["Channel"] = {"Name": names}
        return metadata
    except Exception:
        return {}


def read_tiff(path: Path) -> TiffPayload:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes.upper()
        if len(axes) != data.ndim or axes.count("Y") != 1 or axes.count("X") != 1:
            raise ValueError(
                f"Cannot identify one X and one Y axis (axes={axes!r}, shape={data.shape})."
            )
        page = series.pages[0]
        write_options: dict[str, Any] = {}
        for tag_name, option_name in (
            ("PhotometricInterpretation", "photometric"),
            ("PlanarConfiguration", "planarconfig"),
            ("ResolutionUnit", "resolutionunit"),
            ("Software", "software"),
            ("DateTime", "datetime"),
            ("ICCProfile", "iccprofile"),
        ):
            value = _tag_value(page, tag_name)
            if value is not None:
                write_options[option_name] = value
        x_resolution = _tag_value(page, "XResolution")
        y_resolution = _tag_value(page, "YResolution")
        if x_resolution is not None and y_resolution is not None:
            write_options["resolution"] = (x_resolution, y_resolution)
        colormap = _tag_value(page, "ColorMap")
        if colormap is not None:
            write_options["colormap"] = colormap

        if tif.is_ome:
            kind = "ome"
            metadata = _extract_ome_metadata(tif.ome_metadata)
        elif tif.is_imagej:
            kind = "imagej"
            metadata = dict(tif.imagej_metadata or {})
        else:
            kind = "generic"
            metadata = {}
            if not tif.is_shaped and page.description:
                write_options["description"] = page.description
    metadata["axes"] = axes
    return TiffPayload(data=data, axes=axes, kind=kind, metadata=metadata, write_options=write_options)


def spatial_axes(axes: str) -> tuple[int, int]:
    return axes.index("Y"), axes.index("X")


def _filled_canvas(
    moved: np.ndarray,
    fill_mode: str,
    fill_value: float,
) -> np.ndarray:
    if fill_mode == "constant":
        return np.full(moved.shape, fill_value, dtype=moved.dtype)
    if fill_mode != "border_median":
        raise ValueError(f"Unsupported fill mode: {fill_mode}")

    # Calculate one background value for every non-spatial plane. The nearest
    # quantile method selects an observed border pixel instead of averaging two
    # values, so no new intensity value is introduced.
    border = np.concatenate(
        (
            moved[..., 0, :],
            moved[..., -1, :],
            moved[..., 1:-1, 0],
            moved[..., 1:-1, -1],
        ),
        axis=-1,
    )
    background = np.quantile(border, 0.5, axis=-1, method="nearest")
    canvas = np.empty_like(moved)
    canvas[...] = np.asarray(background, dtype=moved.dtype)[..., None, None]
    return canvas


def _rotate_nearest(
    image: np.ndarray,
    angle_degrees: float,
    y_axis: int,
    x_axis: int,
    fill_value: float,
    fill_mode: str,
) -> np.ndarray:
    """Rotate without creating interpolated intensities.

    Nearest-neighbour sampling preserves the input value set, but it can duplicate
    or omit pixels and create stair-stepped boundaries.
    """
    moved = np.moveaxis(image, (y_axis, x_axis), (-2, -1))
    height, width = moved.shape[-2:]
    yy, xx = np.indices((height, width), dtype=np.float64)
    center_y, center_x = (height - 1) / 2, (width - 1) / 2
    radians = math.radians(angle_degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    x0, y0 = xx - center_x, yy - center_y
    source_x = np.rint(cosine * x0 + sine * y0 + center_x).astype(np.int64)
    source_y = np.rint(-sine * x0 + cosine * y0 + center_y).astype(np.int64)
    valid = (
        (source_x >= 0) & (source_x < width)
        & (source_y >= 0) & (source_y < height)
    )
    rotated = _filled_canvas(moved, fill_mode, fill_value)
    rotated[..., valid] = moved[..., source_y[valid], source_x[valid]]
    return np.moveaxis(rotated, (-2, -1), (y_axis, x_axis))


def _translate_exact(
    image: np.ndarray,
    shift_y: int,
    shift_x: int,
    y_axis: int,
    x_axis: int,
    fill_value: float,
    fill_mode: str,
) -> np.ndarray:
    moved = np.moveaxis(image, (y_axis, x_axis), (-2, -1))
    height, width = moved.shape[-2:]
    translated = _filled_canvas(moved, fill_mode, fill_value)
    if abs(shift_y) >= height or abs(shift_x) >= width:
        return np.moveaxis(translated, (-2, -1), (y_axis, x_axis))
    src_y = slice(max(0, -shift_y), min(height, height - shift_y))
    dst_y = slice(max(0, shift_y), min(height, height + shift_y))
    src_x = slice(max(0, -shift_x), min(width, width - shift_x))
    dst_x = slice(max(0, shift_x), min(width, width + shift_x))
    translated[..., dst_y, dst_x] = moved[..., src_y, src_x]
    return np.moveaxis(translated, (-2, -1), (y_axis, x_axis))


def apply_augmentation(
    image: np.ndarray,
    axes: str,
    augmentation: Augmentation,
    fill_value: float = 0,
    fill_mode: str = "constant",
) -> np.ndarray:
    y_axis, x_axis = spatial_axes(axes)
    transformed = image
    if augmentation.quarter_turns:
        transformed = np.rot90(
            transformed, k=augmentation.quarter_turns, axes=(y_axis, x_axis)
        )
    if augmentation.flip_lr:
        transformed = np.flip(transformed, axis=x_axis)
    if augmentation.angle_degrees:
        transformed = _rotate_nearest(
            transformed, augmentation.angle_degrees, y_axis, x_axis,
            fill_value, fill_mode,
        )
    if augmentation.shift_y or augmentation.shift_x:
        transformed = _translate_exact(
            transformed, augmentation.shift_y, augmentation.shift_x,
            y_axis, x_axis, fill_value, fill_mode,
        )
    return np.ascontiguousarray(transformed)


def _transformed_metadata(
    payload: TiffPayload, augmentation: Augmentation
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = dict(payload.metadata)
    write_options = dict(payload.write_options)
    if augmentation.quarter_turns % 2:
        if "resolution" in write_options:
            x_resolution, y_resolution = write_options["resolution"]
            write_options["resolution"] = (y_resolution, x_resolution)
        for x_key, y_key in (
            ("PhysicalSizeX", "PhysicalSizeY"),
            ("PhysicalSizeXUnit", "PhysicalSizeYUnit"),
        ):
            if x_key in metadata and y_key in metadata:
                metadata[x_key], metadata[y_key] = metadata[y_key], metadata[x_key]
    return metadata, write_options


def write_tiff_atomic(
    destination: Path,
    data: np.ndarray,
    payload: TiffPayload,
    augmentation: Augmentation,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata, write_options = _transformed_metadata(payload, augmentation)
    temp_handle, temp_name = tempfile.mkstemp(
        prefix=f".{destination.stem}_", suffix=destination.suffix, dir=destination.parent
    )
    os.close(temp_handle)
    temp_path = Path(temp_name)
    try:
        tifffile.imwrite(
            temp_path,
            data,
            ome=payload.kind == "ome",
            imagej=payload.kind == "imagej",
            metadata=metadata,
            **write_options,
        )
        with tifffile.TiffFile(temp_path) as check:
            if check.series[0].shape != data.shape:
                raise OSError("Written TIFF shape does not match the transformed image.")
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)


def destination_for(
    source_path: Path,
    output_dir: Path,
    relative_parent: Path,
    augmentation: Augmentation,
    masks: bool = False,
) -> Path:
    root = output_dir / "masks" if masks else output_dir / "images"
    return root / relative_parent / f"{source_path.stem}_{augmentation.suffix}{source_path.suffix}"


def _mask_for(
    source_path: Path, input_root: Path, mask_input_path: Path | None
) -> Path | None:
    if mask_input_path is None:
        return None
    if input_root.is_file():
        return mask_input_path
    return mask_input_path / source_path.relative_to(input_root)


def augment_file(
    source_path: Path,
    input_root: Path,
    output_dir: Path,
    overwrite: bool,
    augmentations: list[Augmentation],
    fill_mode: str,
    fill_value: float,
    mask_input_path: Path | None = None,
) -> list[AugmentResult]:
    relative_parent = Path() if input_root.is_file() else source_path.relative_to(input_root).parent
    mask_source = _mask_for(source_path, input_root, mask_input_path)
    try:
        payload = read_tiff(source_path)
        if mask_source is not None and not mask_source.exists():
            raise FileNotFoundError(f"Matching mask not found: {mask_source}")
        mask_payload = read_tiff(mask_source) if mask_source is not None else None
        if mask_payload is not None:
            image_y, image_x = spatial_axes(payload.axes)
            mask_y, mask_x = spatial_axes(mask_payload.axes)
            image_size = (payload.data.shape[image_y], payload.data.shape[image_x])
            mask_size = (mask_payload.data.shape[mask_y], mask_payload.data.shape[mask_x])
            if image_size != mask_size:
                raise ValueError(f"Image/mask spatial sizes differ: {image_size} vs {mask_size}")
    except Exception as exc:
        return [AugmentResult(
            str(source_path), "unsupported", "", "skipped", str(exc),
            mask_source_path=str(mask_source or ""),
        )]

    results: list[AugmentResult] = []
    for augmentation in augmentations:
        destination = destination_for(source_path, output_dir, relative_parent, augmentation)
        mask_destination = (
            destination_for(mask_source, output_dir, relative_parent, augmentation, masks=True)
            if mask_source is not None else None
        )
        existing = destination.exists() or (
            mask_destination is not None and mask_destination.exists()
        )
        if existing and not overwrite:
            results.append(AugmentResult(
                str(source_path), augmentation.suffix, str(destination), "skipped",
                "Destination already exists. Use --overwrite to replace it.", payload.axes,
                str(mask_source or ""), str(mask_destination or ""),
            ))
            continue
        try:
            transformed = apply_augmentation(
                payload.data, payload.axes, augmentation, fill_value, fill_mode
            )
            transformed_mask = (
                apply_augmentation(
                    mask_payload.data, mask_payload.axes, augmentation, 0, "constant"
                ) if mask_payload is not None else None
            )
            write_tiff_atomic(destination, transformed, payload, augmentation)
            if mask_destination is not None and transformed_mask is not None and mask_payload is not None:
                write_tiff_atomic(mask_destination, transformed_mask, mask_payload, augmentation)
            results.append(AugmentResult(
                str(source_path), augmentation.suffix, str(destination), "written", "",
                payload.axes, str(mask_source or ""), str(mask_destination or ""),
            ))
        except Exception as exc:
            destination.unlink(missing_ok=True)
            if mask_destination is not None:
                mask_destination.unlink(missing_ok=True)
            results.append(AugmentResult(
                str(source_path), augmentation.suffix, str(destination), "failed", str(exc),
                payload.axes, str(mask_source or ""), str(mask_destination or ""),
            ))
    return results


def write_report(output_dir: Path, results: list[AugmentResult]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = output_dir / f"augmentation_report_{timestamp}.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AugmentResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)
    return report_path


def augment_images(
    input_path: Path,
    output_dir: Path,
    recursive: bool,
    overwrite: bool,
    options: AugmentationOptions | None = None,
    mask_input_path: Path | None = None,
) -> tuple[list[AugmentResult], Path]:
    options = options or AugmentationOptions()
    if options.fill_mode not in {"border_median", "constant"}:
        raise ValueError(f"Unsupported fill mode: {options.fill_mode}")
    validate_paths(input_path, output_dir, mask_input_path)
    excluded_directories = (output_dir,) if input_path.is_dir() else ()
    source_paths = iter_tiff_files(
        input_path,
        recursive,
        excluded_directories=excluded_directories,
    )
    if not source_paths:
        raise FileNotFoundError(f"No TIF/TIFF files found in: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    augmentations = build_augmentations(options)
    results: list[AugmentResult] = []
    for source_path in source_paths:
        results.extend(augment_file(
            source_path, input_path, output_dir, overwrite, augmentations,
            options.fill_mode, options.fill_value, mask_input_path,
        ))
    return results, write_report(output_dir, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Axis-safe TIF/TIFF image and mask augmentation."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mask-input", type=Path, help="Matching mask file or mirrored mask folder.")
    parser.add_argument("--no-quarter-turns", action="store_true")
    parser.add_argument("--no-flips", action="store_true")
    parser.add_argument(
        "--small-angles", type=float, nargs="*", default=(), metavar="DEG",
        help="Optional nearest-neighbour rotations, for example -5 5.",
    )
    parser.add_argument(
        "--shift-pixels", type=int, default=0,
        help="Create exact integer up/down/left/right translations.",
    )
    parser.add_argument(
        "--fill-mode",
        choices=("border-median", "constant"),
        default="border-median",
        help="Fill empty borders per plane from observed border pixels (default) or with a constant.",
    )
    parser.add_argument("--fill-value", type=float, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = AugmentationOptions(
        include_quarter_turns=not args.no_quarter_turns,
        include_horizontal_flips=not args.no_flips,
        small_angles=tuple(args.small_angles),
        shift_pixels=args.shift_pixels,
        fill_mode=args.fill_mode.replace("-", "_"),
        fill_value=args.fill_value,
    )
    results, report_path = augment_images(
        args.input_path, args.output_dir, args.recursive, args.overwrite,
        options, args.mask_input,
    )
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("written", "skipped", "failed")
    }
    print(
        f"Done. Written: {counts['written']}, skipped: {counts['skipped']}, "
        f"failed: {counts['failed']}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
