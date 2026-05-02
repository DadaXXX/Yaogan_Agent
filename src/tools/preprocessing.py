"""预处理工具 — 云掩膜、归一化、直方图匹配、重采样。"""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import cv2
except ImportError:
    cv2 = None


class PreprocessingToolkit:
    """遥感影像预处理工具集"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir

    def cloud_mask(self, image_path: str, sensor: str = "auto") -> str:
        """利用 QA 波段自动生成云掩膜。

        sensor: "sentinel2" / "landsat8" / "auto"
        输出 0-1 掩膜（1=晴空，0=云/云影）。
        """
        if rasterio is None:
            return "云掩膜失败：缺少 rasterio 依赖。"

        path = Path(image_path)
        if not path.exists():
            return f"文件不存在: {image_path}"

        try:
            with rasterio.open(path) as src:
                n_bands = src.count
                descriptions = src.descriptions or ()

                qa_band = None
                band_idx = None

                for i, desc in enumerate(descriptions):
                    d = desc.lower() if desc else ""
                    if 'qa' in d or 'qa60' in d or 'qa_pixel' in d or 'pixel_qa' in d or 'fmask' in d:
                        qa_band = src.read(i + 1)
                        band_idx = i + 1
                        break

                if qa_band is None and n_bands >= 10:
                    band_idx = 10
                    qa_band = src.read(band_idx)

                if qa_band is not None:
                    cloud_mask_arr = np.zeros_like(qa_band, dtype=np.uint8)
                    cloud_mask_arr[qa_band == 0] = 1
                    mask_out = cloud_mask_arr.astype(np.float32)
                    method = f"(QA 波段 {band_idx})"
                else:
                    return (
                        "未找到 QA 波段，无法自动云掩膜。\n"
                        "请确保影像包含 QA 波段（如 Sentinel-2 QA60 或 Landsat QA_PIXEL）。\n"
                        "或使用手动掩膜: band_math 工具自定义表达式。"
                    )

            import datetime
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = f"{path.stem}_cloudmask_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tiff_path = out_dir / f"{base}.tif"

            with rasterio.open(path) as src:
                profile = src.profile.copy()
                profile.update(count=1, dtype=mask_out.dtype.name)
                with rasterio.open(tiff_path, "w", **profile) as dst:
                    dst.write(mask_out, 1)

            cloud_pct = 100 - np.mean(mask_out) * 100
            return (
                f"云掩膜生成完成 {method}。\n"
                f"  估算云覆盖: {cloud_pct:.1f}%\n"
                f"  晴空像元: {int(np.sum(mask_out))} / {mask_out.size}\n"
                f"  结果已保存至: {tiff_path}"
            )

        except Exception as e:
            return f"云掩膜失败: {e}"

    def normalize_image(self, image_path: str, method: str = "minmax", bands: Optional[str] = None) -> str:
        """影像归一化。

        method: "minmax" (0-1) 或 "zscore" (均值0 标准差1)
        bands: 指定要归一化的波段，如 "1,2,3"，默认全部
        """
        if rasterio is None:
            return "归一化失败：缺少 rasterio 依赖。"

        path = Path(image_path)
        if not path.exists():
            return f"文件不存在: {image_path}"

        try:
            with rasterio.open(path) as src:
                data = src.read()
                if bands:
                    indices = [int(b.strip()) - 1 for b in bands.replace(",", " ").split()]
                else:
                    indices = list(range(data.shape[0]))

                result = data.astype(np.float32)
                for i in indices:
                    band = data[i].astype(np.float32)
                    bmin, bmax = np.percentile(band, [2, 98])
                    if method == "minmax":
                        result[i] = np.clip((band - bmin) / (bmax - bmin + 1e-6), 0, 1)
                    elif method == "zscore":
                        bmean, bstd = np.nanmean(band), np.nanstd(band)
                        result[i] = (band - bmean) / (bstd + 1e-6)

            import datetime
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = f"{path.stem}_norm_{method}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tiff_path = out_dir / f"{base}.tif"

            profile = src.profile.copy()
            profile.update(dtype="float32")
            with rasterio.open(tiff_path, "w", **profile) as dst:
                dst.write(result)

            return (
                f"归一化完成 ({method})。\n"
                f"  处理波段: {len(indices)} 个\n"
                f"  结果已保存至: {tiff_path}"
            )
        except Exception as e:
            return f"归一化失败: {e}"

    def histogram_match(self, source_path: str, reference_path: str) -> str:
        """将源影像直方图匹配到参考影像（多时相分析必备）。

        只处理重叠波段。
        """
        if rasterio is None:
            return "直方图匹配失败：缺少 rasterio 依赖。"

        s_path = Path(source_path)
        r_path = Path(reference_path)
        if not s_path.exists():
            return f"源文件不存在: {source_path}"
        if not r_path.exists():
            return f"参考文件不存在: {reference_path}"

        try:
            with rasterio.open(s_path) as s_src, rasterio.open(r_path) as r_src:
                src_data = s_src.read().astype(np.float32)
                ref_data = r_src.read().astype(np.float32)
                n_bands = min(src_data.shape[0], ref_data.shape[0])
                matched = src_data.copy()

                for i in range(n_bands):
                    s = src_data[i].ravel()
                    r = ref_data[i].ravel()
                    src_sorted = np.sort(s)
                    ref_sorted = np.sort(r)
                    src_p = np.interp(s, src_sorted, ref_sorted)
                    matched[i] = src_p.reshape(src_data[i].shape)

            import datetime
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = f"{s_path.stem}_histmatch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tiff_path = out_dir / f"{base}.tif"

            profile = s_src.profile.copy()
            profile.update(dtype="float32", count=n_bands)
            with rasterio.open(tiff_path, "w", **profile) as dst:
                dst.write(matched[:n_bands])

            return (
                f"直方图匹配完成。\n"
                f"  源: {source_path} → 参考: {reference_path}\n"
                f"  匹配波段: {n_bands}\n"
                f"  结果已保存至: {tiff_path}"
            )
        except Exception as e:
            return f"直方图匹配失败: {e}"

    def resample_raster(self, image_path: str, target_resolution: str) -> str:
        """重采样到目标分辨率（米）。

        target_resolution: 目标分辨率，如 "30"
        """
        if rasterio is None:
            return "重采样失败：缺少 rasterio 依赖。"

        path = Path(image_path)
        if not path.exists():
            return f"文件不存在: {image_path}"

        try:
            target_res = float(target_resolution)
            from rasterio.warp import calculate_default_transform, reproject, Resampling

            with rasterio.open(path) as src:
                dst_crs = src.crs
                transform, width, height = calculate_default_transform(
                    src.crs, src.crs, src.width, src.height, *src.bounds,
                    resolution=target_res,
                )

                kwargs = src.meta.copy()
                kwargs.update(transform=transform, width=width, height=height)

                import datetime
                out_dir = Path(self.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                base = f"{path.stem}_resample_{int(target_res)}m_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                tiff_path = out_dir / f"{base}.tif"

                with rasterio.open(tiff_path, "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform, src_crs=src.crs,
                            dst_transform=transform, dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                        )

                return (
                    f"重采样完成。\n"
                    f"  源分辨率: {src.res[0]:.1f}m → 目标: {target_res}m\n"
                    f"  输出大小: {width}x{height}\n"
                    f"  结果已保存至: {tiff_path}"
                )
        except Exception as e:
            return f"重采样失败: {e}"
