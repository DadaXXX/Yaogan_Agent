"""地理空间工具 — 坐标转换、地理裁剪、重投影、分区统计。"""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import rowcol, from_bounds
except ImportError:
    rasterio = None

from src.tools._utils import safe_path


class GeoToolkit:
    """地理空间分析工具集"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir

    def clip_by_geo(self, image_path: str, bounds: str) -> str:
        """按地理坐标（经纬度）裁剪影像。

        bounds 格式: "lon_min,lat_min,lon_max,lat_max"
        """
        if rasterio is None:
            return "地理裁剪失败：缺少 rasterio 依赖。"

        try:
            path = safe_path(image_path, self.output_dir)
        except ValueError as e:
            return f"路径错误: {e}"
        if not path.exists():
            return f"文件不存在: {image_path}"

        try:
            parts = [float(p.strip()) for p in bounds.replace(",", " ").split()]
            if len(parts) != 4:
                return "bounds 格式错误，应为 4 个数值: lon_min lat_min lon_max lat_max"
            lon_min, lat_min, lon_max, lat_max = parts

            with rasterio.open(path) as src:
                if src.crs is None:
                    return "影像无 CRS 信息，无法进行地理坐标裁剪。"

                row_min, col_min = rowcol(src.transform, lon_min, lat_max)
                row_max, col_max = rowcol(src.transform, lon_max, lat_min)

                row_min = max(0, int(row_min))
                col_min = max(0, int(col_min))
                row_max = min(src.height, int(row_max) + 1)
                col_max = min(src.width, int(col_max) + 1)

                if row_min >= row_max or col_min >= col_max:
                    return f"裁剪范围在影像外。影像范围: {src.bounds}"

                data = src.read(window=((row_min, row_max), (col_min, col_max)))

                import datetime
                out_dir = Path(self.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                base = f"{path.stem}_geoclip_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

                profile = src.profile.copy()
                new_transform = src.window_transform(((row_min, row_max), (col_min, col_max)))
                profile.update(
                    height=data.shape[1], width=data.shape[2],
                    transform=new_transform,
                )

                tiff_path = out_dir / f"{base}.tif"
                with rasterio.open(tiff_path, "w", **profile) as dst:
                    dst.write(data)

                return (
                    f"地理坐标裁剪完成。\n"
                    f"  裁剪范围: [{lon_min:.4f}, {lat_min:.4f}] - [{lon_max:.4f}, {lat_max:.4f}]\n"
                    f"  输出大小: {data.shape[2]}x{data.shape[1]}\n"
                    f"  结果已保存至: {tiff_path}"
                )
        except Exception as e:
            return f"地理裁剪失败: {e}"

    def reproject_raster(self, image_path: str, target_epsg: str, resolution: Optional[str] = None) -> str:
        """重投影影像到目标 CRS。

        target_epsg: 如 "EPSG:4326" 或 "32650"
        resolution: 可选，如 "30" 表示 30m
        """
        if rasterio is None:
            return "重投影失败：缺少 rasterio 依赖。"

        try:
            path = safe_path(image_path, self.output_dir)
        except ValueError as e:
            return f"路径错误: {e}"
        if not path.exists():
            return f"文件不存在: {image_path}"

        try:
            epsg = target_epsg.replace("EPSG:", "").replace("epsg:", "").strip()
            dst_crs = f"EPSG:{epsg}"

            with rasterio.open(path) as src:
                if resolution:
                    res = float(resolution)
                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height,
                        *src.bounds, resolution=res,
                    )
                else:
                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds,
                    )

                kwargs = src.meta.copy()
                kwargs.update(
                    crs=dst_crs, transform=transform,
                    width=width, height=height,
                )

                import datetime
                out_dir = Path(self.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                base = f"{path.stem}_reproject_{epsg}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                    f"重投影完成。\n"
                    f"  源 CRS: {src.crs} → 目标 CRS: {dst_crs}\n"
                    f"  输出大小: {width}x{height}\n"
                    f"  结果已保存至: {tiff_path}"
                )
        except Exception as e:
            return f"重投影失败: {e}"

    def zonal_statistics(self, image_path: str, geojson_path: str) -> str:
        """按 GeoJSON / shapefile 矢量区域计算分区统计。

        geojson_path: 矢量文件路径（GeoJSON 或 shp）
        """
        if rasterio is None:
            return "分区统计失败：缺少 rasterio 依赖。"

        try:
            path = safe_path(image_path, self.output_dir)
            vec_path = safe_path(geojson_path, self.output_dir)
        except ValueError as e:
            return f"路径错误: {e}"
        if not path.exists():
            return f"影像文件不存在: {image_path}"
        if not vec_path.exists():
            return f"矢量文件不存在: {geojson_path}"

        try:
            import json
            try:
                import fiona
            except ImportError:
                return "分区统计需要 fiona 库: pip install fiona"

            with rasterio.open(path) as src:
                data = src.read()
                if data.ndim == 3:
                    data = np.transpose(data, (1, 2, 0))

                with fiona.open(vec_path) as features:
                    results = []
                    for feat in features:
                        geom = feat.get("geometry", {})
                        props = feat.get("properties", {})
                        name = props.get("name", props.get("Name", f"Feature_{feat['id']}"))

                        mask = rasterio.features.geometry_mask(
                            [geom], transform=src.transform,
                            invert=True, out_shape=(src.height, src.width),
                        )

                        if not mask.any():
                            continue

                        if data.ndim == 3:
                            zone_data = data[mask]
                            band_means = [np.mean(zone_data[:, i]) for i in range(zone_data.shape[1])]
                        else:
                            zone_data = data[mask]
                            band_means = [np.mean(zone_data)]

                        results.append({
                            "name": name,
                            "pixel_count": int(mask.sum()),
                            "area": f"{mask.sum() * abs(src.transform.a * src.transform.e):.2f} m2",
                            "band_means": band_means,
                        })

                if not results:
                    return "矢量范围与影像无交集。"

                lines = ["分区统计结果:"]
                for r in results:
                    means_str = ", ".join(f"{m:.3f}" for m in r["band_means"])
                    lines.append(
                        f"  {r['name']}: {r['pixel_count']} 像素, "
                        f"面积约 {r['area']}, 均值: [{means_str}]"
                    )

                return "\n".join(lines)

        except Exception as e:
            return f"分区统计失败: {e}"

    def extract_roi(self, image_path: str, geojson_path: str) -> str:
        """按矢量边界提取 ROI，输出掩码后的影像。

        geojson_path: 矢量文件路径
        """
        if rasterio is None:
            return "ROI 提取失败：缺少 rasterio 依赖。"

        try:
            path = safe_path(image_path, self.output_dir)
            vec_path = safe_path(geojson_path, self.output_dir)
        except ValueError as e:
            return f"路径错误: {e}"
        if not path.exists():
            return f"影像文件不存在: {image_path}"
        if not vec_path.exists():
            return f"矢量文件不存在: {geojson_path}"

        try:
            import fiona

            with rasterio.open(path) as src:
                data = src.read()

                with fiona.open(vec_path) as features:
                    geom = features[0].get("geometry", {})
                    mask = rasterio.features.geometry_mask(
                        [geom], transform=src.transform,
                        invert=True, out_shape=(src.height, src.width),
                    )

                masked = data.astype(np.float32)
                masked[:, ~mask] = np.nan

                import datetime
                out_dir = Path(self.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                base = f"{path.stem}_roi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                tiff_path = out_dir / f"{base}.tif"

                profile = src.profile.copy()
                profile.update(dtype="float32")
                with rasterio.open(tiff_path, "w", **profile) as dst:
                    dst.write(masked)

                return (
                    f"ROI 提取完成。\n"
                    f"  有效像素: {int(mask.sum())}\n"
                    f"  结果已保存至: {tiff_path}"
                )

        except Exception as e:
            return f"ROI 提取失败: {e}。需要 fiona: pip install fiona"
