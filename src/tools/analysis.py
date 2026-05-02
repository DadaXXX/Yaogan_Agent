"""遥感分析工具 — 变化检测、PCA 降维、光谱剖面、详细统计。"""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None


class AnalysisToolkit:
    """遥感分析工具集"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir

    def change_detection(self, image_before: str, image_after: str, method: str = "difference") -> str:
        """两期影像变化检测。

        method: "difference" / "ratio" / "cva"
        """
        if rasterio is None:
            return "变化检测失败：缺少 rasterio 依赖。"

        b_path = Path(image_before)
        a_path = Path(image_after)
        if not b_path.exists():
            return f"前期影像不存在: {image_before}"
        if not a_path.exists():
            return f"后期影像不存在: {image_after}"

        try:
            with rasterio.open(b_path) as b_src, rasterio.open(a_path) as a_src:
                b_data = b_src.read(1).astype(np.float32)
                a_data = a_src.read(1).astype(np.float32)

                if method == "difference":
                    change = a_data - b_data
                    desc = "差分 (后期 - 前期)"
                elif method == "ratio":
                    change = np.log((a_data + 1e-6) / (b_data + 1e-6))
                    desc = "对数比值 log(after/before)"
                elif method == "cva":
                    diff = a_data - b_data
                    change_magnitude = np.abs(diff)
                    change = change_magnitude
                    desc = "变化矢量分析 (CVA) 幅度"

            change_mean = np.nanmean(change)
            change_std = np.nanstd(change)

            import datetime
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = f"change_{method}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tiff_path = out_dir / f"{base}.tif"
            png_path = out_dir / f"{base}.png"

            profile = b_src.profile.copy()
            profile.update(dtype=change.dtype.name, count=1)
            with rasterio.open(tiff_path, "w", **profile) as dst:
                dst.write(change, 1)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(change, cmap="RdBu_r", vmin=change_mean - 3 * change_std, vmax=change_mean + 3 * change_std)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(png_path, dpi=150, pad_inches=0)

            leg_fig, leg_ax = plt.subplots(figsize=(1.2, 6))
            from matplotlib.colorbar import ColorbarBase
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=change_mean - 3 * change_std, vmax=change_mean + 3 * change_std)
            ColorbarBase(leg_ax, cmap="RdBu_r", norm=norm, orientation="vertical")
            leg_fig.subplots_adjust(left=0.3, right=0.6, top=0.92, bottom=0.08)
            legend_path = out_dir / f"{base}_legend.png"
            leg_fig.savefig(legend_path, dpi=150)
            plt.close("all")

            # 分区统计
            pos = np.count_nonzero(change > change_std)
            neg = np.count_nonzero(change < -change_std)
            stable = change.size - pos - neg

            return (
                f"变化检测完成 ({desc})。\n"
                f"  变化均值: {change_mean:.4f}\n"
                f"  变化标准差: {change_std:.4f}\n"
                f"  显著增加 (>1σ): {pos / change.size * 100:.1f}%\n"
                f"  显著减少 (<-1σ): {neg / change.size * 100:.1f}%\n"
                f"  稳定区 (±1σ内): {stable / change.size * 100:.1f}%\n"
                f"  结果已保存至: {tiff_path}; {png_path}"
            )

        except Exception as e:
            return f"变化检测失败: {e}"

    def pca_transform(self, image_path: str, n_components: str = "3") -> str:
        """PCA 降维变换。

        n_components: 保留的主成分数量
        """
        if rasterio is None:
            return "PCA 失败：缺少 rasterio 依赖。"

        path = Path(image_path)
        if not path.exists():
            return f"文件不存在: {image_path}"

        try:
            from sklearn.decomposition import PCA

            n_comp = int(n_components)

            with rasterio.open(path) as src:
                data = src.read()
                profile = src.profile.copy()
                bands, h, w = data.shape
                n_comp = min(n_comp, bands)
                pixels = data.reshape(bands, -1).T.astype(np.float32)

                valid_mask = np.all(np.isfinite(pixels), axis=1)
                valid_pixels = pixels[valid_mask]

                pca = PCA(n_components=n_comp)
                transformed = pca.fit_transform(valid_pixels)

                result = np.full((h * w, n_comp), np.nan, dtype=np.float32)
                result[valid_mask] = transformed
                result = result.reshape(h, w, n_comp).transpose(2, 0, 1)

            evr = pca.explained_variance_ratio_

            import datetime
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = f"{path.stem}_pca{n_comp}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tiff_path = out_dir / f"{base}.tif"
            profile.update(count=n_comp, dtype="float32")
            with rasterio.open(tiff_path, "w", **profile) as dst:
                dst.write(result)

            evr_lines = [f"  PC{i+1}: {evr[i] * 100:.1f}% 方差" for i in range(n_comp)]
            total_var = sum(evr) * 100

            return (
                f"PCA 变换完成。\n"
                f"  原波段数: {bands} → 主成分数: {n_comp}\n"
                f"  解释方差:\n" + "\n".join(evr_lines) + "\n"
                f"  前 {n_comp} 个主成分累计解释 {total_var:.1f}% 方差\n"
                f"  结果已保存至: {tiff_path}"
            )

        except Exception as e:
            return f"PCA 变换失败: {e}（需要 scikit-learn）"

    def spectral_profile(self, image_path: str, x: str, y: str) -> str:
        """提取指定像素的光谱剖面。

        x, y: 像素坐标
        """
        if rasterio is None:
            return "光谱分析失败：缺少 rasterio 依赖。"

        path = Path(image_path)
        if not path.exists():
            return f"文件不存在: {image_path}"

        try:
            px, py = int(x), int(y)

            with rasterio.open(path) as src:
                if px < 0 or px >= src.width or py < 0 or py >= src.height:
                    return f"坐标 ({px},{py}) 超出影像范围 ({src.width}x{src.height})。"
                profile_data = src.read()[:, py, px]
                descriptions = src.descriptions or tuple(f"Band {i+1}" for i in range(src.count))

            lines = [f"像素 ({px},{py}) 光谱剖面:"]
            for i, (desc, val) in enumerate(zip(descriptions, profile_data)):
                lines.append(f"  {desc}: {val:.4f}")

            return "\n".join(lines)

        except Exception as e:
            return f"光谱分析失败: {e}"

    def compute_statistics(self, image_path: str, bands: Optional[str] = None) -> str:
        """计算影像详细统计信息（含直方图特征）。

        bands: 指定波段，如 "1,2,3"
        """
        if rasterio is None:
            return "统计失败：缺少 rasterio 依赖。"

        path = Path(image_path)
        if not path.exists():
            return f"文件不存在: {image_path}"

        try:
            with rasterio.open(path) as src:
                if bands:
                    indices = [int(b.strip()) - 1 for b in bands.replace(",", " ").split()]
                else:
                    indices = list(range(src.count))
                data = src.read()
                descriptions = src.descriptions or tuple(f"Band {i+1}" for i in range(src.count))

            lines = [f"影像统计 (共 {len(indices)} 个波段):"]
            for i in indices:
                band = data[i].astype(np.float32)
                valid = band[np.isfinite(band)]
                if len(valid) == 0:
                    lines.append(f"  Band {i+1}: 无效数据")
                    continue
                desc = descriptions[i]
                p2, p98 = np.percentile(valid, [2, 98])
                lines.append(
                    f"  {desc}: 均值={np.mean(valid):.3f}, 中位数={np.median(valid):.3f}, "
                    f"Std={np.std(valid):.3f}, Min={np.min(valid):.3f}, Max={np.max(valid):.3f}, "
                    f"P2={p2:.3f}, P98={p98:.3f}"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"统计失败: {e}"
