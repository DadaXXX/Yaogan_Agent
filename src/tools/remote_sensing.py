"""遥感影像分析工具集 — 图像加载、NDVI、SAR、分割、分类等。"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
# import rasterio
try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


class RemoteSensingToolkit:
    """遥感影像分析工具箱，所有方法返回文本描述供 LLM 消费。"""

    # SAM 模型缓存（懒加载）
    _sam_predictor = None
    _sam_model_type = None

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self._src_profile = None
        self._band_descriptions: Optional[tuple] = None

    # ── 基础图像信息 ──────────────────────────────────

    def describe_image(self, image_path: str) -> str:
        """读取影像基本信息"""
        img = self._load_image(image_path)
        if img is None:
            return "无法读取影像，请检查文件路径和格式。"
        return (
            f"路径: {image_path}\n"
            f"形状 (高×宽×波段): {img.shape}\n"
            f"数据类型: {img.dtype}\n"
            f"最小值: {np.min(img):.4f}\n"
            f"最大值: {np.max(img):.4f}\n"
            f"均值: {np.mean(img):.4f}\n"
            f"标准差: {np.std(img):.4f}"
        )

    # ── NDVI ───────────────────────────────────────────

    def _identify_red_nir_bands(self, n_bands: int) -> tuple[Optional[int], Optional[int]]:
        """从元数据或常见传感器配置推断红波段和近红外波段的索引（0-based）。"""
        red_idx: Optional[int] = None
        nir_idx: Optional[int] = None

        # 优先级 1：从 rasterio band descriptions 中匹配关键字
        if self._band_descriptions:
            for i, desc in enumerate(self._band_descriptions):
                if not desc:
                    continue
                d = desc.lower()
                # 通用关键词
                if ('red' in d and 'nir' not in d and 'swir' not in d and 'edge' not in d):
                    red_idx = i
                if 'nir' in d or 'near infrared' in d or 'near-ir' in d:
                    nir_idx = i
                # Sentinel-2: B4=Red, B8/B8A=NIR
                if d.startswith('b4') and (i > 0 or 'sentinel' in d):
                    red_idx = i
                if (d.startswith('b8') and 'b8a' not in d) or 'b8a' in d:
                    nir_idx = i
                # MODIS MOD09GA: sur_refl_b01=Red(620-670nm), sur_refl_b02=NIR(841-876nm)
                if 'b01' in d or ('sur_refl_red' in d and 'nir' not in d):
                    red_idx = i
                if 'b02' in d and 'red' not in d:
                    nir_idx = i
                # MODIS MOD13Q1: sur_refl_red, sur_refl_nir
                if d.endswith('_red') or d == 'red':
                    red_idx = i
                if d.endswith('_nir') or d == 'nir':
                    nir_idx = i

        if red_idx is not None and nir_idx is not None:
            return red_idx, nir_idx

        # 优先级 2：按波段数量 fallback 到已知传感器配置
        if n_bands == 4:
            return red_idx or 2, nir_idx or 3
        elif n_bands == 5:
            return red_idx or 3, nir_idx or 4
        elif n_bands == 7:
            return red_idx or 2, nir_idx or 3
        elif n_bands == 8:
            return red_idx or 3, nir_idx or 4
        elif n_bands >= 9:
            return red_idx or 3, nir_idx or 7
        else:
            return red_idx, nir_idx

    def compute_ndvi(self, image_path: str, red_band: Optional[int] = None, nir_band: Optional[int] = None) -> str:
        """计算 NDVI（归一化植被指数）。

        参数:
            red_band:  红波段索引（1-indexed），不指定则自动推断
            nir_band:  近红外波段索引（1-indexed），不指定则自动推断
        """
        img = self._load_image(image_path)
        if img is None:
            return "NDVI 计算失败，无法读取影像。"
        if img.ndim < 3:
            return "NDVI 计算失败：单波段影像无法计算 NDVI。"

        n_bands = img.shape[2]
        if n_bands < 2:
            return "NDVI 计算失败：需要至少 2 个波段。"

        # 确定红波段和近红外波段索引（转为 0-based）
        if red_band is not None and nir_band is not None:
            red_idx = red_band - 1
            nir_idx = nir_band - 1
        else:
            auto_red, auto_nir = self._identify_red_nir_bands(n_bands)
            red_idx = (red_band - 1) if red_band is not None else auto_red
            nir_idx = (nir_band - 1) if nir_band is not None else auto_nir

        if red_idx is None or nir_idx is None or red_idx >= n_bands or nir_idx >= n_bands:
            band_hint = f"影像共 {n_bands} 个波段"
            if self._band_descriptions:
                desc_list = [f"  Band {i+1}: {d or '(无描述)'}" for i, d in enumerate(self._band_descriptions)]
                band_hint += "\n" + "\n".join(desc_list)
            return (
                f"无法自动识别红/近红外波段。{band_hint}\n"
                "请通过 red_band 和 nir_band 参数指定波段索引（1-based），例如：\n"
                '  "计算 NDVI，红波段=4，近红外波段=5"'
            )

        red = img[:, :, red_idx].astype(np.float32)
        nir = img[:, :, nir_idx].astype(np.float32)

        ndvi = (nir - red) / (nir + red + 1e-6)
        mean_ndvi = np.nanmean(ndvi)
        std_ndvi = np.nanstd(ndvi)

        # 保存 NDVI 影像
        save_info = self._save_output(
            ndvi, "ndvi", image_path,
            cmap_name="RdYlGn", vmin=-1.0, vmax=1.0,
        )

        # 分区间统计
        water = np.count_nonzero(ndvi < 0) / ndvi.size * 100
        bare = np.count_nonzero((ndvi >= 0) & (ndvi < 0.2)) / ndvi.size * 100
        sparse = np.count_nonzero((ndvi >= 0.2) & (ndvi < 0.4)) / ndvi.size * 100
        dense = np.count_nonzero(ndvi >= 0.4) / ndvi.size * 100

        return (
            f"NDVI 统计:\n"
            f"  均值: {mean_ndvi:.4f}\n"
            f"  标准差: {std_ndvi:.4f}\n"
            f"  水体/阴影 (NDVI<0): {water:.1f}%\n"
            f"  裸土/建筑 (0~0.2): {bare:.1f}%\n"
            f"  稀疏植被 (0.2~0.4): {sparse:.1f}%\n"
            f"  茂密植被 (>0.4): {dense:.1f}%\n"
            f"{save_info}\n"
            f"结论: NDVI>0.4 为高植被覆盖，0.2~0.4 为中等覆盖，<0.2 为低覆盖或无植被。"
        )

    # ── SAR 背散射 ─────────────────────────────────────

    def compute_sar_backscatter(self, image_path: str) -> str:
        """计算 SAR 背散射统计"""
        img = self._load_image(image_path)
        if img is None:
            return "SAR 统计失败，无法读取影像。"

        if img.ndim == 3:
            img = np.mean(img, axis=2)

        mean_val = np.nanmean(img)
        std_val = np.nanstd(img)
        min_val = np.nanmin(img)
        max_val = np.nanmax(img)
        return (
            f"SAR 背散射统计:\n"
            f"  均值: {mean_val:.4f}\n"
            f"  标准差: {std_val:.4f}\n"
            f"  最小值: {min_val:.4f}\n"
            f"  最大值: {max_val:.4f}\n"
            "建议: 低值区域可能为水体（镜面反射），高值区域可能为城市或粗糙地表。"
        )

    # ── Otsu 阈值分割 ──────────────────────────────────

    def segment_otsu(self, image_path: str) -> str:
        """对影像进行 Otsu 阈值分割"""
        img = self._load_image(image_path)
        if img is None:
            return "分割失败，无法读取影像。"

        gray = self._to_grayscale(img)
        if gray is None:
            return "分割失败：无法转换为灰度图像。"
        if cv2 is None:
            return "分割失败：缺少 OpenCV 依赖。"

        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        save_info = self._save_output(
            mask, "otsu_mask", image_path,
            cmap_name="gray", vmin=0, vmax=255,
        )
        unique, counts = np.unique(mask, return_counts=True)
        total = counts.sum()
        distribution = ", ".join(
            [f"类别 {int(v)}: {int(c)} 像素 ({c / total * 100:.1f}%)"
             for v, c in zip(unique, counts)]
        )
        return (
            f"Otsu 阈值分割结果:\n"
            f"  类别分布: {distribution}\n"
            f"{save_info}\n"
            "可进一步用深度学习模型进行高精度语义分割。"
        )

    # ── SAM 语义分割 ───────────────────────────────────

    def segment_sam(self, image_path: str, model_type: str = "vit_b") -> str:
        """使用 SAM 模型进行自动语义分割，生成多目标掩码"""
        img = self._load_image(image_path)
        if img is None:
            return "SAM 分割失败，无法读取影像。"

        # 转换为 RGB（SAM 要求）
        rgb = self._to_rgb(img)
        if rgb is None:
            return "SAM 分割失败：无法转换为 RGB 图像。"

        # 加载 SAM（自动下载缺失权重）
        if self._load_sam(model_type) is None:
            return (
                "SAM 模型加载失败。请确认:\n"
                "1. segment-anything 已安装 (pip install segment-anything)\n"
                "2. 权重文件会自动下载到 models/sam/ 目录\n"
                "3. 网络连接正常（权重约 357MB）"
            )

        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

            weight_name = self.WEIGHT_URLS[model_type].rsplit("/", 1)[-1]
            weight_path = Path(__file__).parents[2] / "models" / "sam" / weight_name
            sam = sam_model_registry[model_type](checkpoint=str(weight_path))

            # 大图自动缩放（SAM 长边超过 1500 时内存不足）
            h, w = rgb.shape[:2]
            MAX_SIDE = 1500
            if max(h, w) > MAX_SIDE:
                scale = MAX_SIDE / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                scaled_info = f"  (原始 {w}x{h} → 缩放 {new_w}x{new_h} 后处理)\n"
            else:
                rgb_small = rgb
                scaled_info = ""

            mask_generator = SamAutomaticMaskGenerator(
                model=sam, points_per_side=32,
                pred_iou_thresh=0.7, stability_score_thresh=0.7,
                min_mask_region_area=100,
            )
            masks = mask_generator.generate(rgb_small)

            if not masks:
                return "SAM 分割未生成任何掩码。"

            masks.sort(key=lambda m: m["area"], reverse=True)

            sh, sw = rgb_small.shape[:2]
            label_map = np.zeros((sh, sw), dtype=np.int32)
            area_pcts = []
            for idx, m in enumerate(masks, 1):
                mask = m["segmentation"]
                label_map[mask] = idx
                area_pcts.append(m["area"] / (sh * sw) * 100)

            coverage = sum(area_pcts)
            n_masks = len(masks)

            save_info = self._save_output(
                label_map, "sam_mask", image_path,
                cmap_name="tab20", vmin=0, vmax=min(n_masks, 20),
            )

            top_n = min(5, n_masks)
            top_lines = [
                f"  目标 {i+1}: {area_pcts[i]:.2f}% (置信度: {masks[i]['predicted_iou']:.3f})"
                for i in range(top_n)
            ]

            return (
                f"SAM 语义分割结果:\n"
                f"  模型: SAM-{model_type}\n"
                f"{scaled_info}"
                f"  检测目标数: {n_masks}\n"
                f"  总覆盖面积: {coverage:.1f}%\n"
                + "\n".join(top_lines) + "\n"
                f"{save_info}\n"
                "注: 每个颜色代表一个独立分割目标。"
            )

        except Exception as e:
            return f"SAM 分割执行异常: {e}"

    # ── 土地覆被分类 ───────────────────────────────────

    def classify_landcover(self, image_path: str, n_clusters: int = 4) -> str:
        """对影像进行无监督土地覆被分类（KMeans 聚类）"""
        if KMeans is None:
            return "分类失败：缺少 sklearn 依赖 (pip install scikit-learn)"

        img = self._load_image(image_path)
        if img is None:
            return "分类失败，无法读取影像。"

        h, w = img.shape[:2]
        pixels = img.reshape(-1, img.shape[2]).astype(np.float32)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(pixels)
        labels_2d = labels.reshape(h, w)

        save_info = self._save_output(
            labels_2d.astype(np.int32), "landcover", image_path,
            cmap_name="tab10", vmin=0, vmax=n_clusters - 1,
        )
        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        # 按像素占比排序
        sorted_idx = np.argsort(-counts)
        lines = []
        cluster_names = {
            0: "类别 0 (可能为水体/阴影)",
            1: "类别 1 (可能为植被/农田)",
            2: "类别 2 (可能为裸土/建筑)",
            3: "类别 3 (可能为其他地物)",
        }
        for i, idx in enumerate(sorted_idx):
            pct = counts[idx] / total * 100
            name = cluster_names.get(int(idx), f"类别 {int(idx)}")
            mean_spec = np.mean(pixels[labels == idx], axis=0)
            mean_str = ", ".join([f"{v:.2f}" for v in mean_spec[:3]])
            lines.append(f"  {name}: {pct:.1f}% | 光谱均值 (前3波段): [{mean_str}]")

        return (
            f"土地覆被分类结果（KMeans, k={n_clusters}）:\n"
            + "\n".join(lines) + "\n"
            f"{save_info}\n"
            "注: 无监督分类的类别含义需结合影像实际情况判读。"
        )

    # ── 通用遥感指数计算 ──────────────────────────────

    INDEX_FORMULAS = {
        "NDVI": {
            "desc": "归一化植被指数",
            "expr": lambda R, NIR: (NIR - R) / (NIR + R + 1e-6),
            "params": ["R", "NIR"],
            "range": (-1, 1),
        },
        "LSWI": {
            "desc": "陆地表面水分指数",
            "expr": lambda NIR, SWIR: (NIR - SWIR) / (NIR + SWIR + 1e-6),
            "params": ["NIR", "SWIR"],
            "range": (-1, 1),
        },
        "NDWI": {
            "desc": "归一化水体指数",
            "expr": lambda G, NIR: (G - NIR) / (G + NIR + 1e-6),
            "params": ["G", "NIR"],
            "range": (-1, 1),
        },
        "MNDWI": {
            "desc": "改进归一化水体指数",
            "expr": lambda G, SWIR: (G - SWIR) / (G + SWIR + 1e-6),
            "params": ["G", "SWIR"],
            "range": (-1, 1),
        },
        "EVI": {
            "desc": "增强型植被指数",
            "expr": lambda B, R, NIR: 2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * B + 1 + 1e-6),
            "params": ["B", "R", "NIR"],
            "range": (-1, 1),
        },
        "SAVI": {
            "desc": "土壤调节植被指数",
            "expr": lambda R, NIR: (NIR - R) / (NIR + R + 0.5 + 1e-6) * 1.5,
            "params": ["R", "NIR"],
            "range": (-1, 1),
        },
        "NBR": {
            "desc": "归一化燃烧指数",
            "expr": lambda NIR, SWIR2: (NIR - SWIR2) / (NIR + SWIR2 + 1e-6),
            "params": ["NIR", "SWIR2"],
            "range": (-1, 1),
        },
        "VARI": {
            "desc": "可视大气阻抗指数",
            "expr": lambda B, G, R: (G - R) / (G + R - B + 1e-6),
            "params": ["B", "G", "R"],
            "range": (-1, 1),
        },
    }

    def compute_index(self, image_path: str, index: str, band_map: str, output_name: Optional[str] = None) -> str:
        """计算遥感指数（NDVI/LSWI/NDWI/EVI/SAVI 等）。

        参数:
            image_path: 影像文件路径
            index:      指数名称（NDVI/LSWI/NDWI/MNDWI/EVI/SAVI/NBR/VARI）
            band_map:   波段映射，格式 "键=波段号,..."，如 "R=3,NIR=4"
            output_name: 可选输出文件名标识
        """
        img = self._load_image(image_path)
        if img is None:
            return "指数计算失败，无法读取影像。"
        if img.ndim < 3:
            return "指数计算失败：影像无波段维度。"

        key = index.upper()
        formula = self.INDEX_FORMULAS.get(key)
        if not formula:
            names = ", ".join(self.INDEX_FORMULAS.keys())
            return f"不支持的指数 '{index}'。支持的指数: {names}"

        # 解析 band_map: "R=3,NIR=4" → {"R": 2, "NIR": 3}
        band_indices = {}
        for pair in band_map.replace("，", ",").split(","):
            pair = pair.strip()
            if "=" not in pair:
                return f"波段映射格式错误: '{pair}'，应为 键=波段号（如 R=3）。"
            k, v = pair.split("=", 1)
            k = k.strip().upper()
            try:
                band_indices[k] = int(v.strip()) - 1  # 转为 0-based
            except ValueError:
                return f"波段映射格式错误: '{v}' 不是有效的波段号。"

        n_bands = img.shape[2]
        missing = [p for p in formula["params"] if p not in band_indices]
        if missing:
            return (
                f"缺少波段映射参数: {', '.join(missing)}\n"
                f"指数 {key} 需要: {', '.join(formula['params'])}\n"
                f"示例: band_map=\"{','.join(p+'=?' for p in formula['params'])}\""
            )

        band_arrays = {}
        for param, idx in band_indices.items():
            if idx >= n_bands:
                return f"波段号 {idx + 1} 超出影像范围（共 {n_bands} 个波段）。"
            band_arrays[param] = img[:, :, idx].astype(np.float32)

        try:
            result = formula["expr"](**band_arrays)
        except Exception as e:
            return f"指数 {key} 计算失败: {e}"
        result = np.asarray(result, dtype=np.float32)

        valid = result[np.isfinite(result)]
        mean_val = np.nanmean(valid)
        std_val = np.nanstd(valid)
        min_val = np.nanmin(valid)
        max_val = np.nanmax(valid)
        vmin, vmax = formula["range"]

        cmap = "RdYlGn" if key in ("NDVI", "EVI", "SAVI") else "RdBu"
        save_info = self._save_output(
            result, f"{key.lower()}", image_path,
            cmap_name=cmap, vmin=vmin, vmax=vmax,
        )

        zone_lines = []
        if key in ("NDVI", "EVI", "SAVI"):
            water = np.count_nonzero(result < 0) / result.size * 100
            bare = np.count_nonzero((result >= 0) & (result < 0.2)) / result.size * 100
            sparse = np.count_nonzero((result >= 0.2) & (result < 0.4)) / result.size * 100
            dense = np.count_nonzero(result >= 0.4) / result.size * 100
            zone_lines = [
                f"  水体/阴影 (<0): {water:.1f}%",
                f"  裸土/建筑 (0~0.2): {bare:.1f}%",
                f"  稀疏植被 (0.2~0.4): {sparse:.1f}%",
                f"  茂密植被 (>0.4): {dense:.1f}%",
            ]
        elif key in ("NDWI", "MNDWI", "LSWI"):
            water_pct = np.count_nonzero(result > 0) / result.size * 100
            land_pct = np.count_nonzero(result <= 0) / result.size * 100
            zone_lines = [
                f"  水体 (>0): {water_pct:.1f}%",
                f"  非水体 (≤0): {land_pct:.1f}%",
            ]

        band_desc = ", ".join(f"{k}=Band{v+1}" for k, v in band_indices.items())
        return (
            f"{key} ({formula['desc']}) 计算结果:\n"
            f"  波段映射: {band_desc}\n"
            f"  均值: {mean_val:.4f}\n"
            f"  标准差: {std_val:.4f}\n"
            f"  范围: [{min_val:.4f}, {max_val:.4f}]\n"
            + ("\n".join(zone_lines) + "\n" if zone_lines else "")
            + f"{save_info}"
        )

    # ── 波段运算 ─────────────────────────────────────

    def band_math(self, image_path: str, expression: str, output_name: Optional[str] = None) -> str:
        """对影像波段执行算术表达式运算。

        参数:
            image_path: 影像文件路径
            expression: 运算表达式，用 b1/b2/... 表示波段，如 "(b4 - b3) / (b4 + b3)"
            output_name: 可选输出文件名标识
        """
        img = self._load_image(image_path)
        if img is None:
            return "波段运算失败，无法读取影像。"
        if img.ndim < 3:
            return "波段运算失败：影像无波段维度。"

        # 将 b1/b2/... 绑定到波段数组
        local_vars = {}
        for i in range(img.shape[2]):
            local_vars[f'b{i+1}'] = img[:, :, i].astype(np.float32)

        try:
            result = eval(expression, {"__builtins__": {}, "np": np}, local_vars)
        except Exception as e:
            return f"波段运算失败: {e}\n支持的语法: b1, b2, ... 表示波段，支持 + - * / () 和 np.* 函数"

        result = np.asarray(result, dtype=np.float32)
        if result.ndim not in (2, 3):
            return f"波段运算结果维度异常: {result.ndim}D，期望 2D 或 3D。"

        # 扩展为 3 维统一处理
        result_3d = result[:, :, np.newaxis] if result.ndim == 2 else result
        tag = output_name or f"bandmath_{Path(image_path).stem}"
        save_info = self._save_output(
            result_3d, tag, image_path,
            cmap_name="viridis",
        )
        return f"波段运算完成。结果形状: {result.shape}\n{save_info}"

    # ── 裁剪影像 ─────────────────────────────────────

    def clip_raster(self, image_path: str, bounds: str, output_name: Optional[str] = None) -> str:
        """按像素坐标裁剪影像。

        参数:
            image_path: 影像文件路径
            bounds:     裁剪范围 "xmin,ymin,xmax,ymax"（像素坐标）
            output_name: 可选输出文件名标识
        """
        img = self._load_image(image_path)
        if img is None:
            return "裁剪失败，无法读取影像。"

        parts = [float(p.strip()) for p in bounds.replace(",", " ").split()]
        if len(parts) != 4:
            return "裁剪失败：bounds 需要 4 个数值 (xmin ymin xmax ymax)。"

        xmin, ymin, xmax, ymax = map(int, parts)
        h, w = img.shape[:2]

        # 边界检查
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        if xmin >= xmax or ymin >= ymax:
            return f"裁剪失败：无效范围 ({xmin},{ymin}) - ({xmax},{ymax})，影像大小 ({w},{h})。"

        clipped = img[ymin:ymax, xmin:xmax]
        tag = output_name or f"clip_{xmin}_{ymin}_{xmax}_{ymax}"
        save_info = self._save_output(
            clipped, tag, image_path,
            cmap_name="gray",
        )
        return (
            f"裁剪完成。原始大小: ({w},{h}) → 裁剪后: ({clipped.shape[1]},{clipped.shape[0]})\n"
            f"范围: ({xmin},{ymin}) - ({xmax},{ymax})\n"
            f"{save_info}"
        )

    # ── 提取波段 ─────────────────────────────────────

    def extract_bands(self, image_path: str, bands: str, output_name: Optional[str] = None) -> str:
        """从多波段影像中提取指定波段子集。

        参数:
            image_path: 影像文件路径
            bands:      要提取的波段列表（1-indexed），如 "4,3,2"（RGB）或 "8,4,3"
            output_name: 可选输出文件名标识
        """
        img = self._load_image(image_path)
        if img is None:
            return "提取波段失败，无法读取影像。"
        if img.ndim < 3:
            return "提取波段失败：影像无波段维度。"

        indices = []
        for s in bands.replace(",", " ").split():
            try:
                idx = int(s) - 1  # 转为 0-based
            except ValueError:
                return f"提取波段失败：无法解析 '{s}'，请用逗号或空格分隔波段号。"
            if idx < 0 or idx >= img.shape[2]:
                return f"提取波段失败：波段 {s} 超出范围 [1, {img.shape[2]}]。"
            indices.append(idx)

        if not indices:
            return "提取波段失败：未指定波段。"

        extracted = img[:, :, indices]
        n_out = len(indices)

        # 判断是否为 RGB 合成 (3 波段)
        is_rgb = n_out == 3
        tag = output_name or f"bands_{'_'.join(str(i+1) for i in indices)}"

        save_info = self._save_output(
            extracted, tag, image_path,
            cmap_name=None if is_rgb else "gray",
        )
        band_desc = ", ".join(f"Band {i+1}" for i in indices)
        return (
            f"波段提取完成。原始波段数: {img.shape[2]} → 输出波段: {n_out}\n"
            f"提取波段: {band_desc}\n"
            f"{save_info}"
        )

    # ── 输出保存 ─────────────────────────────────────

    def _save_output(
        self,
        array: np.ndarray,
        tool_name: str,
        image_path: str,
        cmap_name: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> str:
        """将结果数组保存为影像文件，返回保存路径文本。支持多波段。"""
        import datetime
        from pathlib import Path

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        input_stem = Path(image_path).stem
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{input_stem}_{tool_name}_{ts}"

        saved = []

        # ── GeoTIFF ──────────────────────────────────
        if self._src_profile is not None and rasterio is not None:
            if array.ndim == 3:
                n_bands = array.shape[2]
                data = np.transpose(array, (2, 0, 1))  # (H,W,C) → (C,H,W)
            else:
                n_bands = 1
                data = array[np.newaxis, :, :]

            profile = self._src_profile.copy()
            profile.update(count=n_bands, dtype=data.dtype.name, compress="lzw")
            # 清除不兼容的 nodata（原 dtype 与输出 dtype 不匹配时）
            nodata = profile.get("nodata")
            if nodata is not None:
                try:
                    np.array([nodata], dtype=data.dtype)
                except (ValueError, OverflowError):
                    profile["nodata"] = None
            tiff_path = out_dir / f"{base}.tif"
            with rasterio.open(tiff_path, "w", **profile) as dst:
                dst.write(data)
            saved.append(str(tiff_path))

        # ── PNG 可视化 ───────────────────────────────
        if cmap_name is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colorbar import ColorbarBase
            import matplotlib.colors as mcolors

            fig, ax = plt.subplots(figsize=(12, 10))

            if array.ndim == 3 and array.shape[2] == 3:
                rgb = np.clip(array / (vmax or np.max(array) or 1), 0, 1) if array.dtype.kind == 'f' else array
                if rgb.dtype != np.uint8:
                    rgb = (rgb / (rgb.max() or 1) * 255).astype(np.uint8)
                ax.imshow(rgb)
            else:
                arr_2d = array[:, :, 0] if array.ndim == 3 else array
                ax.imshow(arr_2d, cmap=cmap_name, vmin=vmin, vmax=vmax)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            png_path = out_dir / f"{base}.png"
            plt.savefig(png_path, dpi=150, pad_inches=0)
            plt.close(fig)
            saved.append(str(png_path))

            # 单独生成图例（仅对浮点连续数据，不用于分割/分类图）
            is_continuous = array.dtype.kind == 'f'
            if is_continuous and (array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 1)):
                norm = mcolors.Normalize(vmin=vmin or 0, vmax=vmax or 1)
                leg_fig, leg_ax = plt.subplots(figsize=(1.2, 6))
                ColorbarBase(leg_ax, cmap=cmap_name, norm=norm, orientation="vertical")
                leg_ax.set_title(tool_name.upper(), fontsize=10, pad=10)
                leg_fig.subplots_adjust(left=0.3, right=0.6, top=0.92, bottom=0.08)
                legend_path = out_dir / f"{base}_legend.png"
                leg_fig.savefig(legend_path, dpi=150)
                plt.close(leg_fig)
                saved.append(str(legend_path))

        if not saved:
            return "（输出未保存：无支持格式）"

        return "结果影像已保存至: " + "; ".join(saved)

    # ── 内部辅助方法 ──────────────────────────────────

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        self._src_profile = None
        self._band_descriptions = None
        path = Path(image_path)
        if not path.exists():
            return None

        # 用 rasterio 读取 GeoTIFF
        if rasterio and path.suffix.lower() in (".tif", ".tiff", ".vrt"):
            try:
                with rasterio.open(path) as src:
                    self._src_profile = src.profile
                    self._band_descriptions = src.descriptions
                    data = src.read()
                    if data.ndim == 3:
                        data = np.transpose(data, (1, 2, 0))
                    return data
            except Exception:
                pass

        # 用 OpenCV 读取常规格式
        if cv2:
            try:
                buf = np.fromfile(str(path), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img
            except Exception:
                pass

        return None

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> Optional[np.ndarray]:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] >= 3:
            return np.uint8(np.mean(image[:, :, :3], axis=2))
        return None

    @staticmethod
    def _to_rgb(image: np.ndarray) -> Optional[np.ndarray]:
        if image.ndim == 2:
            return cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2RGB) if cv2 else np.stack([image] * 3, axis=-1)
        if image.ndim == 3:
            if image.shape[2] >= 3:
                return image[:, :, :3].astype(np.uint8)
        return None

    WEIGHT_URLS = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }

    def _load_sam(self, model_type: str):
        """懒加载 SAM 模型（单例），权重缺失时自动下载。"""
        if self._sam_predictor is not None and self._sam_model_type == model_type:
            return self._sam_predictor

        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            return None

        url = self.WEIGHT_URLS.get(model_type)
        if not url:
            return None

        weight_name = url.rsplit("/", 1)[-1]
        weight_path = Path(__file__).parents[2] / "models" / "sam" / weight_name

        if not weight_path.exists():
            print(f"  [SAM] 正在下载 {model_type} 权重 ({weight_name})...")
            import urllib.request
            weight_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, str(weight_path))
                print(f"  [SAM] 下载完成")
            except Exception:
                return None

        sam = sam_model_registry[model_type](checkpoint=str(weight_path))
        predictor = SamPredictor(sam)
        self._sam_predictor = predictor
        self._sam_model_type = model_type
        return predictor
