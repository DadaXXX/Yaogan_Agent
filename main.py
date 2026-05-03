"""交互式对话入口 — 类似 ChatGPT/DeepSeek 的遥感分析助手。"""

# 运行方式：
# uvicorn app:app --host 0.0.0.0 --port 8000 之后浏览器打开 http://localhost:8000
# 或者直接 python main.py 进入命令行对话模式

import argparse
import re
import shutil
from pathlib import Path

from src.config import load_config
from src.conversation import ConversationManager
from src.llm_client import create_llm_client
from src.agent import Agent
from src.tools.registry import Tool, ToolRegistry
from src.tools.remote_sensing import RemoteSensingToolkit
from src.tools.preprocessing import PreprocessingToolkit
from src.tools.analysis import AnalysisToolkit
from src.tools.classification import ClassificationToolkit
from src.tools.geo_utils import GeoToolkit
from src.tools.gee_client import GEEClient


RE_IMAGE_FILE = re.compile(r'([\w\-]+\.(?:tif|tiff|vrt|png|jpg|jpeg))', re.IGNORECASE)


def find_image_in_text(text: str) -> str | None:
    """从文本中提取影像文件名，返回当前目录下的完整路径；不存在则返回 None。"""
    for m in RE_IMAGE_FILE.finditer(text):
        candidate = Path(m.group(1))
        if candidate.exists():
            return str(candidate.resolve())
    return None


def build_registry(
    toolkit: RemoteSensingToolkit,
    geo: "GeoToolkit",
    pre: "PreprocessingToolkit",
    analysis: "AnalysisToolkit",
    classify: "ClassificationToolkit",
    gee: "GEEClient",
) -> ToolRegistry:
    """注册所有遥感工具到 Registry。"""
    registry = ToolRegistry()

    registry.register(Tool(
        name="describe_image",
        description="读取影像基本信息（形状、数据类型、值范围）",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
            },
            "required": ["image_path"],
        },
        fn=toolkit.describe_image,
    ))
    registry.register(Tool(
        name="compute_ndvi",
        description="计算归一化植被指数（NDVI）",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "red_band": {"type": "integer", "description": "红波段索引（1-based，如不指定则自动推断）"},
                "nir_band": {"type": "integer", "description": "近红外波段索引（1-based，如不指定则自动推断）"},
            },
            "required": ["image_path"],
        },
        fn=toolkit.compute_ndvi,
    ))
    registry.register(Tool(
        name="compute_sar_backscatter",
        description="计算 SAR 背散射统计",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
            },
            "required": ["image_path"],
        },
        fn=toolkit.compute_sar_backscatter,
    ))
    registry.register(Tool(
        name="segment_otsu",
        description="对影像进行 Otsu 阈值分割",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
            },
            "required": ["image_path"],
        },
        fn=toolkit.segment_otsu,
    ))
    registry.register(Tool(
        name="segment_sam",
        description="使用 SAM 模型进行语义分割，生成影像掩码",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "model_type": {
                    "type": "string",
                    "description": "SAM 模型类型 (vit_b/vit_l/vit_h)",
                    "enum": ["vit_b", "vit_l", "vit_h"],
                },
            },
            "required": ["image_path"],
        },
        fn=toolkit.segment_sam,
    ))
    registry.register(Tool(
        name="classify_landcover",
        description="对影像进行土地覆被分类（KMeans 无监督聚类）",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "n_clusters": {
                    "type": "integer",
                    "description": "分类数量（默认 4）",
                },
            },
            "required": ["image_path"],
        },
        fn=toolkit.classify_landcover,
    ))
    registry.register(Tool(
        name="band_math",
        description="对影像波段执行算术表达式运算，如 (b4-b3)/(b4+b3)",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "expression": {"type": "string", "description": "运算表达式，b1/b2/... 表示波段，如 (b4-b3)/(b4+b3)"},
                "output_name": {"type": "string", "description": "可选输出文件名标识"},
            },
            "required": ["image_path", "expression"],
        },
        fn=toolkit.band_math,
    ))
    registry.register(Tool(
        name="compute_index",
        description="计算通用遥感指数（NDVI/LSWI/NDWI/EVI/SAVI/NBR/VARI），需指定波段映射",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "index": {"type": "string", "description": "指数名（NDVI/LSWI/NDWI/MNDWI/EVI/SAVI/NBR/VARI）"},
                "band_map": {"type": "string", "description": "波段映射，如 R=3,NIR=4 或 NIR=5,SWIR=6"},
                "output_name": {"type": "string", "description": "可选输出文件名标识"},
            },
            "required": ["image_path", "index", "band_map"],
        },
        fn=toolkit.compute_index,
    ))
    registry.register(Tool(
        name="clip_raster",
        description="按像素坐标裁剪影像",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "bounds": {"type": "string", "description": "裁剪范围 (xmin ymin xmax ymax)，如 100 100 500 500"},
                "output_name": {"type": "string", "description": "可选输出文件名标识"},
            },
            "required": ["image_path", "bounds"],
        },
        fn=toolkit.clip_raster,
    ))
    registry.register(Tool(
        name="extract_bands",
        description="从多波段影像中提取指定波段子集",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "bands": {"type": "string", "description": "波段列表（1-based），如 4,3,2 或 8,4,3"},
                "output_name": {"type": "string", "description": "可选输出文件名标识"},
            },
            "required": ["image_path", "bands"],
        },
        fn=toolkit.extract_bands,
    ))

    # ── 预处理工具 ──────────────────────────────────
    registry.register(Tool(
        name="cloud_mask",
        description="利用 QA 波段生成云掩膜，去除云和云影",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "sensor": {"type": "string", "description": "传感器类型 (sentinel2/landsat8/auto)"},
            },
            "required": ["image_path"],
        },
        fn=pre.cloud_mask,
    ))
    registry.register(Tool(
        name="normalize_image",
        description="影像归一化 (minmax 或 zscore)",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "method": {"type": "string", "description": "归一化方法 (minmax/zscore)"},
                "bands": {"type": "string", "description": "指定波段，如 1,2,3"},
            },
            "required": ["image_path"],
        },
        fn=pre.normalize_image,
    ))
    registry.register(Tool(
        name="histogram_match",
        description="将影像直方图匹配到参考影像（多时相分析）",
        parameters={
            "type": "object",
            "properties": {
                "source_path": {"type": "string", "description": "源影像路径"},
                "reference_path": {"type": "string", "description": "参考影像路径"},
            },
            "required": ["source_path", "reference_path"],
        },
        fn=pre.histogram_match,
    ))
    registry.register(Tool(
        name="resample_raster",
        description="重采样影像到目标分辨率",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "target_resolution": {"type": "string", "description": "目标分辨率 (m)，如 30"},
            },
            "required": ["image_path", "target_resolution"],
        },
        fn=pre.resample_raster,
    ))

    # ── 分析工具 ────────────────────────────────────
    registry.register(Tool(
        name="change_detection",
        description="两期影像变化检测 (difference/ratio/cva)",
        parameters={
            "type": "object",
            "properties": {
                "image_before": {"type": "string", "description": "前期影像路径"},
                "image_after": {"type": "string", "description": "后期影像路径"},
                "method": {"type": "string", "description": "检测方法 (difference/ratio/cva)"},
            },
            "required": ["image_before", "image_after"],
        },
        fn=analysis.change_detection,
    ))
    registry.register(Tool(
        name="pca_transform",
        description="PCA 降维变换，输出前 N 个主成分",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "n_components": {"type": "string", "description": "保留的主成分数量，如 3"},
            },
            "required": ["image_path"],
        },
        fn=analysis.pca_transform,
    ))
    registry.register(Tool(
        name="spectral_profile",
        description="提取指定像素的光谱曲线",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "x": {"type": "string", "description": "像素 X 坐标"},
                "y": {"type": "string", "description": "像素 Y 坐标"},
            },
            "required": ["image_path", "x", "y"],
        },
        fn=analysis.spectral_profile,
    ))
    registry.register(Tool(
        name="compute_statistics",
        description="计算影像详细统计信息（含直方图特征）",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "bands": {"type": "string", "description": "指定波段，如 1,2,3"},
            },
            "required": ["image_path"],
        },
        fn=analysis.compute_statistics,
    ))

    # ── 分类工具 ────────────────────────────────────
    registry.register(Tool(
        name="train_classifier",
        description="从样本点训练监督分类器 (Random Forest/SVM)",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "samples_csv": {"type": "string", "description": "样本 CSV 文件 (x,y,label)"},
                "classifier": {"type": "string", "description": "分类器类型 (rf/svm)"},
            },
            "required": ["image_path", "samples_csv"],
        },
        fn=classify.train_classifier,
    ))
    registry.register(Tool(
        name="predict_classify",
        description="用已训练的分类器对影像分类",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
            },
            "required": ["image_path"],
        },
        fn=classify.predict_classify,
    ))
    registry.register(Tool(
        name="accuracy_assessment",
        description="精度评价：混淆矩阵、OA、Kappa、F1",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "validation_csv": {"type": "string", "description": "验证样本 CSV"},
            },
            "required": ["image_path", "validation_csv"],
        },
        fn=classify.accuracy_assessment,
    ))

    # ── 地理空间工具 ────────────────────────────────
    registry.register(Tool(
        name="clip_by_geo",
        description="按地理坐标（经纬度）裁剪影像",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "bounds": {"type": "string", "description": "经纬度范围: lon_min lat_min lon_max lat_max"},
            },
            "required": ["image_path", "bounds"],
        },
        fn=geo.clip_by_geo,
    ))
    registry.register(Tool(
        name="reproject_raster",
        description="重投影影像到目标坐标系",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "target_epsg": {"type": "string", "description": "目标 EPSG 代码，如 4326"},
                "resolution": {"type": "string", "description": "可选目标分辨率"},
            },
            "required": ["image_path", "target_epsg"],
        },
        fn=geo.reproject_raster,
    ))
    registry.register(Tool(
        name="zonal_statistics",
        description="按矢量区域计算分区统计",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "geojson_path": {"type": "string", "description": "矢量文件路径 (GeoJSON/shp)"},
            },
            "required": ["image_path", "geojson_path"],
        },
        fn=geo.zonal_statistics,
    ))
    registry.register(Tool(
        name="extract_roi",
        description="按矢量边界提取 ROI 影像",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "影像文件路径"},
                "geojson_path": {"type": "string", "description": "矢量文件路径 (GeoJSON/shp)"},
            },
            "required": ["image_path", "geojson_path"],
        },
        fn=geo.extract_roi,
    ))

    # ── GEE 数据获取 ────────────────────────────────
    registry.register(Tool(
        name="search_gee_images",
        description="搜索 GEE 影像（Sentinel-2/Landsat/MODIS/DEM 等）",
        parameters={
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "数据集 (sentinel2/landsat8/srtm/modis_ndvi)"},
                "lon": {"type": "string", "description": "中心点经度"},
                "lat": {"type": "string", "description": "中心点纬度"},
                "buffer_km": {"type": "string", "description": "搜索半径 km"},
                "start_date": {"type": "string", "description": "起始日期 YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                "cloud_cover": {"type": "string", "description": "最大云量百分比"},
            },
            "required": ["collection", "lon", "lat"],
        },
        fn=gee.search_images,
    ))
    registry.register(Tool(
        name="download_gee_image",
        description="下载 GEE 影像到本地",
        parameters={
            "type": "object",
            "properties": {
                "collection": {"type": "string", "description": "数据集 (sentinel2/landsat8/srtm)"},
                "image_id": {"type": "string", "description": "影像 ID（从 search 结果获取）"},
                "lon": {"type": "string", "description": "中心点经度"},
                "lat": {"type": "string", "description": "中心点纬度"},
                "buffer_km": {"type": "string", "description": "下载半径 km"},
                "bands": {"type": "string", "description": "波段列表，如 B4,B3,B2,B8"},
                "scale": {"type": "string", "description": "分辨率 m"},
            },
            "required": ["collection", "image_id", "lon", "lat"],
        },
        fn=gee.download_image,
    ))

    return registry


def ensure_output_dir_tool(registry: ToolRegistry, setter_fn) -> None:
    """注册 output_dir 控制工具（在 main() 运行时动态注入）"""
    registry.register(Tool(
        name="set_output_dir",
        description="用户指定结果保存目录时调用，更改所有工具的输出路径",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "输出目录路径，如 ./results 或 ./ndvi_output"},
            },
            "required": ["path"],
        },
        fn=setter_fn,
    ))


def print_header():
    cols, _ = shutil.get_terminal_size()
    print("=" * cols)
    print("  遥感分析助手 v2.0 — 交互式对话模式".center(cols))
    print("=" * cols)
    print()
    print("  可用命令:")
    print("    /clear         清除对话历史")
    print("    /help          显示帮助")
    print("    /exit          退出")
    print()
    print("  直接输入问题，例如:")
    print('    "帮我用 GeoNDC_N22E099_20251124.tif 计算 NDVI"')
    print('    "提取波段 4,3,2 保存为 RGB"')
    print('    "波段运算 (b4 - b3) / (b4 + b3)"')
    print()


def print_help():
    print()
    print("  对话式遥感分析助手")
    print("  ──────────────────")
    print("  使用方法：")
    print("    直接输入问题，助手会自动识别你提到的影像文件，例如：")
    print('       "帮我用 GeoNDC_N22E099_20251124.tif 计算 NDVI"')
    print('       "搜索北京 Landsat8 影像"')
    print('       "对两期影像做变化检测"')
    print('       "PCA 降维到 3 个主成分"')
    print()
    print("  2. 支持 GEE 数据获取 (sentinel2/landsat8/srtm/modis...)")
    print("  3. 支持变化检测、PCA、监督分类、精度评价")
    print("  4. 分析结果自动保存到输出目录")
    print()


def main():
    parser = argparse.ArgumentParser(description="遥感分析助手 — 交互式对话")
    parser.add_argument("--output-dir", default=None, help="结果输出目录（默认 ./output）")
    args = parser.parse_args()

    config = load_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    # 初始化各模块
    toolkit = RemoteSensingToolkit(output_dir=config.output_dir)
    geo = GeoToolkit(output_dir=config.output_dir)
    pre = PreprocessingToolkit(output_dir=config.output_dir)
    analysis = AnalysisToolkit(output_dir=config.output_dir)
    classify = ClassificationToolkit(output_dir=config.output_dir)
    gee = GEEClient(download_dir=config.output_dir)
    _all_modules = [toolkit, geo, pre, analysis, classify, gee]

    def set_output_dir(path: str) -> str:
        """动态修改所有工具的输出目录"""
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        for m in _all_modules:
            m.output_dir = str(p)
        return f"输出目录已切换至: {p}"

    registry = build_registry(toolkit, geo, pre, analysis, classify, gee)
    ensure_output_dir_tool(registry, set_output_dir)
    llm = create_llm_client(
        config.llm_provider,
        config.api_key,
        config.llm_model,
        config.api_base_url,
    )
    conv_mgr = ConversationManager()
    agent = Agent(llm, registry, conv_mgr, config)

    current_image = None
    print_header()

    while True:
        try:
            raw = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("再见！")
            break

        if not raw:
            continue

        # ── 内置命令 ──────────────────────────────────
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/exit":
                print("再见！")
                break

            elif cmd == "/help":
                print_help()
                continue

            elif cmd == "/clear":
                conv_mgr.clear()
                current_image = None
                print("  对话历史已清除")
                continue

            else:
                print(f"  未知命令: {cmd}，输入 /help 查看帮助")
                continue

        # ── 从对话中自动识别影像文件 ──────────────────
        if not current_image:
            found = find_image_in_text(raw)
            if found:
                test_info = toolkit.describe_image(found)
                if not test_info.startswith("无法读取"):
                    current_image = found
                    print(f"  [自动加载影像] {current_image}")
                    print(f"  {test_info.split(chr(10))[0]}")
                else:
                    print(f"  找到影像文件，但无法读取 — 请检查依赖是否安装（pip install -r requirements.txt）")
                    print(f"  {test_info}")
                    continue
            else:
                print("  未检测到影像文件，请在问题中指明影像文件名（如 GeoNDC_N22E099_20251124.tif）")
                continue

        # 如果用户问题中提到了新的影像，自动切换
        found = find_image_in_text(raw)
        if found and found != current_image:
            test_info = toolkit.describe_image(found)
            if not test_info.startswith("无法读取"):
                current_image = found
                print(f"  [切换到影像] {current_image}")
                print(f"  {test_info.split(chr(10))[0]}")
            else:
                print(f"  找到影像文件，但无法读取: {test_info}")
                continue

        # 将用户问题附加上当前影像上下文
        full_input = f"当前影像: {current_image}\n用户问题: {raw}"
        result = agent.run(full_input)
        print(f"\n助手 > {result}\n")


if __name__ == "__main__":
    main()
