"""Google Earth Engine 数据下载客户端。

使用方法:
1. 首次使用自动提示认证
2. 搜索 → 筛选 → 导出 → 下载
3. 支持 Sentinel-2， Landsat 8/9， MODIS， DEM 等

GEE 安装要求: pip install earthengine-api
"""

import json
import os
from pathlib import Path
from typing import Optional


class GEEClient:
    """Google Earth Engine 下载助手"""

    # 常用数据集
    COLLECTIONS = {
        "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
        "sentinel2_toa": "COPERNICUS/S2_HARMONIZED",
        "landsat8": "LANDSAT/LC08/C02/T1_L2",
        "landsat9": "LANDSAT/LC09/C02/T1_L2",
        "landsat8_toa": "LANDSAT/LC08/C02/T1_TOA",
        "landsat7": "LANDSAT/LE07/C02/T1_L2",
        "landsat5": "LANDSAT/LT05/C02/T1_L2",
        "modis_ndvi": "MODIS/061/MOD13Q1",
        "modis_lst": "MODIS/061/MOD11A2",
        "srtm": "USGS/SRTMGL1_003",
        "alos_dem": "JAXA/ALOS/AW3D30/V3_2",
        "dynamic_world": "GOOGLE/DYNAMICWORLD/V1",
    }

    # Sentinel-2 可用波段
    S2_BANDS = {
        "B1": "B1", "B2": "B2", "B3": "B3", "B4": "B4",
        "B5": "B5", "B6": "B6", "B7": "B7", "B8": "B8",
        "B8A": "B8A", "B9": "B9", "B11": "B11", "B12": "B12",
    }

    # Landsat 8/9 SR 波段
    L8_SR_BANDS = {
        "SR_B1": "SR_B1", "SR_B2": "SR_B2", "SR_B3": "SR_B3",
        "SR_B4": "SR_B4", "SR_B5": "SR_B5", "SR_B6": "SR_B6",
        "SR_B7": "SR_B7",
    }

    def __init__(self, project_id: str = "", download_dir: str = "./gee_downloads"):
        self.project_id = project_id
        self.download_dir = download_dir
        self._initialized = False
        self._ee = None

    def _init_gee(self) -> str:
        """初始化 GEE 连接，首次需要认证。返回空字符串表示成功，否则返回错误信息。"""
        if self._initialized and self._ee:
            return ""

        try:
            import ee
            self._ee = ee
        except ImportError:
            return "GEE 未安装。请运行: pip install earthengine-api\n认证: earthengine authenticate"

        try:
            self._ee.Initialize(project=self.project_id or None)
            self._initialized = True
            return ""
        except self._ee.ee_exception.EEException:
            return (
                "GEE 未认证。请在「设置」面板中进行认证，"
                "或在终端运行: earthengine authenticate"
            )
        except Exception as e:
            return f"GEE 初始化失败: {e}"

    def authenticate(self, project_id: str = "") -> dict:
        """触发 GEE 认证流程。返回认证 URL 或状态信息。"""
        try:
            import ee
        except ImportError:
            return {"error": "GEE 未安装。请运行: pip install earthengine-api"}

        try:
            # Try to initialize first — if it works, already authenticated
            ee.Initialize(project=project_id or None)
            self._initialized = True
            self._ee = ee
            self.project_id = project_id
            return {"status": "already_authenticated", "message": "GEE 已认证，无需重复认证"}
        except Exception:
            pass

        try:
            # Trigger auth flow — generates a URL for the user to visit
            ee.Authenticate(auth_mode="notebook")
            ee.Initialize(project=project_id or None)
            self._initialized = True
            self._ee = ee
            self.project_id = project_id
            return {"status": "success", "message": "GEE 认证成功！"}
        except Exception as e:
            return {"error": f"认证失败: {e}"}

    def check_auth(self) -> dict:
        """检查 GEE 认证状态。"""
        try:
            import ee
        except ImportError:
            return {"authenticated": False, "error": "earthengine-api 未安装"}

        try:
            ee.Initialize(project=self.project_id or None)
            return {"authenticated": True, "project_id": self.project_id or "(默认项目)"}
        except Exception:
            return {"authenticated": False, "error": "未认证或项目无效"}

    def search_images(
        self,
        collection: str,
        lon: str,
        lat: str,
        buffer_km: str = "20",
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        cloud_cover: str = "10",
        max_results: str = "10",
    ) -> str:
        """搜索符合条件的 GEE 影像。

        collection:  数据集名称 (sentinel2 / landsat8 / srtm / modis_ndvi 等)
        lon, lat:    中心点经纬度
        buffer_km:   搜索半径 (km)
        start_date / end_date: 时间范围 (YYYY-MM-DD)
        cloud_cover: 最大云量百分比 (仅光学影像)
        max_results: 最多返回结果数
        """
        err = self._init_gee()
        if err:
            return err

        ee = self._ee

        coll_name = self.COLLECTIONS.get(collection.lower(), collection)
        if coll_name != collection:
            collection = collection.lower()

        try:
            point = ee.Geometry.Point([float(lon), float(lat)])
            region = point.buffer(float(buffer_km) * 1000)

            img_coll = ee.ImageCollection(coll_name)
            img_coll = img_coll.filterBounds(region)
            img_coll = img_coll.filterDate(start_date, end_date)

            if collection in ("sentinel2", "sentinel2_toa", "landsat8", "landsat9"):
                img_coll = img_coll.filter(ee.Filter.lte("CLOUD_COVER", float(cloud_cover)))

            img_list = img_coll.toList(int(max_results))
            n_imgs = img_list.length().getInfo()

            if n_imgs == 0:
                return "未找到满足条件的影像，请调整搜索参数。"

            lines = [f"搜索 {coll_name}，找到 {n_imgs} 景影像:"]
            for i in range(n_imgs):
                img = ee.Image(img_list.get(i))
                img_id = img.get("system:index").getInfo()
                date = img.get("DATE_ACQUIRED").getInfo() or img.get("system:time_start").getInfo()
                cc = img.get("CLOUD_COVER").getInfo() if collection in ("sentinel2", "landsat8", "landsat9") else "N/A"
                if isinstance(cc, (int, float)):
                    cc = f"{cc:.1f}%"

                lines.append(f"  {i + 1}. {img_id} | 日期: {date} | 云量: {cc}")

            lines.append("\n使用 download_gee_image 下载指定影像，格式: collection,image_id")
            return "\n".join(lines)

        except Exception as e:
            return f"搜索失败: {e}"

    def download_image(
        self,
        collection: str,
        image_id: str,
        lon: str,
        lat: str,
        buffer_km: str = "20",
        bands: Optional[str] = None,
        scale: str = "10",
        filename: Optional[str] = None,
    ) -> str:
        """下载单景 GEE 影像到本地。

        collection: 数据集名称
        image_id:   影像 ID（从 search_images 结果获取）
        lon, lat:   中心点经纬度
        buffer_km:  下载范围半径 (km)
        bands:      要下载的波段，如 "B4,B3,B2,B8"，默认全波段
        scale:      输出分辨率 (m)
        filename:   输出文件名（不含扩展名）
        """
        err = self._init_gee()
        if err:
            return err

        ee = self._ee

        coll_name = self.COLLECTIONS.get(collection.lower(), collection)

        try:
            point = ee.Geometry.Point([float(lon), float(lat)])
            region = point.buffer(float(buffer_km) * 1000)

            img = ee.Image(f"{coll_name}/{image_id}")
            sel_bands = [b.strip() for b in bands.replace(",", " ").split()] if bands else None

            if sel_bands:
                img = img.select(sel_bands)

            out_dir = Path(self.download_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = filename or f"{collection}_{image_id}"
            tiff_path = out_dir / f"{out_file}.tif"

            print(f"  [GEE] 正在导出 {image_id} 到 Google Drive...")
            print(f"  [GEE] 文件将保存在 Google Drive 根目录，下载后移到本地。")

            task = ee.batch.Export.image.toDrive(
                image=img,
                description=out_file[:50],
                folder="gee_exports",
                scale=float(scale),
                region=region,
                maxPixels=1e9,
                fileFormat="GeoTIFF",
            )
            task.start()

            # 生成 Python 脚本供用户单独运行（异步导出需要等待）
            script_path = out_dir / f"{out_file}_download.py"
            script = f'''"""GEE 下载脚本 — 在终端运行以完成下载"""
import ee
ee.Initialize()

task_list = ee.batch.Task.list()
print("GEE 任务状态:")
for t in task_list:
    status = t.status()
    print(f"  {{status['description']}}: {{status['state']}}")
    if status['state'] == 'COMPLETED':
        print(f"    → 前往 Google Drive 'gee_exports' 文件夹下载文件")
'''
            script_path.write_text(script)

            return (
                f"GEE 导出任务已提交。\n"
                f"  影像: {coll_name}/{image_id}\n"
                f"  范围: [{lon},{lat}] 半径 {buffer_km}km\n"
                f"  分辨率: {scale}m\n"
                f"\n导出是异步的（通常 2-10 分钟完成）。\n"
                f"1. 打开 Google Drive → gee_exports 文件夹查看进度\n"
                f"2. 下载完成后将文件放到 {self.download_dir} 目录\n"
                f"3. 或运行以下 Python 脚本检查任务状态:\n"
                f"   python {script_path}\n"
                f"\nGEE 任务页面: https://code.earthengine.google.com/tasks\n"
                "提示: 也可直接用 earthengine 命令行: earthengine task list"
            )

        except Exception as e:
            return f"下载失败: {e}"

    def get_collection_info(self, collection: str) -> str:
        """获取数据集的基本信息"""
        coll_name = self.COLLECTIONS.get(collection.lower(), collection)
        info = self.COLLECTIONS.get(collection.lower())

        details = {
            "sentinel2": "Sentinel-2 Level-2A 地表反射率 (ESA)\n  波段: B1~B12\n  分辨率: 10m/20m/60m\n  时间: 2017-至今",
            "landsat8": "Landsat 8 Collection 2 Tier 1 Level-2 SR\n  波段: SR_B1~SR_B7\n  分辨率: 30m\n  时间: 2013-至今",
            "landsat9": "Landsat 9 Collection 2 Tier 1 Level-2 SR\n  波段: SR_B1~SR_B7\n  分辨率: 30m\n  时间: 2021-至今",
            "srtm": "SRTM Digital Elevation 30m\n  波段: elevation\n  分辨率: 30m",
            "modis_ndvi": "MODIS NDVI 16天合成 (MOD13Q1)\n  波段: NDVI, EVI, VI_Quality\n  分辨率: 250m\n  时间: 2000-至今",
            "dynamic_world": "Dynamic World 10m 土地利用 (Google)\n  波段: 9个地类概率\n  分辨率: 10m\n  时间: 2015-至今",
        }

        desc = details.get(collection, f"数据集: {coll_name}")
        return f"{desc}\n\nGEE ID: {coll_name}\n\n可用数据集列表: {', '.join(self.COLLECTIONS.keys())}"

    # ── JSON 搜索（供 API 使用）─────────────────────────

    def search_images_json(
        self,
        collection: str,
        lon: float,
        lat: float,
        buffer_km: float = 20,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        cloud_cover: float = 10,
        max_results: int = 50,
    ) -> dict:
        """搜索 GEE 影像，返回结构化 dict。

        与 search_images() 功能相同，但返回 JSON 可序列化的 dict。
        """
        err = self._init_gee()
        if err:
            return {"error": err, "images": [], "count": 0}

        ee = self._ee
        coll_name = self.COLLECTIONS.get(collection.lower(), collection)
        is_optical = collection.lower() in (
            "sentinel2", "sentinel2_toa", "landsat8", "landsat9", "landsat8_toa",
        )

        try:
            point = ee.Geometry.Point([float(lon), float(lat)])
            region = point.buffer(float(buffer_km) * 1000)

            img_coll = ee.ImageCollection(coll_name)
            img_coll = img_coll.filterBounds(region)
            img_coll = img_coll.filterDate(start_date, end_date)

            # Cloud cover filter — property name differs by collection
            if is_optical:
                if collection.lower().startswith("sentinel"):
                    cc_prop = "CLOUDY_PIXEL_PERCENTAGE"
                else:
                    cc_prop = "CLOUD_COVER"
                img_coll = img_coll.filter(ee.Filter.lte(cc_prop, float(cloud_cover)))

            img_list = img_coll.toList(int(max_results))
            n_imgs = img_list.length().getInfo()

            images = []
            for i in range(n_imgs):
                img = ee.Image(img_list.get(i))
                img_id = img.get("system:index").getInfo()

                # Date — different property names
                date = ""
                for prop in ["DATE_ACQUIRED", "SENSING_TIME", "system:time_start"]:
                    val = img.get(prop).getInfo()
                    if val:
                        if prop == "system:time_start" and isinstance(val, (int, float)):
                            import datetime
                            date = datetime.datetime.utcfromtimestamp(val / 1000).strftime("%Y-%m-%d")
                        else:
                            date = str(val)[:10]
                        break

                # Cloud cover
                cc = None
                if is_optical:
                    for prop in ["CLOUD_COVER", "CLOUDY_PIXEL_PERCENTAGE"]:
                        val = img.get(prop).getInfo()
                        if isinstance(val, (int, float)):
                            cc = round(val, 1)
                            break

                # Get image footprint geometry for map display
                try:
                    geom = img.geometry().bounds().getInfo()
                    coords = geom.get("coordinates", [[]])[0]
                except Exception:
                    coords = []

                images.append({
                    "id": img_id,
                    "date": date,
                    "cloud_cover": cc,
                    "bounds": coords,  # [[lon,lat], ...] polygon
                })

            return {
                "collection": coll_name,
                "collection_alias": collection.lower(),
                "count": len(images),
                "images": images,
                "region": {"lon": lon, "lat": lat, "buffer_km": buffer_km},
            }

        except Exception as e:
            return {"error": str(e), "images": [], "count": 0}

    # ── 批量下载 ────────────────────────────────────────

    def batch_download(
        self,
        collection: str,
        image_ids: list,
        lon: float,
        lat: float,
        buffer_km: float = 20,
        bands: str = "",
        scale: str = "10",
        cloud_mask: bool = False,
        add_ndvi: bool = False,
    ) -> list:
        """批量下载多景影像到 Google Drive。

        返回任务列表 [{image_id, task_id, status}]。
        """
        err = self._init_gee()
        if err:
            return [{"error": err}]

        ee = self._ee
        coll_name = self.COLLECTIONS.get(collection.lower(), collection)

        try:
            point = ee.Geometry.Point([float(lon), float(lat)])
            region = point.buffer(float(buffer_km) * 1000)

            tasks = []
            for img_id in image_ids:
                img = ee.Image(f"{coll_name}/{img_id}")

                # 选波段
                sel_bands = [b.strip() for b in bands.replace(",", " ").split()] if bands else None
                if sel_bands:
                    img = img.select(sel_bands)

                # 去云（光学影像）
                if cloud_mask and collection.lower() in (
                    "sentinel2", "sentinel2_toa", "landsat8", "landsat9",
                ):
                    if collection.lower().startswith("sentinel"):
                        qa = img.select("QA60")
                        cloud_bit = 1 << 10
                        shadow_bit = 1 << 11
                        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
                            qa.bitwiseAnd(shadow_bit).eq(0)
                        )
                        img = img.updateMask(mask)
                    elif collection.lower().startswith("landsat"):
                        qa = img.select("QA_PIXEL")
                        cloud_bit = 1 << 3
                        shadow_bit = 1 << 4
                        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
                            qa.bitwiseAnd(shadow_bit).eq(0)
                        )
                        img = img.updateMask(mask)

                # 添加 NDVI
                if add_ndvi:
                    if collection.lower().startswith("sentinel"):
                        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
                        img = img.addBands(ndvi)
                    elif collection.lower().startswith("landsat"):
                        ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
                        img = img.addBands(ndvi)

                # Cast all bands to Float32 to avoid type mismatch errors
                img = img.toFloat()

                desc = f"{collection}_{img_id}"[:50]
                task = ee.batch.Export.image.toDrive(
                    image=img,
                    description=desc,
                    folder="gee_exports",
                    scale=float(scale),
                    region=region,
                    maxPixels=1e9,
                    fileFormat="GeoTIFF",
                )
                task.start()
                tasks.append({
                    "image_id": img_id,
                    "task_id": task.id,
                    "status": "SUBMITTED",
                })

            return tasks

        except Exception as e:
            return [{"error": str(e)}]

    # ── 用 GeoJSON geometry 搜索 ────────────────────────

    def search_with_geometry(
        self,
        collection: str,
        geometry: dict,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        cloud_cover: float = 10,
        max_results: int = 50,
    ) -> dict:
        """用 GeoJSON geometry 搜索影像（用于 SHP 选区）。

        geometry: GeoJSON geometry dict (type: Polygon/MultiPolygon/Point)
        """
        err = self._init_gee()
        if err:
            return {"error": err, "images": [], "count": 0}

        ee = self._ee
        coll_name = self.COLLECTIONS.get(collection.lower(), collection)
        is_optical = collection.lower() in (
            "sentinel2", "sentinel2_toa", "landsat8", "landsat9", "landsat8_toa",
        )

        try:
            ee_geom = ee.Geometry(geometry)
            img_coll = ee.ImageCollection(coll_name)
            img_coll = img_coll.filterBounds(ee_geom)
            img_coll = img_coll.filterDate(start_date, end_date)

            if is_optical:
                if collection.lower().startswith("sentinel"):
                    cc_prop = "CLOUDY_PIXEL_PERCENTAGE"
                else:
                    cc_prop = "CLOUD_COVER"
                img_coll = img_coll.filter(ee.Filter.lte(cc_prop, float(cloud_cover)))

            img_list = img_coll.toList(int(max_results))
            n_imgs = img_list.length().getInfo()

            images = []
            for i in range(n_imgs):
                img = ee.Image(img_list.get(i))
                img_id = img.get("system:index").getInfo()

                date = ""
                for prop in ["DATE_ACQUIRED", "SENSING_TIME", "system:time_start"]:
                    val = img.get(prop).getInfo()
                    if val:
                        if prop == "system:time_start" and isinstance(val, (int, float)):
                            import datetime
                            date = datetime.datetime.utcfromtimestamp(val / 1000).strftime("%Y-%m-%d")
                        else:
                            date = str(val)[:10]
                        break

                cc = None
                if is_optical:
                    for prop in ["CLOUD_COVER", "CLOUDY_PIXEL_PERCENTAGE"]:
                        val = img.get(prop).getInfo()
                        if isinstance(val, (int, float)):
                            cc = round(val, 1)
                            break

                try:
                    geom = img.geometry().bounds().getInfo()
                    coords = geom.get("coordinates", [[]])[0]
                except Exception:
                    coords = []

                images.append({"id": img_id, "date": date, "cloud_cover": cc, "bounds": coords})

            return {"collection": coll_name, "count": len(images), "images": images}

        except Exception as e:
            return {"error": str(e), "images": [], "count": 0}

    # ── 时间合成 ────────────────────────────────────────

    def composite_by_period(
        self,
        collection: str,
        lon: float,
        lat: float,
        buffer_km: float = 20,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        cloud_cover: float = 10,
        period_days: int = 16,
        bands: str = "",
        scale: str = "10",
        cloud_mask: bool = False,
        add_ndvi: bool = False,
    ) -> list:
        """按时间窗口合成影像并批量导出。

        period_days: 合成周期（天），如 8、16、30
        返回任务列表。
        """
        err = self._init_gee()
        if err:
            return [{"error": err}]

        ee = self._ee
        coll_name = self.COLLECTIONS.get(collection.lower(), collection)
        is_optical = collection.lower() in (
            "sentinel2", "sentinel2_toa", "landsat8", "landsat9", "landsat8_toa",
        )

        try:
            point = ee.Geometry.Point([float(lon), float(lat)])
            region = point.buffer(float(buffer_km) * 1000)

            img_coll = ee.ImageCollection(coll_name)
            img_coll = img_coll.filterBounds(region)
            img_coll = img_coll.filterDate(start_date, end_date)

            if is_optical:
                if collection.lower().startswith("sentinel"):
                    cc_prop = "CLOUDY_PIXEL_PERCENTAGE"
                else:
                    cc_prop = "CLOUD_COVER"
                img_coll = img_coll.filter(ee.Filter.lte(cc_prop, float(cloud_cover)))

            # Generate time windows
            from datetime import datetime, timedelta
            sd = datetime.strptime(start_date, "%Y-%m-%d")
            ed = datetime.strptime(end_date, "%Y-%m-%d")

            windows = []
            current = sd
            while current < ed:
                window_end = min(current + timedelta(days=period_days), ed)
                windows.append((current.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
                current = window_end

            # Select bands
            sel_bands = [b.strip() for b in bands.replace(",", " ").split()] if bands else None

            tasks = []
            for idx, (ws, we) in enumerate(windows):
                subset = img_coll.filterDate(ws, we)
                count = subset.size().getInfo()
                if count == 0:
                    continue

                # Median composite
                composite = subset.median()
                if sel_bands:
                    composite = composite.select(sel_bands)

                # Cloud mask
                if cloud_mask and is_optical:
                    if collection.lower().startswith("sentinel"):
                        qa = subset.select("QA60").median()
                        cloud_bit = 1 << 10
                        shadow_bit = 1 << 11
                        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
                            qa.bitwiseAnd(shadow_bit).eq(0)
                        )
                        composite = composite.updateMask(mask)
                    elif collection.lower().startswith("landsat"):
                        qa = subset.select("QA_PIXEL").median()
                        cloud_bit = 1 << 3
                        shadow_bit = 1 << 4
                        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
                            qa.bitwiseAnd(shadow_bit).eq(0)
                        )
                        composite = composite.updateMask(mask)

                # Add NDVI
                if add_ndvi:
                    if collection.lower().startswith("sentinel"):
                        ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
                        composite = composite.addBands(ndvi)
                    elif collection.lower().startswith("landsat"):
                        ndvi = composite.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
                        composite = composite.addBands(ndvi)

                composite = composite.toFloat()

                desc = f"{collection}_{ws[:10]}_{idx+1}"[:50]
                task = ee.batch.Export.image.toDrive(
                    image=composite,
                    description=desc,
                    folder="gee_exports",
                    scale=float(scale),
                    region=region,
                    maxPixels=1e9,
                    fileFormat="GeoTIFF",
                )
                task.start()
                tasks.append({
                    "image_id": f"{ws[:10]}~{we[:10]} ({count}景合成)",
                    "task_id": task.id,
                    "status": "SUBMITTED",
                })

            return tasks

        except Exception as e:
            return [{"error": str(e)}]

    # ── 获取影像瓦片 URL ────────────────────────────────

    def get_image_tile(self, collection: str, image_id: str, vis_params: dict = None) -> dict:
        """获取单景影像的 MapID（瓦片 URL），用于在 Leaflet 上显示。

        返回 {tile_url, map_id, token}。
        """
        err = self._init_gee()
        if err:
            return {"error": err}

        ee = self._ee
        coll_name = self.COLLECTIONS.get(collection.lower(), collection)

        try:
            img = ee.Image(f"{coll_name}/{image_id}")

            if vis_params is None:
                # Auto visualization
                if collection.lower().startswith("sentinel"):
                    vis_params = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
                elif collection.lower().startswith("landsat"):
                    vis_params = {"bands": ["SR_B4", "SR_B3", "SR_B2"], "min": 0, "max": 0.3}
                elif collection.lower() in ("modis_ndvi",):
                    vis_params = {"bands": ["NDVI"], "min": 0, "max": 9000, "palette": ["red", "yellow", "green"]}
                else:
                    vis_params = {"min": 0, "max": 1}

            map_id = img.getMapId(vis_params)
            tile_url = map_id["tile_fetcher"].url_format

            return {
                "tile_url": tile_url,
                "map_id": str(map_id.get("mapid", "")),
                "token": str(map_id.get("token", "")),
            }

        except Exception as e:
            return {"error": str(e)}

    # ── 查询任务状态 ────────────────────────────────────

    def get_tasks(self) -> list:
        """查询所有 GEE 任务状态。"""
        err = self._init_gee()
        if err:
            return [{"error": err}]

        try:
            ee = self._ee
            task_list = ee.batch.Task.list()
            results = []
            for t in task_list[:20]:  # 最多返回 20 个
                status = t.status()
                results.append({
                    "description": status.get("description", ""),
                    "state": status.get("state", ""),
                    "type": status.get("task_type", ""),
                    "id": status.get("id", ""),
                })
            return results
        except Exception as e:
            return [{"error": str(e)}]
