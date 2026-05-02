"""分类工具 — 监督分类(RF/SVM)、非监督聚类、精度评价。"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
except ImportError:
    RandomForestClassifier = None
    SVC = None
    KMeans = None


class ClassificationToolkit:
    """遥感分类工具集"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self._model = None
        self._model_type = None

    def train_classifier(self, image_path: str, samples_csv: str, classifier: str = "rf") -> str:
        """从样本点训练分类器。

        samples_csv: CSV 文件，格式: x,y,label（pixel 坐标）
        classifier: "rf" (Random Forest) 或 "svm"
        """
        if RandomForestClassifier is None:
            return "训练失败：缺少 scikit-learn。"

        path = Path(image_path)
        csv_path = Path(samples_csv)
        if not path.exists():
            return f"影像不存在: {image_path}"
        if not csv_path.exists():
            return f"样本文件不存在: {samples_csv}"

        try:
            import csv

            with rasterio.open(path) as src:
                data = src.read()
                data_3d = np.transpose(data, (1, 2, 0))
                n_bands = src.count

            samples = []
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        x, y, label = int(row[0]), int(row[1]), row[2]
                        if 0 <= y < data_3d.shape[0] and 0 <= x < data_3d.shape[1]:
                            samples.append((x, y, label))

            if not samples:
                return "样本文件解析失败或坐标全部越界。"

            X = np.array([data_3d[y, x, :].astype(np.float32) for x, y, _ in samples])
            y = np.array([label for _, _, label in samples])

            classes, class_counts = np.unique(y, return_counts=True)

            if classifier == "svm" and SVC is not None:
                model = SVC(kernel="rbf", probability=True, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

            model.fit(X, y)

            self._model = model
            self._model_type = classifier

            model_path = Path(self.output_dir) / "classifier_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            class_info = [f"  {c}: {cnt} 样本" for c, cnt in zip(classes, class_counts)]

            return (
                f"分类器训练完成 ({classifier.upper()})。\n"
                f"  样本总数: {len(samples)}\n"
                f"  输入波段: {n_bands}\n"
                f"  类别分布:\n" + "\n".join(class_info) + "\n"
                f"  模型已保存至: {model_path}"
            )

        except Exception as e:
            return f"训练失败: {e}"

    def predict_classify(self, image_path: str) -> str:
        """用已训练的分类器对影像分类。

        需先调用 train_classifier 训练模型。
        """
        if RandomForestClassifier is None:
            return "分类失败：缺少 scikit-learn。"

        if self._model is None:
            model_path = Path(self.output_dir) / "classifier_model.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
            else:
                return "分类失败：没有已训练的模型。请先调用 train_classifier 训练。"

        path = Path(image_path)
        if not path.exists():
            return f"影像不存在: {image_path}"

        try:
            with rasterio.open(path) as src:
                data = src.read()
                data_3d = np.transpose(data, (1, 2, 0))
                pixels = data_3d.reshape(-1, src.count).astype(np.float32)

            labels = self._model.predict(pixels)
            label_map = labels.reshape(data_3d.shape[0], data_3d.shape[1])

            import datetime
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = f"{path.stem}_classified_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tiff_path = out_dir / f"{base}.tif"

            profile = src.profile.copy()
            profile.update(count=1, dtype="int32")
            with rasterio.open(tiff_path, "w", **profile) as dst:
                dst.write(label_map.astype(np.int32), 1)

            classes, counts = np.unique(labels, return_counts=True)
            class_lines = [f"  类别 {c}: {cnt} 像素 ({cnt / len(labels) * 100:.1f}%)" for c, cnt in zip(classes, counts)]

            return (
                f"分类预测完成。\n"
                f"  模型: {self._model_type or 'unknown'}\n"
                f"  类别分布:\n" + "\n".join(class_lines) + "\n"
                f"  结果已保存至: {tiff_path}"
            )

        except Exception as e:
            return f"分类预测失败: {e}"

    def accuracy_assessment(self, image_path: str, validation_csv: str) -> str:
        """精度评价：计算混淆矩阵、OA、Kappa、F1。

        validation_csv: 验证样本 CSV，格式同训练样本 (x, y, label)
        """
        if RandomForestClassifier is None:
            return "精度评价失败：缺少 scikit-learn。"

        path = Path(image_path)
        csv_path = Path(validation_csv)
        if not path.exists():
            return f"影像不存在: {image_path}"
        if not csv_path.exists():
            return f"验证样本不存在: {validation_csv}"

        if self._model is None:
            model_path = Path(self.output_dir) / "classifier_model.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
            else:
                return "精度评价失败：没有已训练的模型。"

        try:
            import csv

            with rasterio.open(path) as src:
                data = src.read()
                data_3d = np.transpose(data, (1, 2, 0))

            val_samples = []
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        x, y, label = int(row[0]), int(row[1]), row[2]
                        if 0 <= y < data_3d.shape[0] and 0 <= x < data_3d.shape[1]:
                            val_samples.append((x, y, label))

            if not val_samples:
                return "验证样本解析失败。"

            X_val = np.array([data_3d[y, x, :].astype(np.float32) for x, y, _ in val_samples])
            y_true = np.array([label for _, _, label in val_samples])
            y_pred = self._model.predict(X_val)

            oa = accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")

            cm = confusion_matrix(y_true, y_pred)

            return (
                f"精度评价结果（{len(val_samples)} 个验证样本）:\n"
                f"  总体精度 (OA): {oa * 100:.2f}%\n"
                f"  Kappa 系数: {kappa:.4f}\n"
                f"  加权 F1: {f1:.4f}\n"
                f"  混淆矩阵:\n{cm}"
            )

        except Exception as e:
            return f"精度评价失败: {e}"

    def unsupervised_kmeans(self, image_path: str, n_clusters: int = 4) -> str:
        """非监督 KMeans 聚类分类"""
        if KMeans is None:
            return "KMeans 失败：缺少 scikit-learn。"

        path = Path(image_path)
        if not path.exists():
            return f"影像不存在: {image_path}"

        try:
            with rasterio.open(path) as src:
                data = src.read()
                data_3d = np.transpose(data, (1, 2, 0))
                h, w, bands = data_3d.shape
                pixels = data_3d.reshape(-1, bands).astype(np.float32)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(pixels)
            label_map = labels.reshape(h, w)

            import datetime
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            base = f"{path.stem}_kmeans_{n_clusters}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tiff_path = out_dir / f"{base}.tif"

            profile = src.profile.copy()
            profile.update(count=1, dtype="int32")
            with rasterio.open(tiff_path, "w", **profile) as dst:
                dst.write(label_map.astype(np.int32), 1)

            classes, counts = np.unique(labels, return_counts=True)
            lines = [f"  类别 {c}: {cnt} 像素 ({cnt / len(labels) * 100:.1f}%)" for c, cnt in zip(classes, counts)]

            return (
                f"KMeans 非监督分类完成 (k={n_clusters})。\n"
                f"  输入波段: {bands}\n"
                f"  类别分布:\n" + "\n".join(lines) + "\n"
                f"  结果已保存至: {tiff_path}\n"
                "注: 类别含义需根据实际地物判读。"
            )

        except Exception as e:
            return f"KMeans 分类失败: {e}"
