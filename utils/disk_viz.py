# -*- coding: utf-8 -*-
"""
disk_viz.py
超简洁“磁盘即协议”的可视化器：
- 任意代码处只做一件事：save(stream, image, step) -> 保存成 PNG（atomic）
- 每个循环末尾 refresh(step) -> 从磁盘把本步的 PNG 读取出来，动态网格展示
- 离线复盘 = 在线展示：用相同的 refresh(step) 逐帧读磁盘

典型用法：
    from agents.utils.disk_viz import DiskViz

    # 在线跑：创建新的 run 目录
    viz = DiskViz(output_root=cfg.visualization_dir)  # 会自动 runs/<run_id>/
    ...
    viz.save("rgb", rgb_img, step)
    viz.save("global_occupancy", occ_map, step, cmap="gray")
    viz.refresh(step)   # 循环末尾调用
    ...
    viz.close()

    # 离线复盘：打开已有 run 目录
    viz = DiskViz.open_existing("<path-to>/visualization/runs/<run_id>")
    for step in viz.available_steps():
        viz.refresh(step, block=False)  # 或者 viz.play(fps=10)

注意：
- save() 已经把图转成“可直接观看的 PNG”（单通道可选 cmap）。
- refresh() 只读 PNG，不做额外处理；在线/离线完全一致。
"""

import os
import math
import time
import glob
import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm

try:
    import torch
    _TORCH = True
except Exception:
    _TORCH = False


def save_rgb_512(img_hwc, path="./output.jpg"):
    """Center-crop to square, resize to 512x512, save as RGB (PNG/JPEG by suffix)."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    img = np.asarray(img_hwc)            # HWC, RGB
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = img[y0:y0+s, x0:x0+s]

    out = Image.fromarray(crop).resize((512, 512), Image.BILINEAR).convert("RGB")

    plt.figure(1); plt.clf()
    plt.imshow(out)
    plt.title('RGB Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)

def show_image(img, fig_id=1):

    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().numpy()
    else:
        img_np = np.asarray(img)

    img_np = np.squeeze(img_np)

    if img_np.ndim == 3 and img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))

    plt.figure(fig_id); plt.clf()
    plt.imshow(img_np)
    plt.title('RGB Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)



def _canonicalize_image_shape(arr: np.ndarray) -> np.ndarray:
    """
    将任意 HW / HWC / CHW / BCHW / BHWC / (1,H,W) 等形状，规整为：
      - 单通道: HW
      - 多通道: HWC（只保留前3通道）
    规则：若存在 batch 维，默认取第 0 个样本。
    """
    arr = np.asarray(arr)

    # 去 batch 维
    if arr.ndim == 4:
        # 可能是 BCHW 或 BHWC：统一取第 0 个 batch
        arr = arr[0]

    # 处理 3 维的通道位置
    if arr.ndim == 3:
        H, W = None, None
        # 情况 A：CHW（C 在前）
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 4 and arr.shape[2] > 4:
            C = arr.shape[0]
            if C == 1:
                arr = arr[0]               # -> HW
            else:
                arr = np.transpose(arr, (1, 2, 0))  # -> HWC
        # 情况 B：HWC（C 在后）
        elif arr.shape[2] in (1, 3, 4) and arr.shape[0] > 4 and arr.shape[1] > 4:
            if arr.shape[2] == 1:
                arr = arr[..., 0]         # -> HW
            # else: 已是 HWC，保持
        # 情况 C：1HW（channel=1 在最前，但 H/W 很小/很大不容易判）
        elif arr.shape[0] == 1:
            arr = arr[0]                   # -> HW
        # 其他情况保持原样（例如已经是 HW）

    return arr

# ------------------------- 基础工具 -------------------------

def _now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if _TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"))
    return np.asarray(x)


def _minmax01(a: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    if vmin is not None and vmax is not None:
        d = (vmax - vmin) if (vmax - vmin) != 0 else 1e-6
        return np.clip((a - vmin) / d, 0, 1)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)


#def _to_rgb_u8(img: Any, cmap: Optional[str] = None,
               #vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    #arr = _to_numpy(img)
    #if arr.ndim == 3 and arr.shape[2] == 1:
        #arr = arr[..., 0]

    #if arr.ndim == 2:
        #x01 = _minmax01(arr, vmin, vmax)
        #if cmap:
            #cm = mpl_cm.get_cmap(cmap)
            #rgb01 = cm(x01)[..., :3]
            #return (rgb01 * 255.0 + 0.5).astype(np.uint8)
        #else:
            #u8 = (x01 * 255.0 + 0.5).astype(np.uint8)
            #return np.stack([u8, u8, u8], axis=-1)

    #if arr.ndim == 3 and arr.shape[2] >= 3:
        #rgb = arr[..., :3]
        #if rgb.dtype == np.uint8:
            #return rgb
        #rgb = rgb.astype(np.float32, copy=False)
        #mn, mx = float(rgb.min()), float(rgb.max())
        #if 0.0 <= mn and mx <= 1.0:
            #rgb01 = rgb
        #else:
            #rgb01 = _minmax01(rgb)
        #return (rgb01 * 255.0 + 0.5).astype(np.uint8)

    #raise ValueError(f"Unsupported image shape: {arr.shape}")

def _to_rgb_u8(img: Any, cmap: Optional[str] = None,
               vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """
    将任意形状与类型图像转换为 HxWx3 的 uint8 RGB。
    现在支持：HW, HWC, CHW, BCHW, BHWC, (1,H,W) 等。
    """
    arr = _to_numpy(img)
    arr = _canonicalize_image_shape(arr)

    # 单通道 -> 灰度/伪彩
    if arr.ndim == 2:
        x01 = _minmax01(arr, vmin, vmax)
        if cmap:
            cm = mpl_cm.get_cmap(cmap)
            rgb01 = cm(x01)[..., :3]
            return (rgb01 * 255.0 + 0.5).astype(np.uint8)
        else:
            u8 = (x01 * 255.0 + 0.5).astype(np.uint8)
            return np.stack([u8, u8, u8], axis=-1)

    # 多通道（HWC）
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            # 保险：如果还剩 1 通道，按灰度处理
            x01 = _minmax01(arr[..., 0], vmin, vmax)
            u8 = (x01 * 255.0 + 0.5).astype(np.uint8)
            return np.stack([u8, u8, u8], axis=-1)

        # 只取前三通道
        rgb = arr[..., :3]
        if rgb.dtype == np.uint8:
            return rgb
        rgb = rgb.astype(np.float32, copy=False)
        mn, mx = float(rgb.min()), float(rgb.max())
        if 0.0 <= mn and mx <= 1.0:
            rgb01 = rgb
        else:
            rgb01 = _minmax01(rgb)
        return (rgb01 * 255.0 + 0.5).astype(np.uint8)

    raise ValueError(f"Unsupported image shape after canonicalization: {arr.shape}")


#def _atomic_save_png(rgb_u8: np.ndarray, path: str):
    ## 临时文件与目标文件在同一目录，扩展名保留 .png，保存时显式指定 format
    #dirpath = os.path.dirname(path)
    #base = os.path.basename(path)
    #tmp = os.path.join(dirpath, f".{base}.tmp.png")  # 例如 000001.png -> .000001.png.tmp.png
    #Image.fromarray(rgb_u8).save(tmp, format="PNG")
    #os.replace(tmp, path)

def _atomic_save_png(rgb_u8: np.ndarray, path: str):
    dirpath = os.path.dirname(path)
    base = os.path.basename(path)
    tmp = os.path.join(dirpath, f".{base}.tmp.png")
    with open(tmp, "wb") as f:
        Image.fromarray(rgb_u8).save(f, format="PNG")
        f.flush()
        os.fsync(f.fileno())   # ← 确保写到磁盘
    os.replace(tmp, path)


def _grid(n: int) -> Tuple[int, int]:
    if n <= 0:
        return 0, 0
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


# ------------------------- 主类：DiskViz -------------------------

class DiskViz:
    """
    磁盘统一可视化：
      - save(stream, image, step, cmap=None, vmin=None, vmax=None)
      - refresh(step, title=None, block=False)
      - available_streams(step=None) / available_steps()
      - play(fps=10.0, start=None, end=None)
    目录结构：
      <output_root>/runs/<run_id>/
        streams/<stream>/<000123>.png
    """
    def __init__(self,
                 output_root: str,
                 run_id: Optional[str] = None,
                 index_width: int = 6,
                 figure_title: str = "Dashboard"):
        self.output_root = output_root
        _ensure_dir(self.output_root)

        if run_id is None:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_id = ts
        self.run_id = run_id

        self.run_dir = os.path.join(self.output_root, "runs", self.run_id)
        self.streams_dir = os.path.join(self.run_dir, "streams")
        _ensure_dir(self.streams_dir)

        self.index_width = max(3, int(index_width))
        self.figure_title = figure_title

        # 在线窗口
        self._fig = None
        self._axes = {}     # stream -> Axes
        self._images = {}   # stream -> AxesImage
        self._grid_shape = (0, 0)

        self.preferred_first: List[str] = ["rgb"]  # 你可以改成 []
        self._seen_order: List[str] = []          # 记录首次出现的顺序

        # 让在线刷新更丝滑
        try:
            plt.ion()
        except Exception:
            pass

        # 写个最简 meta
        with open(os.path.join(self.run_dir, "meta.txt"), "w", encoding="utf-8") as f:
            f.write(f"created_at: {_now_str()}\nrun_id: {self.run_id}\n")

    # --------- 静态构造：打开已存在 run（离线复盘） ---------
    @staticmethod
    def open_existing(run_dir_or_path: str, figure_title: str = "Dashboard"):
        """
        参数可传：
          - 直接传 runs/<run_id> 目录
          - 也可传 <output_root>/runs/<run_id>
        """
        run_dir = run_dir_or_path
        # 若传的是 output_root，就直接用
        if os.path.basename(os.path.dirname(run_dir)) != "runs":
            # 兼容传到 streams 层的情况
            if os.path.basename(run_dir) == "streams":
                run_dir = os.path.dirname(run_dir)
        streams_dir = os.path.join(run_dir, "streams")
        if not os.path.isdir(streams_dir):
            raise FileNotFoundError(f"streams dir not found: {streams_dir}")

        output_root = os.path.dirname(os.path.dirname(run_dir))
        run_id = os.path.basename(run_dir)
        return DiskViz(output_root=output_root, run_id=run_id, figure_title=figure_title)

    # --------------------- 保存：任何地方只做这一步 ---------------------
    #def save(self, stream: str, image: Any, step: int,
             #cmap: Optional[str] = None, vmin: Optional[float] = None, vmax: Optional[float] = None):

    def save(self, stream: str, image: Any, step: int,
            cmap: Optional[str] = None,
            vmin: Optional[float] = None, vmax: Optional[float] = None,
            vflip: bool = False,   #
            hflip: bool = False):  #

        _ensure_dir(os.path.join(self.streams_dir, stream))
        idx_name = f"{int(step):0{self.index_width}d}.png"
        path = os.path.join(self.streams_dir, stream, idx_name)
        rgb = _to_rgb_u8(image, cmap=cmap, vmin=vmin, vmax=vmax)

        if vflip:
            rgb = np.flipud(rgb)
        if hflip:
            rgb = np.fliplr(rgb)

        _atomic_save_png(rgb, path)

        # --------------------- 刷新：从磁盘读取并显示 ---------------------
    def refresh(self, step: int, title: Optional[str] = None, block: bool = False):
        """
        只读取本 step 存在的所有流 PNG，动态网格显示。
        在线与离线同样用这个函数。
        """
        streams = self.available_streams(step)
        self._ensure_fig(1 if not streams else len(streams))

        if not streams:
            # 没有任何流的该步文件，清空窗口标题提示一下
            self._fig.clf()
            ax = self._fig.add_subplot(1, 1, 1)
            ax.axis("off")
            ax.set_title(f"(no images for step {step})")
            try:
                self._fig.suptitle(title or self.figure_title, fontsize=12)
                self._fig.tight_layout()
            except Exception:
                pass
            plt.pause(0.001)
            return

        rows, cols = _grid(len(streams))

        # 改进1：若网格形状变化 **或** 流集合变化，则重建布局（避免同数不同流时残留/错位）
        if (rows, cols) != self._grid_shape or set(streams) != set(self._axes.keys()):
            self._fig.clf()
            self._axes.clear()
            self._images.clear()
            self._grid_shape = (rows, cols)

        #ordered = sorted(streams)

        for s in streams:
            if s not in self._seen_order:
                self._seen_order.append(s)

        # ✅ 先放优先名单，再放其余按首次出现顺序
        ordered = [s for s in self.preferred_first if s in streams]
        ordered += [s for s in self._seen_order if s in streams and s not in self.preferred_first]
        for i, stream in enumerate(ordered, start=1):
            path = os.path.join(self.streams_dir, stream, f"{int(step):0{self.index_width}d}.png")

            # 改进2：用 with 打开，出错时跳过该流而不是整帧失败
            try:
                with Image.open(path) as im:
                    img = np.asarray(im.convert("RGB"))
            except Exception:
                continue

            ax = self._axes.get(stream)
            if ax is None:
                ax = self._fig.add_subplot(rows, cols, i)
                ax.axis("off")
                self._axes[stream] = ax

            imshow = self._images.get(stream)
            if imshow is None:
                imshow = ax.imshow(img)
                self._images[stream] = imshow
            else:
                imshow.set_data(img)

            ax.set_title(f"{stream}  #{int(step):0{self.index_width}d}", fontsize=9)

            ######################################################delete after debugging
            #h, w = img.shape[:2]
            ## 防止越画越多，先清掉此轴上旧的线
            #try:
                #ax.lines.clear()
            #except Exception:
                #for ln in list(ax.lines):
                    #try: ln.remove()
                    #except Exception: pass
            ## 画水平/垂直中心线（红色，稍半透明）
            #ax.axhline(y=h/2, color="r", alpha=0.6, linewidth=1.0)
            #ax.axvline(x=w/2, color="r", alpha=0.6, linewidth=1.0)

            #######################################################

        # （可留可去）额外清理：把本步未出现的旧流轴移除，双保险
        current = set(ordered)
        stale = [s for s in list(self._axes.keys()) if s not in current]
        for s in stale:
            try:
                ax = self._axes.pop(s)
                self._images.pop(s, None)
                ax.remove()
            except Exception:
                pass

        try:
            self._fig.suptitle(title or self.figure_title, fontsize=12)
            self._fig.tight_layout()
        except Exception:
            pass

        try:
            self._fig.canvas.draw()          # 同步绘制
            self._fig.canvas.flush_events()  # 同步刷新事件
            plt.pause(0.001)
            if block:
                plt.show(block=True)
        except Exception:
            pass


    def _ensure_fig(self, n_streams: int):
        if self._fig is None:
            self._fig = plt.figure(self.figure_title, figsize=(12, 8), dpi=100)
            self._grid_shape = (0, 0)
            self._axes.clear()
            self._images.clear()

    # --------------------- 辅助查询 ---------------------
    def available_streams(self, step: Optional[int] = None) -> List[str]:
        if not os.path.isdir(self.streams_dir):
            return []
        streams = [d for d in os.listdir(self.streams_dir)
                   if os.path.isdir(os.path.join(self.streams_dir, d))]
        if step is None:
            return sorted(streams)
        idx_name = f"{int(step):0{self.index_width}d}.png"
        return sorted([s for s in streams if os.path.isfile(os.path.join(self.streams_dir, s, idx_name))])

    def available_steps(self) -> List[int]:
        """扫描所有流下的 png 文件名，合并得到全局可用步号（升序）。"""
        steps = set()
        for s in self.available_streams(step=None):
            for p in glob.glob(os.path.join(self.streams_dir, s, "*.png")):
                try:
                    b = os.path.splitext(os.path.basename(p))[0]
                    steps.add(int(b))
                except Exception:
                    pass
        return sorted(steps)

    # --------------------- 离线播放 ---------------------
    def play(self, fps: float = 10.0, start: Optional[int] = None, end: Optional[int] = None):
        steps = self.available_steps()
        if not steps:
            print("[DiskViz] no steps found.")
            return
        if start is None:
            start = steps[0]
        if end is None:
            end = steps[-1]
        dt = 1.0 / max(1e-6, fps)
        for st in steps:
            if st < start or st > end:
                continue
            t0 = time.time()
            self.refresh(st)
            # 简单限速
            remain = dt - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)

    # --------------------- 清理 ---------------------
    def close(self):
        try:
            plt.ioff()
        except Exception:
            pass


# ------------------------- 自测示例（可删） -------------------------
if __name__ == "__main__":
    # 在线 demo：随机生成两帧
    viz = DiskViz(output_root="./visualization")
    for step in range(2):
        rgb = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
        depth = np.random.rand(240, 320)
        viz.save("rgb", rgb, step)
        viz.save("depth", depth, step, cmap="magma")
        viz.refresh(step)
    viz.close()

    # 离线 demo：打开刚才的 run 回放
    viz2 = DiskViz.open_existing(os.path.join("./visualization", "runs", viz.run_id))
    viz2.play(fps=2.0)
    viz2.close()
