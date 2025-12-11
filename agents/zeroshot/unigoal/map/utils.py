def show_depth(depth, fig_id=1):

    if isinstance(depth, torch.Tensor):
        d = depth.detach().cpu().numpy()
    else:
        d = np.asarray(depth)

    d = np.squeeze(d)

    if d.ndim != 2:
        raise ValueError(f"期望二维深度图，拿到 {d.shape}")

    d = d.astype(np.float32)
    mask = np.isfinite(d)
    if mask.any():
        dmin = float(d[mask].min())
        dmax = float(d[mask].max())
        dn = np.zeros_like(d, dtype=np.float32)
        if dmax > dmin:
            dn[mask] = (d[mask] - dmin) / (dmax - dmin)
    else:
        dn = np.zeros_like(d, dtype=np.float32)

    plt.figure(fig_id); plt.clf()
    plt.imshow(dn, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Depth (normalized 0–1)')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

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
    plt.pause(0.001)


def show_map(map_data, fig_id=1, title="Map"):

    if isinstance(map_data, torch.Tensor):
        map_np = map_data.detach().cpu().numpy()
    else:
        map_np = np.asarray(map_data)

    map_np = np.squeeze(map_np)

    plt.figure(fig_id); plt.clf()

    if map_np.ndim == 2:
        plt.imshow(map_np, cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
    elif map_np.ndim == 3:
        if map_np.shape[0] <= 3:
            map_np = map_np.transpose(1, 2, 0)
            plt.imshow(map_np)
        else:
            plt.imshow(map_np[0], cmap='viridis')
            plt.colorbar(fraction=0.046, pad=0.04)
    else:
        raise ValueError(f"不支持的维度: {map_np.shape}")

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def show_segmentation(seg_pred, fig_id=2):
    if isinstance(seg_pred, torch.Tensor):
        seg_np = seg_pred.detach().cpu().numpy()
    else:
        seg_np = np.asarray(seg_pred)
    if seg_np.ndim == 3 and seg_np.shape[0] < seg_np.shape[2]:
        seg_np = seg_np.transpose(1, 2, 0)
    pred_mask = np.argmax(seg_np, axis=2)  #  (H, W)

    colors = plt.cm.tab20(np.linspace(0, 1, seg_np.shape[2]))
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))

    for class_id in range(seg_np.shape[2]):
        mask = pred_mask == class_id
        colored_mask[mask] = colors[class_id][:3]
    plt.figure(fig_id); plt.clf()
    plt.imshow(colored_mask)
    plt.title('Segmentation Result (16 classes)')
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def show_point_cloud(point_cloud, fig_id=1):

    if isinstance(point_cloud, torch.Tensor):
        pc_np = point_cloud.detach().cpu().numpy()
    else:
        pc_np = np.asarray(point_cloud)

    pc_np = np.squeeze(pc_np)

    if pc_np.ndim == 4:
        pc_np = pc_np[0]

    if pc_np.ndim != 3 or pc_np.shape[2] != 3:
        raise ValueError(f"期望点云形状 (H, W, 3)，拿到 {pc_np.shape}")

    points = pc_np.reshape(-1, 3)

    valid_mask = (points[:, 2] != 0) & ~np.isnan(points[:, 2])
    points = points[valid_mask]

    if len(points) == 0:
        print("没有有效的点云数据")
        return

    plt.figure(fig_id); plt.clf()
    ax = plt.axes(projection='3d')

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=points[:, 2], cmap='viridis', s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud')

    plt.colorbar(scatter, ax=ax, label='Z value')

    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                         points[:, 1].max()-points[:, 1].min(),
                         points[:, 2].max()-points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
