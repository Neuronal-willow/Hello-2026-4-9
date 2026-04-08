# %reset -f
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import snntorch as snn

# ================================================================
# %% [网络定义 — 带中间量记录的推理版本]
# ================================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        num_inputs = 196
        num_hidden1 = 40
        num_hidden2 = 20
        num_outputs = 10

        beta1 = 0.875
        beta2 = 0.875
        beta3 = 0.875

        self.fc1 = nn.Linear(num_inputs, num_hidden1)
        self.lif1 = snn.Leaky(beta=beta1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.lif2 = snn.Leaky(beta=beta2)
        self.fc3 = nn.Linear(num_hidden2, num_outputs)
        self.lif3 = snn.Leaky(beta=beta3, learn_beta=True)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []
        mem3_rec = []

        num_steps = 10

        for step in range(num_steps):
            x_pool = F.max_pool2d(x, kernel_size=2)
            cur1 = self.fc1(x_pool.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)


class InferenceNet(Net):
    """
    继承自训练用的 Net，覆写 forward 以记录每一层
    每个时间步的 输入电流 / 脉冲 / 膜电位。
    """
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1_rec, spk1_rec, mem1_rec = [], [], []
        cur2_rec, spk2_rec, mem2_rec = [], [], []
        cur3_rec, spk3_rec, mem3_rec = [], [], []

        num_steps = 10
        for step in range(num_steps):
            x_pool = F.max_pool2d(x, kernel_size=2)
            cur1 = self.fc1(x_pool.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur1_rec.append(cur1); spk1_rec.append(spk1); mem1_rec.append(mem1)
            cur2_rec.append(cur2); spk2_rec.append(spk2); mem2_rec.append(mem2)
            cur3_rec.append(cur3); spk3_rec.append(spk3); mem3_rec.append(mem3)

        return {
            'cur1': torch.stack(cur1_rec), 'spk1': torch.stack(spk1_rec), 'mem1': torch.stack(mem1_rec),
            'cur2': torch.stack(cur2_rec), 'spk2': torch.stack(spk2_rec), 'mem2': torch.stack(mem2_rec),
            'cur3': torch.stack(cur3_rec), 'spk3': torch.stack(spk3_rec), 'mem3': torch.stack(mem3_rec),
        }


# ================================================================
# %% [加载模型 & 数据]
# ================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

net = InferenceNet()
net.load_state_dict(torch.load('./mnist_snn_net.pth', map_location=device))
net.to(device)
net.eval()

transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# 取前 2 张图
dataiter = iter(testloader)
images, labels = next(dataiter)
images_gpu = images.to(device)


# ================================================================
# %% [推理 & 收集中间结果]
# ================================================================
with torch.no_grad():
    results = net(images_gpu)

final_spk = results['spk3'][-1]
_, predicted = torch.max(final_spk, dim=1)

for i in range(2):
    print(f"图片 {i+1}: 真实标签={classes[labels[i]]}, 预测标签={classes[predicted[i]]}")


# ================================================================
# %% [可视化 — 辅助函数]
# ================================================================
N_SHOW = 10   # 热力图展示前 10 个神经元

def show_input_image(ax, img_tensor, true_label, pred_label):
    ax.imshow(img_tensor.cpu().squeeze(), cmap='gray')
    ax.set_title(f"True={true_label}  Pred={pred_label}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

def show_single_dynamic(ax, data, tag_name, title_suffix="", cmap='coolwarm'):
    """绘制独立的一张热力图，data shape: (10, C)"""
    im = ax.imshow(data.cpu().numpy(), aspect='auto', cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([f"T{t}" for t in range(data.shape[0])], fontsize=8)

    C = data.shape[1]
    ax.set_xticks(range(C))
    ax.set_xticklabels([f"n{j}" for j in range(C)], fontsize=7, rotation=45)

    ax.set_ylabel("Time step", fontsize=8)
    ax.set_title(f"{tag_name}\n{title_suffix}", fontsize=9)


# ================================================================
# %% ★ [打印热力图数值 + 全部神经元统计]
# ================================================================
layer_names = ["FC1→LIF1", "FC2→LIF2", "FC3→LIF3 (output)"]
keys = [('cur1', 'spk1', 'mem1'),
        ('cur2', 'spk2', 'mem2'),
        ('cur3', 'spk3', 'mem3')]

for img_idx in range(2):
    print("\n" + "=" * 80)
    print(f"  图片 {img_idx+1}   True={classes[labels[img_idx]]}  Pred={classes[predicted[img_idx]]}")
    print("=" * 80)

    for layer_idx, ((ck, sk, mk), layer_name) in enumerate(zip(keys, layer_names)):
        # 取出该图片在该层的完整数据: (10, C_full)
        cur_full = results[ck][:, img_idx, :]
        spk_full = results[sk][:, img_idx, :]
        mem_full = results[mk][:, img_idx, :]

        # 截取展示部分: (10, N_SHOW)
        cur_show = cur_full[:, :N_SHOW]
        spk_show = spk_full[:, :N_SHOW]
        mem_show = mem_full[:, :N_SHOW]

        # ----- 打印热力图对应的数值 -----
        print(f"\n  ── {layer_name} (show first {min(N_SHOW, cur_full.shape[1])}"
              f" / {cur_full.shape[1]} neurons) ──")

        print(f"\n  [Input Current]")
        print("       " + "".join([f"   n{j:<3d}" for j in range(cur_show.shape[1])]))
        for t in range(10):
            row_vals = cur_show[t].cpu().numpy()
            print(f"  T{t}  " + "".join([f"{v:8.4f}" for v in row_vals]))

        print(f"\n  [Spike]")
        print("       " + "".join([f"   n{j:<3d}" for j in range(spk_show.shape[1])]))
        for t in range(10):
            row_vals = spk_show[t].cpu().numpy()
            print(f"  T{t}  " + "".join([f"{v:8.4f}" for v in row_vals]))

        print(f"\n  [Membrane Potential]")
        print("       " + "".join([f"   n{j:<3d}" for j in range(mem_show.shape[1])]))
        for t in range(10):
            row_vals = mem_show[t].cpu().numpy()
            print(f"  T{t}  " + "".join([f"{v:8.4f}" for v in row_vals]))

        # ----- 全部神经元统计 -----
        cur_np = cur_full.detach().cpu().numpy()
        spk_np = spk_full.detach().cpu().numpy()
        mem_np = mem_full.detach().cpu().numpy()

        print(f"\n  ┌─ 全部 {cur_np.shape[1]} 个神经元统计 (10 个时间步) ─┐")
        print(f"  │  Input Current :  max={cur_np.max():+.6f}  min={cur_np.min():+.6f} │")
        print(f"  │  Spike          :  max={spk_np.max():+.6f}  min={spk_np.min():+.6f} │")
        print(f"  │  Membrane Pot   :  max={mem_np.max():+.6f}  min={mem_np.min():+.6f} │")
        print(f"  └──────────────────────────────────────────────────┘")


# ================================================================
# %% [可视化 — 拆分为独立的小图 (3行4列 GridSpec 布局)]
# ================================================================
#
#   col:  0          1               2          3
#         ┌──────┬──────────────┬────────┬──────────────┐
#   row 0 │      │  L1 Current  │  L1 Spk│ L1 Mem Pot  │
#         │      │  (FC1→LIF1)  │        │              │
#   row 1 │ 原图 ├──────────────┼────────┼──────────────┤
#         │(跨行)│  L2 Current  │  L2 Spk│ L2 Mem Pot  │
#         │      │  (FC2→LIF2)  │        │              │
#   row 2 │      ├──────────────┼────────┼──────────────┤
#         │      │  L3 Current  │  L3 Spk│ L3 Mem Pot  │
#         │      │  (FC3→LIF3)  │        │              │
#         └──────┴──────────────┴────────┴──────────────┘

for img_idx in range(2):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Image {img_idx+1}:  True={classes[labels[img_idx]]}  "
        f"Pred={classes[predicted[img_idx]]}",
        fontsize=15, fontweight='bold', y=0.98
    )

    # 统一管理 3行 4列 网格
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    # 第 0 列：原图（跨 3 行）
    ax_img = fig.add_subplot(gs[:, 0])
    show_input_image(ax_img, images[img_idx],
                     classes[labels[img_idx]], classes[predicted[img_idx]])

    # 遍历 3 层网络，依次填入第 1, 2, 3 行
    for row_idx, ((ck, sk, mk), layer_name) in enumerate(zip(keys, layer_names)):
        # 第 1 列：Input Current
        ax = fig.add_subplot(gs[row_idx, 1])
        show_single_dynamic(ax, results[ck][:, img_idx, :N_SHOW],
                            "Input Current", layer_name, cmap='Reds')

        # 第 2 列：Spike
        ax = fig.add_subplot(gs[row_idx, 2])
        show_single_dynamic(ax, results[sk][:, img_idx, :N_SHOW],
                            "Spike", layer_name, cmap='Blues')

        # 第 3 列：Membrane Potential
        ax = fig.add_subplot(gs[row_idx, 3])
        show_single_dynamic(ax, results[mk][:, img_idx, :N_SHOW],
                            "Membrane Pot", layer_name, cmap='Greens')

    plt.savefig(f"snn_layer_dynamics_split_img{img_idx+1}.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存 → snn_layer_dynamics_split_img{img_idx+1}.png\n")


# ================================================================
# %% [补充可视化 — 选中神经元的膜电位时间曲线]
# ================================================================
for img_idx in range(2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.suptitle(
        f"Image {img_idx+1} — Membrane Potential over time (first {N_SHOW} neurons)",
        fontsize=12, fontweight='bold'
    )
    colors = plt.cm.tab10(np.linspace(0, 1, N_SHOW))

    for col, (ck, sk, mk), layer_name in zip(range(3), keys, layer_names):
        mem_val = results[mk][:, img_idx, :N_SHOW].detach().cpu().numpy()
        for n in range(mem_val.shape[1]):
            axes[col].plot(range(10), mem_val[:, n], '-o', markersize=3.5,
                           color=colors[n], label=f"n{n}", linewidth=1.2)
        axes[col].set_xlabel("Time step", fontsize=9)
        axes[col].set_ylabel("Membrane Potential", fontsize=9)
        axes[col].set_title(layer_name, fontsize=10)
        axes[col].legend(fontsize=6, ncol=2, loc='best')
        axes[col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"snn_membrane_curve_img{img_idx+1}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存 → snn_membrane_curve_img{img_idx+1}.png\n")

print("===== 全部可视化完成 =====")
