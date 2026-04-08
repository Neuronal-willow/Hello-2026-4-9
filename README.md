# 解决：Uncaught TypeError: Cannot read property 'list' of undefined

## 问题描述
在 Vue3 项目中，接口请求后赋值时报错：
`Cannot read property 'list' of undefined`

## 代码片段
```python
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
```

结束
