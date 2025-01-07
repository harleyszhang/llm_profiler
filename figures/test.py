import matplotlib.pyplot as plt

def plot_distribution_pie(data, title, save_path):
    labels = data.keys()
    sizes = data.values()
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948']
    
    plt.figure(figsize=(8, 8))  # 增大图表尺寸
    wedges, texts, autotexts = plt.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors, 
        textprops={'fontsize': 10},  # 设置字体大小
    )
    
    # 优化标签和百分比的样式
    for text in texts:
        text.set_fontsize(12)  # 标签字体大小
    for autotext in autotexts:
        autotext.set_color('white')  # 百分比字体颜色
        autotext.set_fontsize(10)  # 百分比字体大小

    plt.title(title, fontsize=16, weight='bold')  # 设置标题样式
    plt.axis('equal')  # 确保饼图为正圆形

    # 图例放在下方，横向排列
    plt.legend(
        loc="upper center",  # 图例位置
        bbox_to_anchor=(0.5, -0.1),  # 放置在下方
        ncol=3,  # 横向排列
        fontsize=10,
        title="Components", 
        title_fontsize=12
    )

    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, bbox_inches='tight')  # 确保保存时不截断内容
    plt.show()

# 示例数据
data = {
    "attn": 22,
    "mlp": 34,
    "layernorm": 8,
    "tp_comm": 31,
    "input_embedding": 4,
    "output_embedding_loss": 1
}

plot_distribution_pie(data, "LLAMA-7B", "optimized_pie_chart_horizontal_legend.png")