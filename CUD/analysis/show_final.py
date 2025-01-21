import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(file_name):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_name)
        
        # 检查是否有至少两列
        if df.shape[1] < 2:
            print("The CSV file must have at least two columns.")
            return
        
        # 提取第一列和第二列作为横纵坐标
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        
        # 绘制图形
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-', label='Data')
        plt.xlabel('X (First Column)')
        plt.ylabel('Y (Second Column)')
        plt.title(f"Plot of {file_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 显示图形
        plt.show()
    except Exception as e:
        print(f"Error reading or plotting file {file_name}: {e}")

# 示例：输入文件名
if __name__ == "__main__":
    file_name = "./result_summary.csv"
    plot_csv(file_name)
