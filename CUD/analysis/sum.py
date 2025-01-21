import os
import pandas as pd

def process_csv_files(output_file="result_summary1.csv"):
    # 获取当前目录下以"comparison_trail_"开头的所有CSV文件
    files = [f for f in os.listdir('.') if f.startswith("comparison_trail_") and f.endswith(".csv")]
    
    # 用于存储每个文件的计算结果
    result_data = []
    
    for file in files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file)
            
            # 确保有数据，取最后一行的前五列
            if not df.empty and df.shape[1] >= 5:
                last_row = df.iloc[-1, :4]
                range_value = last_row.max() - last_row.min()  # 极差计算
                result_data.append({"File Name": file, "Range": range_value})
            else:
                print(f"Skipping file {file}: not enough columns or empty.")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # 将结果保存到一个新的CSV文件
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")

# 执行脚本
if __name__ == "__main__":
    process_csv_files()
