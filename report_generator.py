import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fpdf import FPDF

# 生成资金曲线图
def generate_equity_curve_chart(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['equity_curve'], label='资金曲线')
    plt.title('资金曲线')
    plt.xlabel('日期')
    plt.ylabel('资金（元）')
    plt.legend()
    plt.savefig('output/equity_curve.png')
    plt.close()

# 生成因子热力图
def generate_factor_heatmap(data):
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title("因子热力图")
    plt.savefig('output/factor_heatmap.png')
    plt.close()

# 生成PDF报告
def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()

    # 添加标题
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="投资策略报告", ln=True, align='C')

    # 添加资金曲线图
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt="资金曲线", ln=True)
    pdf.image('output/equity_curve.png', x=10, y=40, w=180)

    # 添加因子热力图
    pdf.ln(100)
    pdf.cell(200, 10, txt="因子热力图", ln=True)
    pdf.image('output/factor_heatmap.png', x=10, y=150, w=180)

    # 添加策略描述
    pdf.ln(250)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, txt="策略建议：基于多因子动量策略 + 机器学习信号预测。")
    
    # 输出报告
    pdf.output("output/performance_report.pdf")

# =======================
# 主函数
# =======================
def main():
    generate_equity_curve_chart(pd.DataFrame({'date': pd.date_range(start='2021-01-01', periods=10, freq='D'), 'equity_curve': np.random.random(10)*1000}))
    generate_factor_heatmap(pd.DataFrame(np.random.random((10, 5)), columns=['factor1', 'factor2', 'factor3', 'factor4', 'factor5']))
    generate_pdf_report()

if __name__ == "__main__":
    main()
