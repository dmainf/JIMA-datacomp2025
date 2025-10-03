import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False

print("loading data...")
df = pd.read_csv('../data/data.txt', sep='\t')

print(f"総行数: {len(df)}")
print(f"ユニークな書名数: {df['書名'].nunique()}")
print(f"欠損値の数: {df['書名'].isnull().sum()}")
print(f"欠損値の割合: {(df['書名'].isnull().sum()/len(df['書名'])*100).round(2)}")

book_counts = df['書名'].value_counts().head(20)

plt.figure(figsize=(14, 9))
book_names = [name.replace('\u3000', ' ').strip() for name in book_counts.index]
plt.barh(range(len(book_counts)), book_counts.values)
plt.yticks(range(len(book_counts)), book_names, fontsize=10)
plt.xlabel('出現回数', fontsize=11)
plt.ylabel('書名', fontsize=11)
plt.title('書名の分布(上位20件)', fontsize=13)
plt.gca().invert_yaxis()
plt.tight_layout()
print(f"\n上位20件の書名:")
print(book_counts)
plt.savefig('books_distribution.png', dpi=150, bbox_inches='tight')
print("\nグラフを boosk_distribution.png に保存しました")
