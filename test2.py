import jieba.posseg as pseg
from collections import Counter
from pathlib import Path

def demo_tagging(text):
    print("cut 结果：")
    print(" / ".join(f"{w.word}/{w.flag}" for w in pseg.cut(text)))

    print("\nlcut 结果：")
    print([f"{w.word}/{w.flag}" for w in pseg.lcut(text)])

def count_names(txt_path, out_path):
    text = Path(txt_path).read_text(encoding="utf-8")
    words = pseg.lcut(text)

    counts = Counter()
    for w in words:
        if w.flag == "nr" and len(w.word.strip()) > 1:
            counts[w.word] += 1

    result = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    with open(out_path, "w", encoding="utf-8") as f:
        for name, cnt in result:
            line = f"{name}\t{cnt}"
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    demo_tagging("我喜欢自然语言处理")
    count_names("1.txt", "name_counts.txt")