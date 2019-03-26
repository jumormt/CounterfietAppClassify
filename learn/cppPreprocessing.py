with open("/Users/chengxiao/Desktop/file0.txt", 'r', encoding="utf-8") as f:
    content = f.read()
    content = content.strip()
    with open("/Users/chengxiao/Desktop/file0t.txt", 'w', encoding="utf-8") as f1:
        f1.write(content)