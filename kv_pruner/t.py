import torch
import torch.nn.functional as F
import time

# 生成随机输入数据（两个8000x4096的向量）
x1 = torch.randn(8000, 4096).to("cuda:0")
x2 = torch.randn(8000, 4096).to("cuda:0")

# 记录开始时间
start_time = time.time()

# 计算余弦相似
similarity = F.cosine_similarity(x1, x2, dim=1)
#a = [float(similarity[i]) for i in range(10)]
# 记录结束时间
end_time = time.time()

# 输出余弦相似度的计算时间
print("Cosine Similarity Calculation Time: {:.4f} seconds".format(end_time - start_time))

