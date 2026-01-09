import torch

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

model = Qwen3VLEmbedder(
    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# inputs = [
#     {
#         "text": "A woman playing with her dog on a beach at sunset.",
#         "instruction": "Retrieve images or text relevant to the user's query.",
#     },
#     {
#         "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
#     },
#     {
#         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
#     },
#     {
#         "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
#         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#     },
# ]

inputs = [
    # {"image": "2. 少し雑.jpg"},
    {"image": "ORG_39 - コピー (2).jpeg"},
    # {"image": "ORG_39 - コピー (2) copy.jpeg"},
    {"image": "Dwg1 (3) - コピー.jpeg"},
]

embeddings = model.process(inputs)
corr_mat = embeddings @ embeddings.T

print(f"embeddings: {embeddings}, shape: {embeddings.shape}")
print(f"corr_mat: {corr_mat}")
