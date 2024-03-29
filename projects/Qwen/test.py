# from transformers import Qwen2Tokenizer as T2
# from projects.Qwen.tokenizer import Qwen2Tokenizer as T1


# tokenizer1 = T1(
#     vocab_file="/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B/vocab.json",
#     merges_file="/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B/merges.txt"
# )
# tokenizer2 = T2.from_pretrained("/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B")

# text = [
#     "清晨的阳光洒落在树叶上,露珠在叶片上闪烁着晶莹的光泽。微风拂过,树枝微微摇曳,像是在向大自然问好。泥土的芳香弥漫在空气中,一只小鸟欢快地啾啾鸣叫,这是一个美好的新的一天。",
#     "书本总是向我们敞开怀抱,蕴藏着无穷无尽的智慧和知识。当我打开一本书时,仿佛走进了一个全新的世界。字里行间娓娓道来着作者的心血和思想,让我如痴如醉地沉浸其中,收获了许多启迪和感悟。",
#     "夜幕低垂,城市璀璨的灯火像是一颗颗明亮的星星。街道上来来往往的行人、川流不息的车辆,构成了一幅生动活泼的都市夜景。霓虹灯的光影闪烁,将这座城市渲染得更加缤纷多彩。",
#     "The morning dew glistened on the blades of grass, each droplet reflecting the warm rays of the rising sun. A gentle breeze carried the sweet scent of flowers, and birds serenaded the new day with their cheerful melodies. It was a picture-perfect start to what promised to be a beautiful day.",
#     "As I turned the pages of the worn leather-bound book, I found myself transported to distant lands and bygone eras. The author's words painted vivid scenes that danced across my mind's eye, inviting me to explore the depths of human experience and emotion. Reading has always been an escape, a journey without ever leaving my chair.",
# ]

# for i in text:
#     print(i)
#     res1 = tokenizer1.encode(text)
#     # res2 = tokenizer2.tokenize(i)
#     print(res1)
#     # assert res1 == res2

from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B")
t = AutoTokenizer.from_pretrained("/data/home/xiezipeng/hf_models/Qwen/Qwen1.5-7B")
print(t.encode("<|endoftext|>"))
print(t.pad_token_id)

# text = "给出3点关于保持身体健康的意见。"
# input_ids = t.encode(text, return_tensors='pt')
# res = model.generate(input_ids, max_new_tokens=30)
# res = t.decode(res[0])
# print(res)
