import json
import statistics

# 初始化存储各指标值的列表
kggen_entity_coverage = []
kggen_relation_similarity = []
graphrag_entity_coverage = []
graphrag_relation_similarity = []
rakg_entity_coverage = []
rakg_relation_similarity = []

# 读取 JSONL 文件
file_path = 'kg_evaluation_results.jsonl'
with open(file_path, 'r') as file:
    for line in file:
        # 解析每行的 JSON 数据
        data = json.loads(line)
        
        # 提取三个字段的数据
        kggen = data['kggen']
        graphrag = data['graphrag']
        rakg = data['rakg']
        
        # 将指标值添加到对应的列表中
        kggen_entity_coverage.append(kggen['Entity Coverage Rate'])
        kggen_relation_similarity.append(kggen['Relation Similarity'])
        graphrag_entity_coverage.append(graphrag['Entity Coverage Rate'])
        graphrag_relation_similarity.append(graphrag['Relation Similarity'])
        rakg_entity_coverage.append(rakg['Entity Coverage Rate'])
        rakg_relation_similarity.append(rakg['Relation Similarity'])

# 计算均值和方差
kggen_ec_mean = statistics.mean(kggen_entity_coverage)
kggen_ec_var = statistics.variance(kggen_entity_coverage)
kggen_rs_mean = statistics.mean(kggen_relation_similarity)
kggen_rs_var = statistics.variance(kggen_relation_similarity)

graphrag_ec_mean = statistics.mean(graphrag_entity_coverage)
graphrag_ec_var = statistics.variance(graphrag_entity_coverage)
graphrag_rs_mean = statistics.mean(graphrag_relation_similarity)
graphrag_rs_var = statistics.variance(graphrag_relation_similarity)

rakg_ec_mean = statistics.mean(rakg_entity_coverage)
rakg_ec_var = statistics.variance(rakg_entity_coverage)
rakg_rs_mean = statistics.mean(rakg_relation_similarity)
rakg_rs_var = statistics.variance(rakg_relation_similarity)

# 打印结果
print("kggen Entity Coverage Rate: mean =", kggen_ec_mean, ", variance =", kggen_ec_var)
print("kggen Relation Similarity: mean =", kggen_rs_mean, ", variance =", kggen_rs_var)
print("graphrag Entity Coverage Rate: mean =", graphrag_ec_mean, ", variance =", graphrag_ec_var)
print("graphrag Relation Similarity: mean =", graphrag_rs_mean, ", variance =", graphrag_rs_var)
print("rakg Entity Coverage Rate: mean =", rakg_ec_mean, ", variance =", rakg_ec_var)
print("rakg Relation Similarity: mean =", rakg_rs_mean, ", variance =", rakg_rs_var)