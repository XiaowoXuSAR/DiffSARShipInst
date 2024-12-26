import json


def calculate_average_score(json_file_path):
    # 打开并读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

        # 假设JSON文件的结构是一个列表，其中每个元素是一个字典，代表一个检测到的实例
        # 每个实例字典中包含一个'score'键
    scores = []

    # 遍历所有实例并提取score
    for instance in data:  # 这里不再使用.get('instances', [])，因为data已经是一个列表
        # 检查instance是否是一个字典，并且包含'score'键
        if isinstance(instance, dict) and 'score' in instance:
            score = instance['score']
            if score >= 0.5:
                scores.append(score)

    # 统计不同分数范围内的数量
    count_05_06 = sum(1 for s in scores if 0.5 <= s < 0.6)
    count_06_07 = sum(1 for s in scores if 0.6 <= s < 0.7)
    count_07_08 = sum(1 for s in scores if 0.7 <= s < 0.8)
    count_08_09 = sum(1 for s in scores if 0.8 <= s < 0.9)
    count_09_10 = sum(1 for s in scores if 0.9 <= s <= 1.0)
    count_above_05 = len(scores)  # 大于0.5的score数量

    # 输出统计结果
    print(f"Score range 0.5-0.6: {count_05_06}")
    print(f"Score range 0.6-0.7: {count_06_07}")
    print(f"Score range 0.7-0.8: {count_07_08}")
    print(f"Score range 0.8-0.9: {count_08_09}")
    print(f"Score range 0.9-1.0: {count_09_10}")
    print(f"Total scores above 0.5: {count_above_05}")

                # 计算平均分数
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"Average score for instances with score >= 0.5: {average_score:.4f}")
    else:
        print("No instances with score >= 0.5 found.")

    # 使用示例

json_file_path = '../HRSID/BFSS-Inst/test_offshore/inference/coco_instances_results.json'  # 替换为你的JSON文件路径
calculate_average_score(json_file_path)