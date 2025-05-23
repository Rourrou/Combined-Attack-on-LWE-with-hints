from collections import defaultdict


def parse_and_average_dist(file_path, num_experiments_per_nfac=5):
    with open(file_path, 'r') as f:
        lines = f.readlines()


    nfac_dist_map = defaultdict(list)
    current_nfac = None
    current_experiment_dists = []

    for line in lines:
        line = line.strip()

        # 解析 setting_dict
        if line.startswith("runs:"):
            if current_nfac is not None and current_experiment_dists:
                nfac_dist_map[current_nfac].append(current_experiment_dists[-1])
                current_experiment_dists = []
                experiment_count += 1
            continue

        elif line.startswith("setting_dict="):
            try:
                local_dict = {}
                exec("setting=" + line[len("setting_dict="):], {}, local_dict)
                current_nfac = local_dict["setting"]['nfac']
                experiment_count = 0
            except Exception as e:
                print(f"Error parsing line: {line} -> {e}")
                continue

        # 提取 dist 值
        elif line.startswith("dist:"):
            try:
                dist_val = float(line.split("dist:")[1].strip())
                current_experiment_dists.append(dist_val)
            except:
                continue

    # 最后一组数据处理
    if current_nfac is not None and current_experiment_dists:
        nfac_dist_map[current_nfac].append(current_experiment_dists[-1])

    # 计算平均值（最多取 num_experiments_per_nfac 个）
    nfac_avg_map = {
        nfac: sum(dists[:num_experiments_per_nfac]) / min(len(dists), num_experiments_per_nfac)
        for nfac, dists in nfac_dist_map.items()
    }

    return nfac_avg_map


if __name__ == "__main__":
    file_path = "Data/HSMP24/LWE128/Per_hint/norm.txt"
    Norm = []
    averages = parse_and_average_dist(file_path, num_experiments_per_nfac=5)
    for nfac in sorted(averages):
        Norm.append(averages[nfac])
        print(f"nfac = {nfac}, average final dist = {averages[nfac]:.6f}")
    print("Norm:", ", ".join(f"{val:.2f}" for val in Norm))
