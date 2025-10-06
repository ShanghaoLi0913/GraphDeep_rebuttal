# GraphDeEP

GraphDeEP (Graph-based Data Efficient Prompting) 是一个基于图的知识库问答实验系统，主要研究子图选择方法对问答性能的影响。

## 项目结构

```
GraphDeEP/
├── data_utils.py      # 数据处理工具
├── subgraph_utils.py  # 子图选择工具
├── metrics_utils.py   # 评估指标工具
├── main.py           # 主程序
├── data/             # 数据目录
│   └── metaqa/       # MetaQA数据集
│       ├── dev_simple.json
│       ├── entities.txt
│       └── relations.txt
└── experiment_records/ # 实验结果保存目录
```

## 主要功能

1. **基于路径的子图选择**
   - 使用最短路径作为骨干
   - 基于评分的邻居选择
   - 自适应的回退策略

2. **评估指标**
   - 路径一致性得分 (PCS)
   - 实体一致性得分 (ECS)
   - 子图覆盖率统计

3. **数据处理**
   - MetaQA数据集加载和预处理
   - 实体和关系处理
   - Prompt构建

## 使用方法

1. **环境准备**
   ```bash
   # 创建虚拟环境（推荐）
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   .\venv\Scripts\activate  # Windows
   
   # 安装依赖
   pip install -r requirements.txt
   ```

2. **数据准备**
   - 将MetaQA数据集放在 `data/metaqa/` 目录下
   - 确保包含以下文件：
     * dev_simple.json（开发集）
     * entities.txt（实体列表）
     * relations.txt（关系列表）

3. **运行实验**
   ```bash
   python main.py
   ```

4. **查看结果**
   - 实验结果将保存在 `experiment_records/` 目录下
   - 包含以下文件：
     * trimming_results_{timestamp}.jsonl（子图选择结果）
     * inference_results_{timestamp}.jsonl（推理结果）

## 主要参数

- `max_subgraph_size`: 选择的子图最大大小（默认：50）
- `max_path_length`: 路径搜索的最大长度（默认：3）
- `data_path`: 数据集路径
- `output_dir`: 输出目录

## 评估指标说明

1. **路径一致性得分 (PCS)**
   - 衡量选择的子图中问题实体到答案实体的路径质量
   - 得分范围：0-1，越高越好

2. **实体一致性得分 (ECS)**
   - 衡量关键实体（问题和答案实体）在子图中的覆盖程度
   - 得分范围：0-1，越高越好

3. **覆盖率统计**
   - triple_coverage: 三元组覆盖率
   - entity_coverage: 实体覆盖率
   - relation_coverage: 关系覆盖率

## External Context Score (ECS) 实现细节

### 当前ECS计算方法

当前的ECS计算采用了一个复合的评分机制，结合了注意力权重和隐藏状态信息。主要包含以下几个关键组件：

1. **复制头识别机制**
   - 动态识别transformer中的复制头
   - 通过分析注意力矩阵的对角线值来识别
   - 不依赖预定义的复制头配置文件

2. **双信号评分系统**
   - 注意力分布评分：基于复制头的注意力权重
   - 隐藏状态相似度评分：计算上下文表示的相似度
   - 使用alpha参数(0.5)平衡两种评分

3. **评分计算流程**
   ```
   最终ECS = α * 注意力评分 + (1-α) * 隐藏状态评分
   其中：
   - α = 0.5（平衡因子）
   - 注意力评分 = 基于复制头的注意力分布差异
   - 隐藏状态评分 = 上下文表示的余弦相似度
   ```

### 设计理念

1. **为什么使用双信号系统？**
   - 注意力分布：捕获模型对输入信息的关注程度
   - 隐藏状态：考虑语义层面的相似性
   - 结合两种信号可以更全面地评估上下文相关性

2. **动态复制头识别的优势**
   - 适应性：不同问题可能需要不同的复制机制
   - 灵活性：避免硬编码的复制头配置
   - 通用性：可以应用于不同的模型架构

3. **评分范围**
   - 分数范围：0-1
   - 较高分数（>0.1）通常表示较好的上下文相关性
   - 较低分数（<0.05）可能暗示答案质量问题

### 实验观察

- 测试样本全部处理成功，无失败案例
- ECS分数分布：0.07-0.14之间
- 较低的ECS分数（<0.08）往往对应不正确或部分正确的答案
- 较高的ECS分数（>0.1）通常对应准确的答案

### 未来优化方向

1. **简化考虑**
   - 评估是否需要双信号系统
   - 考虑采用更简单的注意力分布方法
   - 可能移除动态复制头识别机制

2. **参数调优**
   - α参数的最优值选择
   - 复制头识别阈值的调整
   - 评分计算方式的优化

## 注意事项

1. 确保数据集格式正确，每个样本包含：
   - question: 问题文本
   - answer_entities: 答案实体ID列表
   - graph: 知识图谱三元组列表

2. 实验结果会自动保存，每次运行会生成新的时间戳文件

3. 可以通过修改 main.py 中的参数来调整实验配置

## 作者

[Your Name]

## 许可证

MIT License

## 文件读取

`dev_simple.json`的结构：

```pseudocode
{
  'answers': [{'kb_id': 'm.01428y', 'text': 'Jamaican English'},
              {'kb_id': 'm.04ygk0', 'text': 'Jamaican Creole English Language'}],
  'entities': [4648],
  'id': 'WebQTest-0',
  'question': 'what does jamaican people speak',
  'subgraph': {
    'entities': [78572, 3372, ...], 
    'tuples': [
       [4648, 430, 77418], 
       [77420, 492, 77421],
       ...
     ]
  }
}
```

- 这里是`entities.txt`的行号，`entities.txt`里面存的是`freebase id` (e.g.,`m.01vrt_c`)
- 格式为`[head_entity_id, relation_name, tail_entity_id]`, 头和尾是`freebase id`，`relation_name`是`relations.txt`的行号，`relations.txt`里面的是`freebase`中的语义路径（如`people.person.nationality`）



问题和子图输入给LLM，输出答案（目前只输入了前两个sample，30个三元组

输出答案

看看我找到的那个数据集是怎么得到这些子图的（问GPT就好了

确定到底要输入多少个三元组，然后输入全部子图，输出正确率还是什么其他常用的metrics

看看`GraphDeEP`原文，确立一下RQs（根据graph的特性提问题

开始计算PCS，ECS



## 实验结果记录

==== Correct? ====
✔️
============================================================████████████████████████████:36<00:00,  7.43s/it]
Correct: 8660
Accuracy: 86.67%
(graphdeep) shanghao-ubuntu@DESKTOP-RET8321:/mnt/d/experiments/GraphDeEP$、



**表格举例：**

| Top-K | Answer Recall (%) | QA Accuracy (%) |
| ----- | ----------------- | --------------- |
| 20    | 87（假设）        | 86.67%          |
| 30    | ...               | ...             |
| 50    | ...               | ...             |



# **Methodology**

**目的**：本项目聚焦于GraphRAG场景下（输入为三元组构成的子图）LLM内部幻觉产生的机制分析。我们跳过传统RAG中的检索过程，直接使用预处理好的、针对每个问题构建的子图作为输入，研究模型在不同机制下对外部知识的利用情况。

**数据集**：本项目选用 processed MetaQA-1hop数据集（来源：[WSDM2021_NSM](https://github.com/RichardHGL/WSDM2021_NSM)），该数据集已通过GraftNet的预处理流程为每个问题构建了高度相关的question-specific 子图（三元组集合）。

**预处理数据方法**：我们采用基于最短路径和重要性评分的启发式方法（裁剪方法详见附录伪代码⭐），将每个问题的相关子图进一步裁剪为Top-20最相关三元组。这一做法在确保所有黄金答案三元组完整保留的同时，显著降低了模型的计算压力。(解释为什么裁剪玩Top-20，可以参考

- Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation
- Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG

根据Hit@1，把实验结果分为hallucinated和truthful：



```
## 裁剪方法
- 最短路径发现
  - 首先找出问题中提到的实体和答案实体
  - 使用广度优先搜索(BFS)找出所有从问题实体到答案实体的最短路径
  - 这些路径上的所有实体和关系构成了核心子图的骨架
- 邻居扩展
  - 对于核心子图中的每个实体，我们会看它的直接邻居（一跳邻居）
  - 这些邻居形成了候选扩展集
  - 目的是为了引入更多上下文信息，帮助理解和推理
- 重要性评分
  - 对每个候选邻居使用一个三维度的评分标准：
  - 如果它连接着路径上的实体，得3分（最重要）
  - 如果它连接着问题中的实体，得2分
  - 如果它连接着答案实体，得2分

这个评分机制确保我们优先选择与核心推理路径相关的信息

- 最终子图组装
  - 首先把所有最短路径上的三元组放入最终子图
  - 然后按照评分从高到低依次添加邻居三元组
  - 直到达到预设的子图大小（比如20个三元组）
  - 如果大小不够就重复使用已有三元组
- 特别的，如果上述主要方法失败了（比如找不到从问题到答案的路径），我们还有一个备选方案：
  - 优先选择包含答案的三元组
  - 然后是问题实体的直接邻居
  - 如果还不够就扩展到二跳邻居
  - 最后随机补充到指定大小
- 这个方法的精妙之处在于：
  - 确保了推理路径的完整性（通过最短路径）
  - 保留了重要的上下文信息（通过邻居扩展）
  - 使用评分机制控制信息的相关性
  - 有备选方案保证鲁棒性
  - 通过固定大小限制计算复杂度
```



## New metrics





# 6.10

- 把ECS跑通
   - 正确率的计算？严格匹配
   - Hit@1：在所有预测中，模型的第一个预测是正确答案的比例（预测对了得1分，错了得0分，最后除以总样本数）。 [我们就用hit@1 判断是truthful还是hallucinated⭐]
   - Hit@K：在所有预测中，正确答案出现在模型前k个预测中的比例（只要正确答案在前k个预测中就得1分，否则得0分，最后除以总样本数）。
   - F1：精确率(Precision)和召回率(Recall)的调和平均数
      - Precision = 正确预测的数量 / 总预测数量
      - Recall = 正确预测的数量 / 实际正确答案数量
      - F1 = 2 * (Precision * Recall) / (Precision + Recall)
      - 通常设置阈值在0.7-0.9之间 例如，相似度 > 0.8：判定为正确，相似度 ≤ 0.8：判定为错误

- **参考ECS的思路，根据图的特点（三元组），创建一个新的metrics**
   - 在图结构中，三元组

- 重新生成一个完整的裁剪后的数据集（trimming_results)
- 在完整的数据集上跑完KBQA，把答对的打错的分开保存，看看ECS score有何不同



