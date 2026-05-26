# ASTRA检索中的层级问题详解

## 核心回答

**ASTRA检索时并不是所有层级的节点一起检索，而是采用"分层选择"策略：**

1. **向量检索层（Entity Level）**：检索所有层级的实体节点
2. **社区选择层（Community Level）**：根据参数动态选择社区的具体层级
3. **路径推理层（Reasoning Path）**：基于选中的节点构建多层次知识路径

---

## 第一部分：节点的层级结构

### 1.1 节点的多层级聚类

在ASTRA的图中，每个节点**可以属于多个层级的社区**，这是通过Leiden聚类实现的。

**数据结构**：每个节点有一个 `clusters` 字段，存储其在不同层级的社区隶属关系

```python
# 节点数据示例
node = {
    "entity_name": "梯度下降",
    "description": "参数优化算法",
    "clusters": [
        {"level": 0, "cluster": "c_1"},        # Level 0 (最底层，最细粒度)
        {"level": 1, "cluster": "c_5"},        # Level 1
        {"level": 2, "cluster": "c_12"},       # Level 2
        {"level": 3, "cluster": "c_20"},       # Level 3 (最高层，最粗粒度)
    ]
}
```

**代码位置**：`astra/_storage/gdb_networkx.py` 第200-230行

```python
async def _leiden_clustering(self):
    """分层Leiden聚类"""
    from graspologic.partition import hierarchical_leiden
    
    community_mapping = hierarchical_leiden(
        graph,
        max_cluster_size=self.global_config["max_graph_cluster_size"],
        random_seed=self.global_config["graph_cluster_seed"],
    )
    
    # 每个节点在不同层级的社区映射
    node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
    for partition in community_mapping:
        node_communities[partition.node].append({
            "level": partition.level,        # 层级编号
            "cluster": partition.cluster     # 该层级的社区ID
        })
```

### 1.2 多个层级的含义

| 层级(Level) | 含义 | 粒度 | 社区数量 |
|-----------|------|------|---------|
| 0 | 最细粒度社区 | 局部知识聚集 | 最多 |
| 1 | 中等粒度社区 | 知识领域 | 中等 |
| 2 | 粗粒度社区 | 大主题 | 较少 |
| 3+ | 全局社区 | 整体知识 | 最少 |

**示例**：机器学习知识图

```
Level 0: {梯度下降, SGD, Adam} → 社区 c_1 (优化算法)
Level 1: {c_1社区, 反向传播, 链式法则} → 社区 c_5 (深度学习基础)
Level 2: {c_5社区, 卷积操作, 池化} → 社区 c_12 (神经网络)
Level 3: {c_12社区, 其他主题...} → 社区 c_20 (机器学习)
```

---

## 第二部分：检索过程中的层级选择

### 2.1 向量检索阶段（不区分层级）

**代码位置**：`astra/_op.py` 第1275行

```python
async def _build_hierarchical_query_context(query, ...):
    # Step 1: 向量检索 - 检索所有层级的实体
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)
    # 这里返回的是与查询最相似的top-k个实体节点
    # 不管这些实体属于哪个层级的社区
```

**特点**：
- ✅ 检索的是**个体实体节点（Entity Level）**，不是社区
- ✅ **不区分层级**：无论该实体属于哪个层级的社区，都可能被返回
- ✅ 基于**语义相似度**（embedding距离）进行排序

**结果示例**：
```
Query: "什么是梯度下降"

返回的top-20个实体节点：
1. 梯度下降 (来自chunk_1)
2. 参数更新 (来自chunk_2)
3. 学习率 (来自chunk_3)
4. SGD (来自chunk_1)
5. Adam (来自chunk_5)
...
```

这些实体可能来自不同层级的社区，但都被返回了。

### 2.2 社区选择阶段（区分层级）

**代码位置**：`astra/_op.py` 第993-1038行

```python
async def _find_most_related_community_from_entities(
    node_datas,
    query_param,
    community_reports,
):
    related_communities = []
    
    # Step 1: 提取这些节点所属的所有社区
    for node_d in node_datas:
        related_communities.extend(json.loads(node_d["clusters"]))
    
    # Step 2: 过滤 - 只保留指定层级的社区
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level  # ← 关键：这里限制了层级
    ]
    
    # Step 3: 按相关性排序和裁剪
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    
    # Step 4: 按token大小限制
    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.max_token_for_community_report,
    )
    
    return use_community_reports
```

**关键过滤条件**：`if dp["level"] <= query_param.level`

这意味着：
- 若 `query_param.level = 2`，只选择 Level 0, 1, 2 的社区
- 若 `query_param.level = 0`，只选择 Level 0 的社区（最细粒度）

### 2.3 层级参数的作用

**QueryParam中的level参数**：`astra/base.py` 第14行

```python
@dataclass
class QueryParam:
    level: int = 2  # 默认值为2
```

**不同level值的检索策略**：

| level值 | 选择的社区层级 | 检索策略 | 使用场景 |
|--------|------------|---------|---------|
| 0 | 仅Level 0 | 局部详细检索 | 问题具体、需要细节 |
| 1 | Level 0, 1 | 细粒度+中等粒度 | 标准检索 |
| 2 | Level 0, 1, 2 | **默认** | 通用检索（平衡） |
| 3+ | 所有层级 | 全局+局部 | 宽泛问题 |

**示例对比**：

```
Query: "深度学习中如何优化参数"

level = 0 检索结果:
├─ 梯度下降社区
│  └─ {梯度下降, SGD, Adam}
└─ 反向传播社区
   └─ {反向传播, 链式法则}

level = 2 检索结果:
├─ 神经网络社区 (Level 2)
│  ├─ 深度学习社区 (Level 1)
│  │  ├─ 优化算法社区 (Level 0)
│  │  └─ 反向传播社区 (Level 0)
│  └─ 其他社区 (Level 1)
├─ 机器学习社区 (Level 2)
│  └─ ...
```

---

## 第三部分：6种查询模式中的层级差异

不同的查询模式对层级的利用方式不同：

### 3.1 Naive模式
```python
# 只使用向量检索，不使用图结构
response = astra.query(
    query,
    param=QueryParam(mode="naive")
)
```
- **涉及的层级**：无（直接在chunk级别检索）
- **返回内容**：最相似的chunks
- **complexity**: 最低

### 3.2 hi_local模式
```python
response = astra.query(
    query,
    param=QueryParam(mode="hi_local", level=0)
)
```
- **涉及的层级**：Level 0（最细粒度）
- **范围**：只看1-hop邻接实体和所在社区
- **特点**：局部、细粒度的知识
- **使用场景**：问题很具体

### 3.3 hi_bridge模式
```python
response = astra.query(
    query,
    param=QueryParam(mode="hi_bridge", level=1)
)
```
- **涉及的层级**：Level 0, 1（跨越多个社区）
- **范围**：寻找社区间的最短路径
- **特点**：打通不同知识领域
- **使用场景**：需要跨领域关联

### 3.4 hi_global模式
```python
response = astra.query(
    query,
    param=QueryParam(mode="hi_global", level=3)
)
```
- **涉及的层级**：Level 2, 3（全局社区）
- **范围**：高层抽象的社区总结
- **特点**：全局、粗粒度的知识
- **使用场景**：需要宏观概览

### 3.5 hi_nobridge模式
```python
response = hiram.query(
    query,
    param=QueryParam(mode="hi_nobridge", level=1)
)
```
- **涉及的层级**：Level 0, 1（但不跨社区）
- **范围**：同一社区内的知识
- **特点**：不使用跨社区路径
- **使用场景**：内聚知识检索

### 3.6 hi模式（默认，最强）
```python
response = astra.query(
    query,
    param=QueryParam(mode="hi", level=2)
)
```
- **涉及的层级**：Level 0, 1, 2（全方位）
- **范围**：综合所有信息
- **特点**：多层次、多角度、最完整
- **使用场景**：通用查询

---

## 第四部分：层级选择的具体实现

### 4.1 关键代码流程

```
用户查询
  ↓
QueryParam(level=2, mode="hi")
  ↓
向量检索实体
  (所有层级的实体都可能被返回)
  ↓
获取这些实体的clusters字段
  ↓
过滤社区：
  only keep: dp["level"] <= 2
  (Level 0, 1, 2的社区)
  ↓
按相关性排序社区
  ↓
按token大小裁剪
  ↓
返回选中的社区报告
```

### 4.2 多个level的理解

关键代码：`astra/_op.py` 第1006行

```python
# 这行代码是关键
related_community_dup_keys = [
    str(dp["cluster"])
    for dp in related_communities
    if dp["level"] <= query_param.level  # 这是过滤条件
]
```

**理解**：
- `<=` 而不是 `==`
- 这意味着会选择**多个层级**的社区
- 例如 `level=2` 时，会同时选择 Level 0, 1, 2 的社区
- 这允许**多粒度、多角度**的知识组织

---

## 第五部分：社区层级与层级聚类的对应

### 5.1 Leiden聚类的多层性质

ASTRA使用 `hierarchical_leiden` 而非普通Leiden，原因是：

```python
# 普通Leiden: 只产生一个层级的社区
# hierarchical_leiden: 产生多个层级的社区

community_mapping = hierarchical_leiden(
    graph,
    max_cluster_size=10,  # 限制最大社区大小
    random_seed=42,
)

# 返回的partition对象包含多个层级
# partition.level → 该分区的层级编号
# partition.cluster → 该层级的社区ID
# partition.node → 属于该社区的节点
```

### 5.2 层级的自动生成

Leiden聚类的分层原理：

```
原始图 (1000个节点)
  ↓ 聚类
Level 0 (50个社区，每个~20个节点)
  ↓ 社区作为超节点，再聚类
Level 1 (10个社区，每个~50个节点)
  ↓ 继续聚类
Level 2 (3个社区，每个~166个节点)
  ↓ 继续聚类（如果还有）
Level 3 (1个社区，所有节点)
```

**max_cluster_size=10 的作用**：
- 控制每个社区的最大节点数
- 自动决定需要多少个层级
- 更大的max_cluster_size → 层级更少

---

## 第六部分：实际查询的完整流程

### 6.1 示例：问答"什么是梯度下降"

```
配置：
  Query: "什么是梯度下降"
  QueryParam(mode="hi", level=2, top_k=20)

Step 1: 向量检索（不区分层级）
  results = await entities_vdb.query(query, top_k=200)
  返回: [梯度下降, 参数更新, SGD, Adam, 学习率, ...]
  (所有层级的实体都可能在这里)

Step 2: 取top-20个实体
  node_datas = results[:20]

Step 3: 提取这20个实体的社区信息
  for node in node_datas:
    clusters = json.loads(node["clusters"])
    # clusters: [
    #   {"level": 0, "cluster": "c_1"},
    #   {"level": 1, "cluster": "c_5"},
    #   {"level": 2, "cluster": "c_12"},
    # ]

Step 4: 过滤社区（关键步骤）
  # 只保留 level <= 2 的社区
  selected = [c for c in all_clusters if c["level"] <= 2]
  # 会选择: c_1 (level 0), c_5 (level 1), c_12 (level 2)
  # 不会选择: c_20 (level 3, 太高层)

Step 5: 按相关性排序
  sorted_communities = sort(selected, by=frequency_and_rating)

Step 6: 按token限制裁剪
  final_communities = truncate(sorted_communities, max_tokens=12500)

Step 7: 返回
  response包括:
  - Communities部分: 选中的社区报告
  - Entities部分: top-20个实体
  - Reasoning Path部分: 关键实体间的路径
  - Source Documents部分: 相关的原文chunks
```

### 6.2 与其他level值的对比

| 配置 | 选择的社区 | 返回内容的特点 |
|------|----------|------------|
| level=0 | 仅Level 0 | 最细粒度、最具体、最详细 |
| level=1 | Level 0,1 | 细粒度+中度、局部关联 |
| **level=2** | **Level 0,1,2** | **平衡（默认）** |
| level=3 | 所有层级 | 全局+局部、最完整、最宽泛 |

---

## 第七部分：常见误解

### 误解1：所有节点都在同一层级

**错误**：ASTRA中所有实体节点都在同一个图层级中

**正确**：
- 所有实体都是同一类型的节点（entity level）
- 但每个实体**可能属于多个层级的社区**
- 社区的层级不是节点的层级，而是对节点的不同粒度的组织方式

### 误解2：level参数控制检索深度

**错误**：level=2表示查询深度为2跳

**正确**：
- level参数控制的是**社区的粒度**
- level=2表示选择Level 0,1,2的社区（三种粒度）
- 并不直接控制图的跳数（6模式中会有不同的跳数，但与level无直接对应）

### 误解3：level=0时只返回1个社区

**错误**：level=0只会选择最细粒度的社区，可能很少

**正确**：
- level=0只选择Level 0社区（即使很多个）
- 会选择所有与检索实体相关的Level 0社区
- 可能返回多个Level 0社区的报告（受token限制）

### 误解4：不同level值会改变实体检索结果

**错误**：level参数会影响向量检索返回的实体

**正确**：
- 实体检索阶段**不受level影响**
- 返回的top-k实体是相同的
- level只影响后续的**社区选择**阶段

---

## 第八部分：调参建议

### 8.1 根据问题类型选择level

```python
# 具体问题（细节查询）
astra.query(
    "梯度下降算法的具体步骤",
    param=QueryParam(mode="hi_local", level=0)
)

# 标准问题（通用查询）
astra.query(
    "什么是梯度下降",
    param=QueryParam(mode="hi", level=2)  # 默认
)

# 宽泛问题（概览查询）
astra.query(
    "深度学习和强化学习的区别",
    param=QueryParam(mode="hi_global", level=3)
)
```

### 8.2 组合mode和level

```
mode="naive"        → level无关（不使用图结构）
mode="hi_local"     → level=0 (局部细节)
mode="hi_nobridge"  → level=1 (社区内)
mode="hi_bridge"    → level=1,2 (跨社区)
mode="hi_global"    → level=3 (全局)
mode="hi"(默认)     → level=2 (平衡)
```

---

## 总结表

| 方面 | 说法 | 正确 |
|------|------|------|
| 是否所有层级一起检索 | 是 | ❌ |
| 实体检索是否区分层级 | 是 | ❌ |
| 社区选择是否区分层级 | 否 | ❌ |
| level参数是否影响结果 | 不影响 | ❌ |
| 一个节点能属于多个层级社区吗 | 否，只属于一个 | ❌ |
| level=2时是否只选Level 2社区 | 是 | ❌ |
| **正确回答** | **ASTRA检索采用分层选择策略：向量检索不区分层级（所有实体都可能被返回），但社区选择根据level参数（默认=2）只选择Level 0,1,2的社区** | ✅ |


