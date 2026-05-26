# ASTRA 查询时的子图确定机制

## 概览

ASTRA的查询系统采用**多模式分层子图选择**策略，根据不同的检索模式（`QueryParam.mode`）确定返回什么样的子图给LLM进行生成。核心思想是：**不同查询场景需要不同粒度和类型的上下文组合**。

---

## 一、查询的总体流程

```
Query Input
    ↓
[Step 1] 向量检索：查找top-k相关实体
    ↓
[Step 2] 图遍历：根据mode决定扩展策略
    ├─ naive: 不遍历图，直接用chunks
    ├─ hi_local: 1-hop邻居 + 直接相关文本
    ├─ hi_bridge: 社区内的多段最短路径
    ├─ hi_global: 仅社区摘要
    ├─ hi_nobridge: 直接相关的关系（无路径查找）
    └─ hi: 完整层级（上述所有）
    ↓
[Step 3] 上下文组装：多个组件拼接
    ├─ 实体信息（可选）
    ├─ 社区报告
    ├─ 推理路径（可选）
    └─ 源文本
    ↓
[Step 4] LLM生成：调用大模型
```

---

## 二、六种查询模式的子图选择详解

### 2.1 `mode="naive"` - 最简单的向量检索

**子图确定逻辑**：
- **无图遍历**，完全基于向量相似度
- 检索 `top_k` 个最相似的文本块（chunks）
- 返回这些chunks的原始内容

**代码流程**（`naive_query` @ 1948行）：
```python
results = await chunks_vdb.query(query, top_k=query_param.top_k)  # 向量DB查询
chunks_ids = [r["id"] for r in results]                           # 提取chunk IDs
chunks = await text_chunks_db.get_by_ids(chunks_ids)              # 获取chunk内容

# 按token大小截断
maybe_trun_chunks = truncate_list_by_token_size(
    chunks,
    key=lambda x: x["content"],
    max_token_size=query_param.naive_max_token_for_text_unit,     # 默认10000
)
```

**返回的子图**：
```
┌─ TextChunk_1
├─ TextChunk_2
├─ TextChunk_3
└─ ...
（共top_k个，按token预算截断）
```

**特点**：
- ✅ 最快、最简单
- ❌ 无法理解实体间的语义关系
- ❌ 可能包含噪音（无图结构过滤）

---

### 2.2 `mode="hi_local"` - 局部邻域子图

**子图确定逻辑**：
1. **向量检索**：查找`top_k`个相关实体
2. **1-hop展开**：找这些实体的直接邻居
3. **聚合信息**：实体 + 关系 + 源文本

**代码流程**（`_build_hilocal_query_context` @ 1697行）：
```python
# Step 1: 向量检索顶级实体
results = await entities_vdb.query(query, top_k=query_param.top_k)
node_datas = await get_nodes(results)                             # 获取实体详细信息

# Step 2: 1-hop邻域扩展（在_find_most_related_text_unit_from_entities中）
edges = await get_node_edges(node_datas)                         # 获取出边
all_one_hop_nodes = {e[1] for e in edges}                        # 提取目标节点

# Step 3: 聚合
use_text_units: 源文本块
use_relations: 从顶级实体出发的关系
```

**返回的子图**：
```
顶级实体群（top_k）
    ├─ 关系 → 1-hop邻域节点
    │
    ├─ 源文本块（来自顶级实体的source_id）
    └─ 1-hop邻域节点涉及的源文本块
```

**关键函数**（`_find_most_related_text_unit_from_entities` @ 1039行）：
```python
text_units = [dp["source_id"] for dp in node_datas]              # 顶级实体的源文本
edges = await get_node_edges(node_datas)                         # 获取关系
all_one_hop_nodes = {e[1] for e in edges}                        # 1-hop邻居

# 计算关系强度：邻居中有多少个在检索文本块中
relation_counts = sum(1 for e in edges if c_id in neighbor_chunks)

# 按relation_counts排序（优先返回与顶级实体关联更紧密的文本）
```

**特点**：
- 🔍 看到局部图结构（直接邻域）
- 🎯 关系信息按与顶级实体的"紧密度"排序
- ⚡ 中等速度和质量

---

### 2.3 `mode="hi_nobridge"` - 有社区但无跨社区连接

**子图确定逻辑**：
1. **向量检索**：查找`top_k`个实体
2. **社区选择**：这些实体所属的社区
3. **关系收集**：从顶级实体出发的关系（但**不查找跨社区路径**）
4. **信息组合**：社区摘要 + 实体 + 关系 + 源文本

**代码流程**（`_build_local_query_context` @ 1173行）：
```python
results = await entities_vdb.query(query, top_k=query_param.top_k)

# 关键区别：收集社区信息
node_datas = [n with clusters property for n in results]
use_communities = await _find_most_related_community_from_entities(node_datas)
use_relations = await _find_most_related_edges_from_entities(node_datas)
use_text_units = await _find_most_related_text_unit_from_entities(node_datas)
```

**`_find_most_related_community_from_entities`实现**（@ 993行）：
```python
# 从node_datas的clusters字段中提取社区ID
related_communities = []
for node_d in node_datas:
    if "clusters" in node_d:
        related_communities.extend(json.loads(node_d["clusters"]))

# 只保留level <= query_param.level的社区
related_community_keys = [c["cluster"] for c in related_communities 
                          if c["level"] <= query_param.level]

# 按出现频率 + rating排序
related_community_keys_counts = Counter(related_community_keys)
sorted_keys = sorted(keys, 
                     key=lambda k: (count[k], community_data[k]["rating"]),
                     reverse=True)

# 按token预算截断社区摘要
use_communities = truncate_by_token_size(communities, 
                                         max_tokens=query_param.max_token_for_community_report)
```

**返回的子图**：
```
┌─ 社区1（摘要）
│   └─ [对应实体群] (虽不显示，但在clusters中隐含)
│
├─ 社区2（摘要）
│   └─ [对应实体群]
│
├─ 顶级实体群
│
├─ 直接关系（仅从顶级实体出发，无跨社区桥接）
│
└─ 源文本块
```

**特点**：
- 📊 包含**多个社区摘要**（高层信息）
- ⛔ **无跨社区连接**（不构建社区间的路径）
- 🎯 适合需要多个独立话题的查询

---

### 2.4 `mode="hi_bridge"` - 带社区间推理路径

**子图确定逻辑**：
1. **向量检索**：查找`top_k * 10`个实体（扩大范围）
2. **社区选择**：从top_k个实体提取社区
3. **关键实体识别**：从每个社区中选出top_m个高度数节点
4. **路径查找**：构建**多段最短路径**连接社区间的关键实体
5. **边收集**：提取路径上的所有关系

**代码流程**（`_build_hibridge_query_context` @ 1467行）：
```python
# Step 1: 扩大检索范围
results = await entities_vdb.query(query, top_k=query_param.top_k * 10)
node_datas = [n for n in results if n is not None]

# Step 2: 社区选择
use_communities = await _find_most_related_community_from_entities(
    node_datas[:query_param.top_k],  # 仅用前top_k个
    query_param,
    community_reports
)

# Step 3: 从社区中提取关键实体
key_entities = []
for community in use_communities:
    community_key_entities = [
        e['entity_name'] for e in overall_node_datas 
        if e['entity_name'] in community['nodes']
    ][:query_param.top_m]  # 每个社区最多top_m个
    key_entities.append(community_key_entities)

# 去重并收集
key_entities = list(set(k for kk in key_entities for k in kk))

# Step 4: 多段最短路径查找
path = find_path_with_required_nodes(
    graph=knowledge_graph_inst._graph,
    source=key_entities[0],
    target=key_entities[-1],
    required_nodes=key_entities[1:-1]  # 中间必经节点
)

# Step 5: 提取路径上的所有关系
use_reasoning_path = await _find_most_related_edges_from_paths(
    path_nodes, path, query_param, knowledge_graph_inst
)
```

**关键函数：`find_path_with_required_nodes`**（@ 1305行）：
```python
def find_path_with_required_nodes(graph, source, target, required_nodes):
    final_path = []
    current_node = source
    
    # 遍历每个必经节点
    for next_node in required_nodes:
        try:
            sub_path = nx.shortest_path(graph, source=current_node, target=next_node)
        except nx.NetworkXNoPath:
            # 若路径不存在，直接添加节点（容错）
            final_path.extend([next_node])
            current_node = next_node
            continue
        
        # 合并路径（避免重复）
        if final_path:
            final_path.extend(sub_path[1:])
        else:
            final_path.extend(sub_path)
        
        current_node = next_node
    
    # 从最后一个必经节点到目标
    try:
        sub_path = nx.shortest_path(graph, source=current_node, target=target)
        final_path.extend(sub_path[1:])
    except nx.NetworkXNoPath:
        final_path.extend([target])
    
    return final_path  # 所有段的并集，去重
```

**返回的子图（关键是推理路径）**：
```
社区1 ────[最短路径1]──── 社区2 ────[最短路径2]──── 社区3
  ↓                          ↓                        ↓
KeyEntity1 ──(edge)── InterNode1 ──(edge)── KeyEntity2 ──(edge)── KeyEntity3
                                                      ↓
                                              源文本块
```

**特点**：
- 🌉 **桥接不同社区**（跨社区推理）
- 📍 优先基于**图的拓扑结构**而非向量相似度
- 🎯 适合**需要解释推理链**的查询
- ⚠️ 计算量较大（需要多次最短路径搜索）

---

### 2.5 `mode="hi_global"` - 仅社区摘要

**子图确定逻辑**：
1. **向量检索**：查找`top_k * 10`个实体（扩大范围）
2. **社区提取**：从前top_k个实体提取社区
3. **社区摘要**：返回社区报告
4. **源文本**：提取相关源文本块
5. **⚠️ 不返回实体细节**

**代码流程**（`_build_higlobal_query_context` @ 1620行）：
```python
results = await entities_vdb.query(query, top_k=query_param.top_k * 10)
node_datas = results[:query_param.top_k]

# 仅收集社区和文本，不返回实体细节
use_communities = await _find_most_related_community_from_entities(
    node_datas, query_param, community_reports
)
use_text_units = await _find_most_related_text_unit_from_entities(
    node_datas, query_param, text_chunks_db, knowledge_graph_inst
)

# 返回格式（对比bridge/local缺少Entities和Relationships）
return f"""
-----Backgrounds-----
{communities_context}
-----Source Documents-----
{text_units_context}
"""
```

**返回的子图**：
```
┌─ 社区1摘要 (LLM生成的高层总结)
├─ 社区2摘要
├─ 社区3摘要
└─ ...

┌─ 源文本块1
├─ 源文本块2
└─ ...
```

**特点**：
- 🎓 **高度概括**（摘要层面）
- ⚡ **最快**（无图遍历，无路径查找）
- 🔍 **缺少细节**（无实体和关系信息）
- ✅ 适合**需要快速背景信息**的查询

---

### 2.6 `mode="hi"` - 完整层级（默认）

**子图确定逻辑**：
结合上述所有要素的完整流程（`_build_hierarchical_query_context` @ 1267行）

```
[Step 1] 向量检索: top_k * 10 → top_k
    ↓
[Step 2] 社区选择：从顶级实体提取社区
    ↓
[Step 3] 从社区提取关键实体（top_m per community）
    ↓
[Step 4] 构建多段最短路径（同bridge）
    ↓
[Step 5] 路径上的推理关系
    ↓
[组装] 社区摘要 + 顶级实体 + 推理路径 + 源文本
```

**核心实现**（与bridge逻辑几乎相同）：
```python
# 结构与bridge相同，关键区别在于返回的上下文组织
return f"""
-----Reports-----          # 社区摘要
-----Entities-----         # 顶级实体（与bridge不同，local/nobridge都有）
-----Relationships-----    # 直接关系
-----Sources-----          # 源文本块
"""
```

**返回的子图**：
```
完整的多层上下文：
├─ 社区报告（高层）
├─ 顶级实体（中层）
├─ 推理路径/直接关系（中层）
└─ 源文本块（底层）
```

**特点**：
- 🎯 **最完整**（所有类型信息都包含）
- 🔄 **多层次**（从摘要到细节）
- ⏱️ **最慢**（所有操作都做）
- 🥇 **论文推荐模式**（ASTRA论文的主要评估模式）

---

## 三、子图选择的关键参数

在`QueryParam`中：

| 参数 | 含义 | 影响 |
|------|------|------|
| `mode` | 查询模式 | **根本性**决定子图类型 |
| `top_k` | 初始向量检索数 | 顶级实体数（通常20） |
| `top_m` | 每个社区的关键实体数 | 路径查找的节点数（通常10） |
| `level` | 社区层级 | 只使用level ≤ level的社区 |
| `max_token_for_community_report` | 社区摘要token预算 | 社区摘要的长度（通常12500） |
| `max_token_for_text_unit` | 文本块token预算 | 源文本的总长度（通常20000） |
| `max_token_for_macro_traversal` | 宏观遍历token预算 | 推理路径信息的长度（通常2048） |
| `community_single_one` | 是否只用一个社区 | 若True则仅返回top-1社区 |

---

## 四、不同模式的子图对比表

```
┌────────────────┬──────────┬─────────┬──────────┬───────┬──────────┬─────────┐
│ Mode           │ 向量检索 │ 社区    │ 推理路径 │ 实体  │ 关系     │ 速度    │
├────────────────┼──────────┼─────────┼──────────┼───────┼──────────┼─────────┤
│ naive          │ top_k    │ ✗       │ ✗        │ ✗     │ ✗        │ ⚡⚡⚡   │
│ hi_local       │ top_k    │ ✗       │ ✗        │ ✓     │ 1-hop    │ ⚡⚡    │
│ hi_nobridge    │ top_k    │ ✓       │ ✗        │ ✓     │ direct   │ ⚡     │
│ hi_bridge      │ top_k*10 │ ✓       │ ✓多段    │ ✗     │ path     │ 🐢     │
│ hi_global      │ top_k*10 │ ✓       │ ✗        │ ✗     │ ✗        │ ⚡⚡⚡   │
│ hi (default)   │ top_k*10 │ ✓       │ ✓多段    │ ✓     │ ✓+path   │ 🐢🐢   │
└────────────────┴──────────┴─────────┴──────────┴───────┴──────────┴─────────┘
```

---

## 五、子图确定的决策流程图

```
Query(query_text, mode, params)
    ↓
┌──────────────────────────────────────┐
│ 向量检索                             │
│ entities_vdb.query(query, top_k)     │
│ → results: list[entity_metadata]     │
└──────────────────────────────────────┘
    ↓
    ├─→ mode == "naive"
    │   └─→ get_chunks(chunk_vdb)
    │       └─→ [TextChunk1, TextChunk2, ...]
    │
    ├─→ mode == "hi_local"
    │   └─→ get_nodes(entities) → expand_1_hop → get_relations
    │       └─→ [Entities + 1-hop relations + Chunks]
    │
    ├─→ mode == "hi_nobridge"
    │   └─→ get_nodes → get_communities(nodes.clusters)
    │       └─→ [Communities + Entities + Direct Relations + Chunks]
    │
    ├─→ mode == "hi_bridge"
    │   └─→ get_nodes → get_communities
    │       └─→ key_entities_per_community
    │           └─→ find_shortest_paths(key_entities)
    │               └─→ [Communities + Path Relations + Chunks]
    │
    ├─→ mode == "hi_global"
    │   └─→ get_communities
    │       └─→ [Communities + Chunks]
    │
    └─→ mode == "hi"
        └─→ get_nodes + get_communities + find_paths
            └─→ [Communities + Entities + Path Relations + Chunks]
        
    ↓
┌──────────────────────────────────────┐
│ 上下文组装                           │
│ format_context(subgraph_elements)    │
│ → context_string (CSV tables)        │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ LLM生成                              │
│ llm(query, system_prompt=context)    │
│ → response                           │
└──────────────────────────────────────┘
```

---

## 六、实战示例

### 场景：查询"机器学习中的梯度下降算法"

**Query**: "梯度下降在机器学习中的应用"

#### 模式1：`mode="naive"`
```
📋 返回的子图：
- 文本块1: "梯度下降是一种一阶优化算法..."
- 文本块2: "在神经网络训练中，梯度下降..."
- 文本块3: "随机梯度下降(SGD)相比..."

✅ LLM能看到: 原始文本，直接的话题相关性
❌ LLM看不到: 不同话题间的逻辑关联
```

#### 模式2：`mode="hi_local"`
```
📋 返回的子图：
Entities:
- 梯度下降 (degree: 8)
- 损失函数 (degree: 5)
- 学习率 (degree: 3)

Relationships:
- 梯度下降 --update--> 权重参数
- 损失函数 --define--> 目标
- 学习率 --control--> 收敛速度

Sources:
- 文本块1, 文本块2, ...

✅ LLM能看到: 核心概念 + 直接关系 + 文本
❌ LLM看不到: 更高层的概念分组（如整个"优化算法"社区）
```

#### 模式3：`mode="hi_global"`
```
📋 返回的子图：
Communities:
- [社区1] "优化算法": 梯度下降、随机梯度下降、Adam...
  报告: "本社区涉及常见的优化算法，包括..."
- [社区2] "参数更新机制": 权重、学习率、动量...
  报告: "这些参数控制模型如何..."

Sources:
- 文本块1, 文本块2, ...

✅ LLM能看到: 高层社区分组 + 摘要
❌ LLM看不到: 实体细节和跨社区的具体路径
```

#### 模式4：`mode="hi_bridge"`（推荐）
```
📋 返回的子图：
Reasoning Path:
- 梯度下降 --定义于--> 优化算法概念
- 优化算法 --应用于--> 神经网络训练
- 神经网络训练 --依赖--> 权重参数
- 权重参数 --由梯度控制--> 梯度计算

Communities:
- [社区1] "优化算法": ...
- [社区2] "深度学习基础": ...

Sources:
- 文本块1-10, ...

✅ LLM能看到: 从问题 → 解决方案 → 背景的完整推理链
✅ LLM能看到: 不同社区如何联系起来
✅ LLM能看到: 原始文本支撑
```

#### 模式5：`mode="hi"`（完整）
```
📋 返回的子图（所有上述信息都包含）：
Reports:
- 社区1摘要
- 社区2摘要

Entities:
- 梯度下降 (degree: 8)
- ...

Relationships:
- 所有直接关系

Reasoning Path:
- 跨社区最短路径

Sources:
- 所有相关文本块

✅ 最完整的信息
❌ 最长的上下文（可能超出LLM窗口）
```

---

## 七、子图确定的核心算法

### 7.1 向量检索的子图初始化
```
candidates = vector_search(query_embedding, top_k=K)
subgraph_nodes = {n.entity_name: get_node_data(n.entity_name) 
                  for n in candidates}
```

### 7.2 社区选择的优先级排序
```
community_candidates = {}
for node in subgraph_nodes:
    for cluster_record in json.loads(node.clusters):  # clusters是JSON字符串
        community_id = cluster_record["cluster"]
        level = cluster_record["level"]
        
        if level <= max_level:  # 过滤层级
            if community_id not in community_candidates:
                community_candidates[community_id] = 0
            community_candidates[community_id] += 1  # 计数

# 按出现频率排序，其次按rating排序
sorted_communities = sorted(
    community_candidates.items(),
    key=lambda x: (x[1], community_data[x[0]]["rating"]),
    reverse=True
)
```

### 7.3 推理路径的多段最短路径
```
# BFS多段最短路径
key_entities = [e1, e2, e3, e4]  # 排序的关键实体
path = []

for i in range(len(key_entities)-1):
    segment = nx.shortest_path(graph, 
                               source=key_entities[i], 
                               target=key_entities[i+1])
    path.extend(segment[:-1])  # 避免重复添加中点

path.append(key_entities[-1])  # 添加最后一个节点

# 提取路径上的所有边
path_relations = [graph.edges[path[i]][path[i+1]] 
                  for i in range(len(path)-1)]
```

### 7.4 基于Token预算的截断
```
def truncate_by_token_size(items, key_func, max_tokens):
    """
    items: list[dict] - 待截断的项目
    key_func: Callable - 提取文本内容的函数
    max_tokens: int - 最大token数
    """
    result = []
    token_count = 0
    
    for item in items:
        text = key_func(item)
        item_tokens = count_tokens(text)
        
        if token_count + item_tokens > max_tokens:
            break  # 超出预算则停止
        
        result.append(item)
        token_count += item_tokens
    
    return result
```

---

## 八、性能特征与选择建议

### 查询速度 vs 质量 的权衡

```
质量
  ▲
  │
  │     hi ★★★★★
  │    /
  │   / hi_bridge ★★★★
  │  /        /
  │ /        / hi_nobridge ★★★
  │/        /        /
  │────────────────────────────→ 速度
  hi_global      hi_local    naive
  (快)                       (慢)
```

### 选择建议

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| 需要速度（mobile、实时） | `naive` | 无图操作，最快 |
| 基础QA（答案在单个chunk） | `hi_local` | 1-hop足够，快速 |
| 需要背景信息快速了解 | `hi_global` | 社区摘要，够快 |
| 需要解释推理链 | `hi_bridge` | 最佳信息量/速度比 |
| 需要最完整上下文（论文评估） | `hi` | 所有信息都有 |
| 多话题复杂查询 | `hi_nobridge` | 多社区但无路径复杂度 |

---

## 九、常见问题

**Q: 为什么bridge模式要查询`top_k*10`个实体，然后只用前`top_k`个？**
A: 这样做是为了获得更多候选社区。前`top_k*10`个实体涉及更多社区，从中筛选出前`top_k`个最相关的实体后，仍能保留这些社区信息。

**Q: 推理路径是如何处理不连通的图的？**
A: 使用try-catch容错：若`nx.shortest_path`找不到路径，则直接添加目标节点，继续下一段。

**Q: 社区的`level`有什么意义？**
A: `level`代表社区在层级聚类中的层数。`level=0`是原始实体，`level=1`是一阶抽象，`level=2`是二阶抽象。高level的社区包含更抽象的信息。

**Q: 为什么要有token预算？**
A: LLM有上下文长度限制（如GPT-4的8K/32K）。按组件设置token预算可防止超出窗口，确保生成完整回复。

**Q: hi_local和hi_nobridge有什么区别？**
A: 
- `hi_local`: 无社区信息，仅看1-hop邻域
- `hi_nobridge`: 有社区摘要，但无跨社区路径

---

## 总结

ASTRA的子图确定机制是一个**多层级、可配置的策略**：

1. **base**: 向量检索获得候选节点
2. **expansion**: 根据mode决定如何展开图（无→1-hop→社区→多段路径）
3. **aggregation**: 组合多种类型的信息（实体→关系→路径→文本）
4. **truncation**: 按token预算进行截断
5. **formatting**: 组装成LLM可理解的上下文

通过调整`QueryParam`中的参数，可以灵活地在"速度"和"信息完整性"之间权衡，满足不同应用场景的需求。

