# 教学知识图谱系统 - 完整技术方案

## 执行摘要

本方案在ASTRA基础上进行三层扩展，构建一个**结构感知、权重驱动、掌握度追踪**的教学知识图谱系统。

**核心创新**：
1. **自顶向下的结构化构建**：利用教材元数据直接生成多粒度节点
2. **关联度量化**：实体关系权重化，表达知识点间的密切程度
3. **动态掌握度模型**：学生学习数据驱动的节点权值更新

---

## 第一部分：自顶向下的结构化图谱构建

### 1.1 系统架构变动

```
原ASTRA流程：
文本块 → LLM提取实体 → 合并 → 图存储 → 聚类
       ↑
    无结构信息

改进流程：
教材元数据 → 结构节点初始化 → 内容分层提取 → 关系建立 → 权重赋值
     ↓              ↓              ↓          ↓        ↓
  章→节→小节   高→中→低层节点   补充细节   连接    量化强度
```

### 1.2 教材元数据规范

**输入格式** (JSON/YAML):

```json
{
  "document_id": "ML_101",
  "document_title": "机器学习基础",
  "document_type": "textbook",
  "structure": {
    "hierarchy_levels": ["part", "chapter", "section", "subsection"],
    "parts": [
      {
        "id": "part_1",
        "level": 0,
        "title": "第一部分：基础数学",
        "learning_objectives": ["理解线性代数基础", "掌握概率论"],
        "chapters": [
          {
            "id": "ch_1_1",
            "level": 1,
            "title": "第一章：线性代数",
            "summary": "介绍向量、矩阵和特征值等基本概念",
            "estimated_importance": 0.9,  # 教师评估的重要度
            "sections": [
              {
                "id": "sec_1_1_1",
                "level": 2,
                "title": "1.1 向量与矩阵",
                "content_range": [0, 2000],  # 字符范围
                "key_concepts": ["向量", "矩阵", "转置"],
                "prerequisite_sections": [],
                "next_sections": ["sec_1_1_2"],
                "subsections": [
                  {
                    "id": "subsec_1_1_1_1",
                    "level": 3,
                    "title": "1.1.1 向量定义与运算",
                    "content_range": [0, 800],
                    "key_terms": ["向量空间", "点积", "范数"]
                  },
                  {
                    "id": "subsec_1_1_1_2",
                    "level": 3,
                    "title": "1.1.2 矩阵及其性质",
                    "content_range": [800, 1600],
                    "key_terms": ["方阵", "奇异矩阵", "逆矩阵"]
                  }
                ]
              },
              {
                "id": "sec_1_1_2",
                "level": 2,
                "title": "1.2 特征值与特征向量",
                "content_range": [2000, 3500],
                "key_concepts": ["特征值", "特征向量", "对角化"],
                "prerequisite_sections": ["sec_1_1_1"]
              }
            ]
          },
          {
            "id": "ch_1_2",
            "level": 1,
            "title": "第二章：概率论",
            "summary": "概率论基础和常见分布",
            ...
          }
        ]
      }
    ]
  },
  "relationships": [
    {
      "from_id": "sec_1_1_1",
      "to_id": "sec_1_1_2",
      "relation_type": "prerequisite",
      "relation_strength": 0.9  # 教师标注的依赖强度
    },
    {
      "from_id": "sec_1_1_1",
      "to_id": "ch_1_2",
      "relation_type": "supports",
      "relation_strength": 0.6
    }
  ]
}
```

**关键字段**：
- `level`: 层级（0=部分, 1=章, 2=节, 3=小节）
- `learning_objectives`: 学习目标
- `key_concepts`: 核心概念
- `prerequisite_sections`: 前置知识
- `estimated_importance`: 教师估计的重要度
- `relation_strength`: 关系强度

### 1.3 自顶向下的节点创建流程

#### **阶段1：结构节点初始化**（直接从元数据）

**代码流程**（添加到ASTRA）：

```python
class StructureAwareASTRA(ASTRA):
    """结构感知的ASTRA"""
    
    async def _init_structural_nodes(
        self,
        metadata: dict,
        knowledge_graph_inst: BaseGraphStorage
    ):
        """
        从元数据直接创建结构节点
        """
        for part in metadata.get('parts', []):
            await self._process_hierarchy_level(
                parent_id=None,
                level_data=part,
                level=0,
                knowledge_graph_inst=knowledge_graph_inst,
                metadata=metadata
            )
    
    async def _process_hierarchy_level(
        self,
        parent_id: str,
        level_data: dict,
        level: int,
        knowledge_graph_inst: BaseGraphStorage,
        metadata: dict
    ):
        """
        递归处理层级结构
        """
        node_id = compute_mdhash_id(level_data['id'])
        
        # 创建结构节点
        node_data = {
            "entity_type": "structural",  # 新增类型
            "entity_name": level_data['title'],
            "description": level_data.get('summary', level_data['title']),
            "structural_level": level,  # 从元数据获得的层级
            "parent_structural_id": parent_id,
            "original_id": level_data['id'],  # 教材中的ID
            "learning_objectives": level_data.get('learning_objectives', []),
            "key_concepts": level_data.get('key_concepts', []),
            "estimated_importance": level_data.get('estimated_importance', 0.5),
            "mastery_degree": 0.0,  # 初始掌握度
            "content_boundary": {
                "start": level_data.get('content_range', [0, 0])[0],
                "end": level_data.get('content_range', [0, 0])[1],
            },
            "source_id": "",  # 暂无，后续添加
            "clusters": "[]",  # 初始无聚类
        }
        
        # 插入节点
        await knowledge_graph_inst.upsert_node(node_id, node_data)
        
        # 建立层级边
        if parent_id:
            await knowledge_graph_inst.upsert_edge(
                source_node_id=parent_id,
                target_node_id=node_id,
                edge_data={
                    "relation_type": "hierarchical_contains",
                    "relation_strength": 1.0,  # 确定性边
                    "is_structural": True,
                }
            )
        
        # 递归处理子级
        for child_key in ['chapters', 'sections', 'subsections']:
            for child in level_data.get(child_key, []):
                await self._process_hierarchy_level(
                    parent_id=node_id,
                    level_data=child,
                    level=level + 1,
                    knowledge_graph_inst=knowledge_graph_inst,
                    metadata=metadata
                )
    
    async def _build_structural_edges(
        self,
        metadata: dict,
        knowledge_graph_inst: BaseGraphStorage
    ):
        """
        建立元数据中指定的前置/支持关系
        """
        for rel in metadata.get('relationships', []):
            from_id = compute_mdhash_id(rel['from_id'])
            to_id = compute_mdhash_id(rel['to_id'])
            
            await knowledge_graph_inst.upsert_edge(
                source_node_id=from_id,
                target_node_id=to_id,
                edge_data={
                    "relation_type": rel['relation_type'],
                    "relation_strength": rel['relation_strength'],
                    "is_structural": True,
                    "from_metadata": True,
                }
            )
```

**结果**：

```
Part_1 (level=0)
  ├─ Chapter_1_1 (level=1)
  │  ├─ Section_1_1_1 (level=2)
  │  │  ├─ Subsection_1_1_1_1 (level=3)
  │  │  └─ Subsection_1_1_1_2 (level=3)
  │  └─ Section_1_1_2 (level=2)
  └─ Chapter_1_2 (level=1)
     ...
```

#### **阶段2：分层内容提取**

**策略**：每层级不同的LLM prompt和粒度

```python
async def _hierarchical_content_extraction(
    self,
    document_text: str,
    metadata: dict,
    knowledge_graph_inst: BaseGraphStorage,
    config: dict
):
    """
    不同层级的内容提取，从上到下
    """
    
    # 层级0-1：Part和Chapter
    # 策略：使用已有的summary，LLM仅验证和补充
    for part in metadata['parts']:
        part_node_id = compute_mdhash_id(part['id'])
        part_content = document_text[part.get('content_range', [0, len(document_text)])[0]:
                                      part.get('content_range', [0, len(document_text)])[1]]
        
        # LLM验证并补充摘要
        validated_summary = await self._llm_validate_summary(
            original_summary=part['summary'],
            content=part_content,
            level="part"
        )
        
        # 更新节点
        part_node = await knowledge_graph_inst.get_node(part_node_id)
        part_node['description'] = validated_summary
        await knowledge_graph_inst.upsert_node(part_node_id, part_node)
    
    # 层级2-3：Section和Subsection
    # 策略：抽取关键实体
    for section in self._iter_sections(metadata):
        section_text = document_text[section['content_range'][0]:section['content_range'][1]]
        section_node_id = compute_mdhash_id(section['id'])
        
        # LLM提取该节内的概念
        section_entities = await extract_hierarchical_entities(
            text=section_text,
            chunk_size=500,  # 较小的chunk（细粒度）
            max_gleaning=2,
            context_hint=section['key_concepts']  # 使用已知概念作为hint
        )
        
        # 关联实体到该节
        for entity in section_entities:
            entity_id = compute_mdhash_id(entity['name'])
            
            await knowledge_graph_inst.upsert_node(
                node_id=entity_id,
                node_data={
                    "entity_type": "concept",
                    "entity_name": entity['name'],
                    "description": entity['description'],
                    "structural_level": section['level'] + 1,  # 比节低一级
                    "parent_structural_id": section_node_id,  # 属于该节
                    "source_id": GRAPH_FIELD_SEP.join([section['id']]),
                    "mastery_degree": 0.0,
                    "clusters": "[]",
                }
            )
            
            # 建立"属于"关系
            await knowledge_graph_inst.upsert_edge(
                source_node_id=entity_id,
                target_node_id=section_node_id,
                edge_data={
                    "relation_type": "belongs_to",
                    "relation_strength": 0.95,
                    "is_structural": True,
                }
            )
```

**结果的节点类型**：

```
Layer 0: Part (结构节点) - 由元数据生成
Layer 1: Chapter (结构节点) - 由元数据生成
Layer 2: Section (结构节点) - 由元数据生成
Layer 3: Subsection (结构节点) - 由元数据生成
Layer 4: Concept (提取节点) - 由LLM从内容提取
Layer 5: Detail (细粒度节点) - 可选，通过进一步聚类生成
```

#### **阶段3：节点去重和合并**

```python
async def _merge_extracted_concepts_with_structure(
    self,
    knowledge_graph_inst: BaseGraphStorage,
    embeddings_vdb: BaseVectorStorage,
    threshold: float = 0.85
):
    """
    合并LLM提取的概念与结构节点中的key_concepts
    """
    
    # 获取所有提取的概念节点
    extracted_nodes = [
        (node_id, node_data)
        for node_id, node_data in knowledge_graph_inst._graph.nodes(data=True)
        if node_data.get('entity_type') == 'concept'
    ]
    
    # 获取所有结构节点的key_concepts
    for node_id, node_data in knowledge_graph_inst._graph.nodes(data=True):
        if node_data.get('entity_type') == 'structural' and 'key_concepts' in node_data:
            for key_concept in node_data['key_concepts']:
                # 查找相似的提取概念
                concept_candidates = [
                    n for n in extracted_nodes
                    if similarity(n[1]['entity_name'], key_concept) > threshold
                ]
                
                if concept_candidates:
                    # 合并：更新concept的描述和属性
                    selected_concept = concept_candidates[0]
                    concept_node_data = selected_concept[1]
                    
                    concept_node_data['canonical_name'] = key_concept
                    concept_node_data['from_key_concepts'] = True
                    await knowledge_graph_inst.upsert_node(
                        selected_concept[0],
                        concept_node_data
                    )
                else:
                    # 创建新的concept节点（关键概念未被提取）
                    new_concept_id = compute_mdhash_id(key_concept)
                    await knowledge_graph_inst.upsert_node(
                        node_id=new_concept_id,
                        node_data={
                            "entity_type": "concept",
                            "entity_name": key_concept,
                            "description": key_concept,
                            "structural_level": node_data['structural_level'] + 1,
                            "parent_structural_id": node_id,
                            "canonical_name": key_concept,
                            "from_key_concepts": True,
                            "source_id": node_data.get('original_id', ''),
                            "mastery_degree": 0.0,
                            "clusters": "[]",
                        }
                    )
                    
                    await knowledge_graph_inst.upsert_edge(
                        source_node_id=new_concept_id,
                        target_node_id=node_id,
                        edge_data={
                            "relation_type": "belongs_to",
                            "relation_strength": 0.95,
                            "is_structural": True,
                        }
                    )
```

### 1.4 集成到ASTRA的ainsert流程

```python
class StructureAwareASTRA(ASTRA):
    
    async def ainsert_with_structure(
        self,
        document_text: str,
        structure_metadata: dict,
        **kwargs
    ):
        """
        带结构的插入流程
        
        Args:
            document_text: 原始文本
            structure_metadata: 教材结构元数据
        """
        # 初始化存储
        await self._storage_index_start_callback()
        
        # ========== Phase 1: 结构初始化 ==========
        logger.info("Phase 1: 创建结构节点...")
        await self._init_structural_nodes(structure_metadata, self.knowledge_graph_inst)
        await self._build_structural_edges(structure_metadata, self.knowledge_graph_inst)
        
        # ========== Phase 2: 分层提取 ==========
        logger.info("Phase 2: 分层内容提取...")
        await self._hierarchical_content_extraction(
            document_text=document_text,
            metadata=structure_metadata,
            knowledge_graph_inst=self.knowledge_graph_inst,
            config=self.global_config
        )
        
        # ========== Phase 3: 去重合并 ==========
        logger.info("Phase 3: 节点去重和合并...")
        await self._merge_extracted_concepts_with_structure(
            knowledge_graph_inst=self.knowledge_graph_inst,
            embeddings_vdb=self.entities_vdb,
            threshold=0.85
        )
        
        # ========== Phase 4: 关系抽取（含权重） ==========
        logger.info("Phase 4: 关系抽取和权重赋值...")
        await self._extract_relations_with_weights(
            knowledge_graph_inst=self.knowledge_graph_inst
        )
        
        # ========== Phase 5: 图聚类 ==========
        logger.info("Phase 5: 图聚类...")
        await self.knowledge_graph_inst.clustering(self.graph_cluster_algorithm)
        
        # ========== Phase 6: 社区报告 ==========
        logger.info("Phase 6: 生成社区报告...")
        await self._generate_community_reports()
        
        # 完成回调
        await self._storage_index_done_callback()
        logger.info(f"✅ 完成结构化插入")
```

---

## 第二部分：关系权重量化

### 2.1 权重的定义

权重 = 两个知识点之间的**关联密切程度**

```
权重范围：[0, 1]
0.0 ← 完全无关
0.3 ← 弱关联（间接相关）
0.6 ← 中关联（有共同前置知识）
0.9 ← 强关联（直接依赖）
1.0 ← 必然关联（逻辑上同义或包含）
```

### 2.2 权重计算方法

权重由多个因素决定：

```
weight = w1 * semantic_similarity + 
         w2 * structural_proximity + 
         w3 * co_occurrence_score + 
         w4 * explicit_relation_score

其中：
- semantic_similarity: 语义相似度 [0, 1]
- structural_proximity: 结构接近度 [0, 1]
- co_occurrence_score: 共现程度 [0, 1]
- explicit_relation_score: 显式关系 [0, 1]
- w1=0.3, w2=0.25, w3=0.25, w4=0.2 (可调参)
```

#### **2.2.1 语义相似度** (semantic_similarity)

```python
async def _compute_semantic_similarity(
    self,
    entity1_name: str,
    entity2_name: str,
    entity1_desc: str,
    entity2_desc: str,
    embedding_func: Callable,
) -> float:
    """
    计算两个实体的语义相似度
    """
    # 嵌入实体名称和描述
    emb1 = await embedding_func([entity1_name, entity1_desc])
    emb2 = await embedding_func([entity2_name, entity2_desc])
    
    # 名称相似度
    name_sim = cosine_similarity(emb1[0], emb2[0])
    
    # 描述相似度
    desc_sim = cosine_similarity(emb1[1], emb2[1])
    
    # 加权平均
    semantic_sim = 0.4 * name_sim + 0.6 * desc_sim
    
    return semantic_sim
```

#### **2.2.2 结构接近度** (structural_proximity)

```python
def _compute_structural_proximity(
    self,
    node1_data: dict,
    node2_data: dict,
    max_distance: int = 5
) -> float:
    """
    计算两个节点在结构树中的接近度
    
    逻辑：
    - 同一父节点 → 1.0
    - 父子关系 → 0.95
    - 同一祖父 → 0.8
    - 距离越远 → 分数越低
    """
    level1 = node1_data.get('structural_level', -1)
    level2 = node2_data.get('structural_level', -1)
    
    parent1 = node1_data.get('parent_structural_id')
    parent2 = node2_data.get('parent_structural_id')
    
    if parent1 and parent1 == parent2:
        # 同一父节点
        return 1.0
    
    if parent1 == node2_data.get('node_id') or parent2 == node1_data.get('node_id'):
        # 父子关系
        return 0.95
    
    # 通过BFS计算距离
    distance = self._compute_hierarchical_distance(node1_data, node2_data)
    
    if distance is None or distance > max_distance:
        return 0.1
    
    # 距离转换为相近度（距离越远分数越低）
    proximity = 1.0 / (1.0 + distance * 0.2)
    
    return proximity
```

#### **2.2.3 共现程度** (co_occurrence_score)

```python
def _compute_co_occurrence_score(
    self,
    entity1_name: str,
    entity2_name: str,
    document_text: str,
    window_size: int = 200  # 字符
) -> float:
    """
    计算两个实体在文本中的共现频率
    """
    positions1 = [m.start() for m in re.finditer(re.escape(entity1_name), document_text)]
    positions2 = [m.start() for m in re.finditer(re.escape(entity2_name), document_text)]
    
    if not positions1 or not positions2:
        return 0.0
    
    co_occur_count = 0
    for pos1 in positions1:
        for pos2 in positions2:
            if abs(pos1 - pos2) <= window_size:
                co_occur_count += 1
                break  # 每个positions1只计一次
    
    # 正则化到[0, 1]
    max_possible = min(len(positions1), len(positions2))
    co_occur_score = co_occur_count / max_possible if max_possible > 0 else 0.0
    
    return min(co_occur_score, 1.0)
```

#### **2.2.4 显式关系** (explicit_relation_score)

```python
async def _compute_explicit_relation_score(
    self,
    entity1: str,
    entity2: str,
    document_text: str,
) -> float:
    """
    使用LLM识别两个实体间的显式关系
    """
    prompt = f"""
    分析以下两个知识点之间的关系强度：
    - 知识点1：{entity1}
    - 知识点2：{entity2}
    
    文本摘录：...（包含这两个知识点的文本段落）
    
    请给出关系强度评分 [0, 1]：
    - 0.0: 无关系
    - 0.5: 间接相关
    - 1.0: 直接强关系（如"A定义B"、"A是B的特例"等）
    
    返回JSON格式：{"relation_strength": 0.X, "reason": "..."}
    """
    
    response = await self.best_model_func(prompt)
    result = json.loads(extract_first_complete_json(response))
    
    return result['relation_strength']
```

#### **2.2.5 完整的权重计算**

```python
async def _compute_edge_weight(
    self,
    src_node: dict,
    tgt_node: dict,
    document_text: str,
    embedding_func: Callable,
    config: dict
) -> float:
    """
    计算边的权重（关联度）
    """
    # 各分量的权重
    w1, w2, w3, w4 = 0.3, 0.25, 0.25, 0.2
    
    # 计算各分量
    sem_sim = await self._compute_semantic_similarity(
        src_node['entity_name'],
        tgt_node['entity_name'],
        src_node['description'],
        tgt_node['description'],
        embedding_func
    )
    
    struct_prox = self._compute_structural_proximity(src_node, tgt_node)
    
    co_occur = self._compute_co_occurrence_score(
        src_node['entity_name'],
        tgt_node['entity_name'],
        document_text,
        window_size=200
    )
    
    # 显式关系（可选，耗时）
    if config.get('use_llm_for_relations', False):
        explicit_rel = await self._compute_explicit_relation_score(
            src_node['entity_name'],
            tgt_node['entity_name'],
            document_text
        )
    else:
        explicit_rel = 0.0
    
    # 加权组合
    weight = (w1 * sem_sim + 
              w2 * struct_prox + 
              w3 * co_occur + 
              w4 * explicit_rel)
    
    return max(0.0, min(weight, 1.0))  # 归一化到[0, 1]
```

### 2.3 权重赋值到图中

```python
async def _extract_relations_with_weights(
    self,
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    提取关系并赋予权重
    """
    
    # 获取所有节点对
    all_nodes = list(knowledge_graph_inst._graph.nodes(data=True))
    
    for i, (node1_id, node1_data) in enumerate(all_nodes):
        for node2_id, node2_data in all_nodes[i+1:]:
            
            # 跳过已存在的边
            if knowledge_graph_inst._graph.has_edge(node1_id, node2_id):
                continue
            
            # 计算权重
            weight = await self._compute_edge_weight(
                node1_data,
                node2_data,
                document_text=self.last_inserted_text,  # 缓存
                embedding_func=self.embedding_func,
                config=self.global_config
            )
            
            # 如果权重足够大，才创建边
            if weight > 0.3:  # 阈值可配置
                await knowledge_graph_inst.upsert_edge(
                    source_node_id=node1_id,
                    target_node_id=node2_id,
                    edge_data={
                        "relation_type": "related_to",
                        "relation_strength": weight,  # 权重
                        "is_structural": False,
                    }
                )
                
                logger.debug(f"Added edge {node1_id}→{node2_id}, weight={weight:.2f}")
```

**节点数据结构更新**：

```python
edge_data = {
    "relation_type": "related_to",
    "relation_strength": 0.75,      # ← 新增：权重
    "is_structural": False,
    "src_entity": "梯度下降",
    "tgt_entity": "参数更新",
    "weight_components": {
        "semantic_similarity": 0.8,
        "structural_proximity": 0.6,
        "co_occurrence_score": 0.9,
        "explicit_relation": 0.7
    },
    "weight_formula": "0.3*0.8 + 0.25*0.6 + 0.25*0.9 + 0.2*0.7 = 0.75"
}
```

---

## 第三部分：动态掌握度模型

### 3.1 掌握度数据结构

**每个节点增加掌握度相关字段**：

```python
node_data = {
    # 原有字段
    "entity_name": "梯度下降",
    "description": "...",
    
    # 新增掌握度字段
    "mastery_degree": 0.65,              # 当前掌握度 [0, 1]
    "mastery_last_updated": "2024-11-14T10:30:00Z",
    "mastery_update_count": 5,           # 更新次数
    
    "mastery_records": [                 # 掌握度历史
        {"timestamp": "2024-11-14T10:30:00Z", "degree": 0.65, "reason": "练习题"},
        {"timestamp": "2024-11-13T15:00:00Z", "degree": 0.60, "reason": "考试"},
        ...
    ],
    
    "related_questions": [               # 相关的评估题
        {"question_id": "q123", "correctness": 0.8},
        {"question_id": "q456", "correctness": 0.7},
    ],
}
```

### 3.2 掌握度计算

```python
class MasteryDegreeManager:
    """
    管理学生的知识点掌握度
    """
    
    async def update_mastery_degree(
        self,
        student_id: str,
        knowledge_point_id: str,
        assessment_data: dict,  # 来自练习/考试
        knowledge_graph_inst: BaseGraphStorage,
    ):
        """
        更新单个知识点的掌握度
        
        Args:
            student_id: 学生ID
            knowledge_point_id: 知识点（节点）ID
            assessment_data: {
                "question_ids": ["q1", "q2", ...],
                "correctness": [1.0, 0.5, 1.0],  # 正确率
                "assessment_type": "practice" | "exam",
                "timestamp": "2024-11-14T10:30:00Z"
            }
        """
        
        # Step 1: 获取该知识点当前的掌握度
        node = await knowledge_graph_inst.get_node(knowledge_point_id)
        current_mastery = node.get('mastery_degree', 0.0)
        
        # Step 2: 计算新掌握度
        # 方法A：指数平滑移动平均
        new_responses_avg = sum(assessment_data['correctness']) / len(assessment_data['correctness'])
        
        # 根据评估类型设置权重
        if assessment_data['assessment_type'] == 'exam':
            alpha = 0.4  # 考试权重更大
        else:
            alpha = 0.3  # 练习权重较小
        
        new_mastery = alpha * new_responses_avg + (1 - alpha) * current_mastery
        
        # Step 3: 更新节点
        node['mastery_degree'] = new_mastery
        node['mastery_last_updated'] = assessment_data['timestamp']
        node['mastery_update_count'] = node.get('mastery_update_count', 0) + 1
        
        # 维护历史记录
        if 'mastery_records' not in node:
            node['mastery_records'] = []
        node['mastery_records'].append({
            "timestamp": assessment_data['timestamp'],
            "degree": new_mastery,
            "reason": assessment_data['assessment_type'],
            "score": new_responses_avg,
        })
        
        # 记录相关题目
        if 'related_questions' not in node:
            node['related_questions'] = []
        for qid, correctness in zip(assessment_data['question_ids'], assessment_data['correctness']):
            node['related_questions'].append({
                "question_id": qid,
                "correctness": correctness,
                "timestamp": assessment_data['timestamp']
            })
        
        await knowledge_graph_inst.upsert_node(knowledge_point_id, node)
        
        logger.info(f"Updated mastery for {knowledge_point_id}: {current_mastery:.2f} → {new_mastery:.2f}")
        
        return new_mastery
```

### 3.3 级联更新机制

**核心思想**：当一个节点的掌握度更新时，其邻域节点也要同步更新。更新的范围和幅度由当前节点的掌握度决定。

```python
async def cascade_mastery_update(
    self,
    knowledge_point_id: str,
    knowledge_graph_inst: BaseGraphStorage,
    update_depth: int = None,  # 自动决定
):
    """
    级联更新相邻知识点的掌握度
    
    逻辑：
    1. 获取该节点的掌握度
    2. 根据掌握度决定更新范围
    3. 沿着边权重传播更新
    """
    
    # Step 1: 获取节点信息
    node = await knowledge_graph_inst.get_node(knowledge_point_id)
    current_mastery = node['mastery_degree']
    
    # Step 2: 根据掌握度决定更新深度
    # 掌握度越低，需要查看更多相邻节点（帮助补充前置知识）
    # 掌握度越高，更新范围可以缩小（知识已掌握）
    if update_depth is None:
        if current_mastery < 0.3:
            update_depth = 3  # 掌握度低，查看3层
        elif current_mastery < 0.6:
            update_depth = 2  # 掌握度中等，查看2层
        else:
            update_depth = 1  # 掌握度高，查看1层
    
    # Step 3: BFS遍历邻域节点
    visited = set()
    queue = [(knowledge_point_id, 0)]  # (node_id, depth)
    
    while queue:
        current_id, depth = queue.pop(0)
        
        if current_id in visited or depth > update_depth:
            continue
        visited.add(current_id)
        
        # 获取邻接节点
        neighbors = await knowledge_graph_inst.get_node_edges(current_id)
        
        if not neighbors:
            continue
        
        for src, tgt in neighbors:
            neighbor_id = tgt
            
            if neighbor_id in visited:
                continue
            
            # Step 4: 计算邻域节点的更新幅度
            edge = await knowledge_graph_inst.get_edge(current_id, neighbor_id)
            edge_weight = edge.get('relation_strength', 0.5)
            
            neighbor_node = await knowledge_graph_inst.get_node(neighbor_id)
            neighbor_mastery = neighbor_node.get('mastery_degree', 0.0)
            
            # 根据边权重传播更新
            # 权重高 → 更新幅度大
            # 权重低 → 更新幅度小
            update_factor = edge_weight * 0.5  # 最大幅度：50%
            
            # 决策：是否需要增加或降低邻居的掌握度
            if current_mastery < neighbor_mastery:
                # 当前节点掌握度低，邻居也应该降低
                # （因为邻居可能依赖当前节点）
                new_mastery = neighbor_mastery - (neighbor_mastery - current_mastery) * update_factor
            else:
                # 当前节点掌握度高，邻居可能要提升
                # （因为邻居的先决条件已满足）
                new_mastery = neighbor_mastery + (current_mastery - neighbor_mastery) * update_factor
            
            # 更新邻域节点
            neighbor_node['mastery_degree'] = max(0.0, min(new_mastery, 1.0))
            neighbor_node['mastery_last_updated'] = datetime.now().isoformat()
            neighbor_node['cascade_update_count'] = neighbor_node.get('cascade_update_count', 0) + 1
            
            await knowledge_graph_inst.upsert_node(neighbor_id, neighbor_node)
            
            logger.info(f"Cascade update: {neighbor_id}: {neighbor_mastery:.2f} → {new_mastery:.2f}")
            
            # 加入队列继续传播
            queue.append((neighbor_id, depth + 1))
```

**级联更新的例子**：

```
初始状态：
梯度下降 (mastery=0.3) 
  ↓ (weight=0.8)
参数更新 (mastery=0.7)
  ↓ (weight=0.6)
损失函数 (mastery=0.8)

触发：学生做了关于"梯度下降"的练习，正确率50%，掌握度从0.5降到0.3

【级别1】梯度下降被更新
- 掌握度: 0.5 → 0.3

【级别2】自动更新参数更新（depth=0）
- 边权重: 0.8
- 参数更新掌握度: 0.7 → 0.7 - (0.7-0.3)*0.8*0.5 = 0.54
  （下降是因为梯度下降的掌握度下降了）

【级别3】自动更新损失函数（depth=1）
- 边权重: 0.6
- 损失函数掌握度: 0.8 → 0.8 - (0.8-0.54)*0.6*0.5 = 0.722
  （幅度更小，因为距离更远）

最终状态：
梯度下降 (mastery=0.3)
参数更新 (mastery=0.54) ← 级联更新
损失函数 (mastery=0.722) ← 级联更新
```

### 3.4 学生学习仪表板

```python
class StudentLearningDashboard:
    """
    可视化学生的学习进度
    """
    
    async def get_mastery_overview(
        self,
        student_id: str,
        knowledge_graph_inst: BaseGraphStorage,
    ) -> dict:
        """
        获取学生的掌握度全景
        """
        result = {
            "overall_mastery": 0.0,
            "mastery_by_level": {},
            "weak_areas": [],
            "strong_areas": [],
            "recommended_focus": []
        }
        
        all_nodes = knowledge_graph_inst._graph.nodes(data=True)
        mastery_scores = []
        
        for node_id, node_data in all_nodes:
            mastery = node_data.get('mastery_degree', 0.0)
            mastery_scores.append(mastery)
            
            level = node_data.get('structural_level', 0)
            if level not in result['mastery_by_level']:
                result['mastery_by_level'][level] = []
            result['mastery_by_level'][level].append(mastery)
            
            # 识别弱点区域（掌握度<0.4）
            if mastery < 0.4:
                result['weak_areas'].append({
                    "knowledge_point": node_data['entity_name'],
                    "mastery": mastery,
                    "level": level,
                })
            
            # 识别强点区域（掌握度>0.8）
            if mastery > 0.8:
                result['strong_areas'].append({
                    "knowledge_point": node_data['entity_name'],
                    "mastery": mastery,
                })
        
        # 计算平均掌握度
        result['overall_mastery'] = sum(mastery_scores) / len(mastery_scores)
        
        # 计算各层平均
        for level, scores in result['mastery_by_level'].items():
            result['mastery_by_level'][level] = sum(scores) / len(scores)
        
        # 推荐学习重点（低掌握度的高重要度节点）
        for weak_area in result['weak_areas']:
            node_data = knowledge_graph_inst._graph.nodes[weak_area['knowledge_point']]
            importance = node_data.get('estimated_importance', 0.5)
            
            if importance > 0.7:  # 重要但掌握度低
                result['recommended_focus'].append({
                    "knowledge_point": weak_area['knowledge_point'],
                    "importance": importance,
                    "mastery": weak_area['mastery'],
                    "reason": "重要知识点，需要加强"
                })
        
        return result
```

---

## 第四部分：完整集成示例

### 4.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    教学知识图谱系统                          │
└─────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │ 教材管理  │          │ 学生评估  │          │ 知识图谱 │
    │ 模块      │          │ 模块      │          │ 模块     │
    └────┬────┘          └────┬────┘          └────┬────┘
         │                    │                    │
    ┌────────────────────────────────────────────────────────┐
    │         StructureAwareASTRA (核心类)                   │
    │                                                        │
    │  ├─ _init_structural_nodes()                         │
    │  ├─ _hierarchical_content_extraction()               │
    │  ├─ _merge_extracted_concepts_with_structure()       │
    │  ├─ _extract_relations_with_weights()                │
    │  └─ ainsert_with_structure()                         │
    └────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────────────────────────────────────────────────────────┐
    │              MasteryDegreeManager                       │
    │                                                        │
    │  ├─ update_mastery_degree()                           │
    │  ├─ cascade_mastery_update()                          │
    │  └─ get_mastery_overview()                            │
    └────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────────────────────────────────────────────────────────┐
    │    存储层 (NetworkX + JSON + VectorDB)                │
    │  - Graph: 节点(含掌握度) + 权重边                     │
    │  - KV: 社区报告 + 学生数据                            │
    │  - VectorDB: 节点嵌入                                 │
    └────────────────────────────────────────────────────────┘
```

### 4.2 完整的使用流程

```python
# ============ 第1步：初始化系统 ============
system = StructureAwareASTRA(
    working_dir="./teaching_rag",
    enable_llm_cache=True,
    enable_hierachical_mode=True,
)

# ============ 第2步：准备教材元数据 ============
metadata = load_yaml("machine_learning_structure.yaml")
# 包含：chapter、section、subsection等结构信息

# ============ 第3步：插入教材 ============
with open("machine_learning_textbook.txt", "r") as f:
    document_text = f.read()

await system.ainsert_with_structure(
    document_text=document_text,
    structure_metadata=metadata,
)

# ============ 第4步：初始化掌握度管理器 ============
mastery_manager = MasteryDegreeManager(
    knowledge_graph_inst=system.knowledge_graph_inst,
    config=system.global_config,
)

# ============ 第5步：学生做练习题 ============
# 假设系统出了一份练习：关于"梯度下降"的3道题

student_assessment = {
    "student_id": "stu_001",
    "knowledge_point_id": compute_mdhash_id("梯度下降"),
    "assessment_data": {
        "question_ids": ["q_001", "q_002", "q_003"],
        "correctness": [1.0, 0.5, 1.0],  # 学生的答题情况
        "assessment_type": "practice",
        "timestamp": datetime.now().isoformat()
    }
}

# 更新梯度下降的掌握度
new_mastery = await mastery_manager.update_mastery_degree(
    student_id="stu_001",
    knowledge_point_id=student_assessment["knowledge_point_id"],
    assessment_data=student_assessment["assessment_data"],
    knowledge_graph_inst=system.knowledge_graph_inst,
)
# Result: 掌握度从0.0 → 0.67（平均答题率）

# ============ 第6步：级联更新相关知识点 ============
await mastery_manager.cascade_mastery_update(
    knowledge_point_id=student_assessment["knowledge_point_id"],
    knowledge_graph_inst=system.knowledge_graph_inst,
    update_depth=2,  # 更新周边2层
)
# Result: 参数更新、损失函数等相邻节点自动更新

# ============ 第7步：学生查询 ============
# 系统理解了学生的掌握度后，可以提供个性化的响应

response = await system.query(
    "梯度下降中的参数更新有什么注意事项？",
    param=QueryParam(
        mode="hi_bridge",
        # 新增：考虑学生掌握度进行答案定制
        consider_mastery=True,  # ← 新功能
        student_id="stu_001",
    )
)
# 系统会根据学生的掌握度：
# - 梯度下降：0.67 (中等)
# - 参数更新：0.45 (较低，被级联更新)
# 返回包含前置知识的更详细答案
```

### 4.3 查询的适配

```python
async def hierarchical_query_with_mastery(
    self,
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    考虑学生掌握度的查询
    """
    
    # 标准向量检索
    results = await entities_vdb.query(query, top_k=query_param.top_k * 10)
    
    if query_param.consider_mastery and query_param.student_id:
        # 增强：基于学生掌握度重新排序/扩展
        enhanced_results = []
        
        for result in results:
            node = await knowledge_graph_inst.get_node(result["entity_name"])
            mastery = node.get('mastery_degree', 0.5)
            
            # 掌握度低的节点，增加权重（需要更多上下文）
            if mastery < 0.4:
                # 添加该节点的前置知识节点
                neighbors = await knowledge_graph_inst.get_node_edges(result["entity_name"])
                for src, tgt in neighbors:
                    neighbor = await knowledge_graph_inst.get_node(tgt)
                    if neighbor and neighbor.get('mastery_degree', 0.0) > mastery:
                        # 前置知识掌握度更高，加入
                        enhanced_results.append({
                            "entity_name": tgt,
                            "score": result["score"] * 0.9,  # 稍降权重
                            "is_prerequisite": True,
                        })
            
            enhanced_results.append(result)
        
        results = enhanced_results
    
    # 继续标准流程...
    # (社区选择、路径查找等)
```

---

## 第五部分：实现成本与优先级

### 5.1 实现路线图

**第一阶段（1-2周）**：基础结构
```
- ✅ 教材元数据规范定义
- ✅ 结构节点初始化
- ✅ 分层内容提取基础框架
- ✅ 集成到ASTRA.ainsert
```

**第二阶段（2-3周）**：权重系统
```
- ✅ 权重计算框架
- ✅ 语义相似度、结构接近度
- ✅ 共现程度、显式关系
- ✅ 边权重存储和查询
```

**第三阶段（2周）**：掌握度管理
```
- ✅ 掌握度数据结构
- ✅ 掌握度更新算法
- ✅ 级联更新机制
- ✅ 学生仪表板
```

**第四阶段（1周）**：系统集成与优化
```
- ✅ 完整的工作流集成
- ✅ 查询适配
- ✅ 性能优化
- ✅ 测试与验证
```

### 5.2 关键参数配置

```python
SYSTEM_CONFIG = {
    # 结构提取参数
    "structure": {
        "max_hierarchy_depth": 4,
        "auto_split_chapter": True,
        "key_concepts_extraction": True,
    },
    
    # 权重计算参数
    "weight_components": {
        "semantic_similarity_weight": 0.3,
        "structural_proximity_weight": 0.25,
        "co_occurrence_weight": 0.25,
        "explicit_relation_weight": 0.2,
        "weight_threshold": 0.3,  # 最小权重阈值
    },
    
    # 掌握度参数
    "mastery": {
        "initial_mastery": 0.0,
        "practice_alpha": 0.3,  # 练习的平滑因子
        "exam_alpha": 0.4,      # 考试的平滑因子
        "cascade_depth_low_mastery": 3,
        "cascade_depth_mid_mastery": 2,
        "cascade_depth_high_mastery": 1,
        "cascade_factor": 0.5,
    },
    
    # 查询参数
    "query": {
        "consider_mastery": True,
        "prerequisite_boost_factor": 0.9,
        "related_boost_factor": 0.7,
    }
}
```

---

## 第六部分：预期效果与评估指标

### 6.1 效果对比

| 指标 | 原ASTRA | 本系统 | 改进 |
|------|---------|--------|------|
| **图节点层级数** | 0-2层（自动聚类） | 3-5层（结构显式） | +40% |
| **节点间关系的可解释性** | 低（相似度） | 高（结构+权重） | ↑↑↑ |
| **学生个性化程度** | 无 | 高（基于掌握度） | 新增 |
| **前置知识推荐** | 无 | 自动（基于级联） | 新增 |
| **教师控制力** | 低 | 高（元数据驱动） | ↑↑ |

### 6.2 评估方法

**量化指标**：
1. **图谱质量**：人工标注的关系覆盖率
2. **掌握度准确性**：与真实学习进度的相关性
3. **推荐有效性**：推荐练习的完成率和效果

**定性反馈**：
1. 教师满意度：结构是否符合教学设计
2. 学生体验：个性化程度是否提升学习效率
3. 系统可维护性：添加新教材的成本

---

## 总结

本方案通过三个创新层次重塑ASTRA：

1. **结构化构建**：从无结构的LLM提取，升级到结构感知的自顶向下构建
2. **关系量化**：从单一相似度权重，升级到多因素的关联度度量
3. **动态追踪**：从静态知识图谱，升级到学生掌握度驱动的动态更新

这使系统成为一个**教学系统**而不仅仅是**检索系统**，能够理解和适应学生的学习过程。

