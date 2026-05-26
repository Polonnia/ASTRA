"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "entity_relation_extraction"
] = """-Goal-
Given a text document that is potentially relevant to this activity, identify all entities from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type/category of the entity
- entity_description: Comprehensive description of the entity's attributes and activities;
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Output rules-
######################
For entity records, output exactly 4 fields:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

For relationship records, output exactly 5 fields:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

-Real Data-
######################
Document Name: {doc_name}
Document Summary: {doc_description}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "entity_extraction"
] = """
Given a text document, identify all entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized. It should be brief and retrieval-oriented, do not include any redundant information. If the entity appears in the summary, use its name directly from the summary. If the entity does not appear in the summary, use its name directly from the text, but make sure to keep it concise and retrieval-oriented.
- entity_type: Type/category of the entity
- entity_description: Comprehensive description of the entity, adhere to the original text as possible.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. Return output in English as a single list of all the entities identified in step 1. Use **{record_delimiter}** as the list delimiter.

3. When finished, output {completion_delimiter}

4. You must strictly use the given delimiters. DO NOT misspell or invent any other delimiters.

######################
-Output rules-
######################
For entity records, output exactly 4 fields:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
You may ignore entities that are not relevant to the main content of the document, such as "Chatper 1", or "Figure 1".

######################
-Example-
######################
Document Name: Tesla Motors Overview
Document Summary: A brief history and product lineup of Tesla.
Text: Elon Musk is the CEO of Tesla, which produces the Model 3 sedan and the Cybertruck. The company is based in Austin, Texas.

Output:
("entity"{tuple_delimiter}Elon Musk{tuple_delimiter}Person{tuple_delimiter}CEO of Tesla, leading the company's strategic direction and public communications.){record_delimiter}
("entity"{tuple_delimiter}Tesla{tuple_delimiter}Organization{tuple_delimiter}Automotive and clean energy company that produces electric vehicles including Model 3 and Cybertruck, headquartered in Austin, Texas.){record_delimiter}
("entity"{tuple_delimiter}Model 3{tuple_delimiter}Product{tuple_delimiter}An electric sedan produced by Tesla, known for being more affordable than other Tesla models.){record_delimiter}
("entity"{tuple_delimiter}Cybertruck{tuple_delimiter}Product{tuple_delimiter}An all-electric, stainless steel pickup truck produced by Tesla, noted for its futuristic design.){record_delimiter}
("entity"{tuple_delimiter}Austin{tuple_delimiter}Location{tuple_delimiter}City in Texas, United States, serving as the headquarters location for Tesla.){completion_delimiter}

######################
-Real Data-
######################
Document Name: {doc_name}
Document Summary: {doc_description}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "entity_extraction_zh"
] = """
-目标-
给定一篇文档，从中识别出实体，以及这些被识别实体之间的所有关系。

-步骤-
1. 识别你认为重要的实体。那些与文件主题内容无关或者过于细节的实体可以忽略。对每个识别出的实体，提取以下信息：
- entity_name：实体的名称，中文
- entity_type：实体的类型/类别；必须从 [{entity_types_zh}] 中选择
- entity_description：对实体属性及活动的全面描述
每个实体的格式为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤 1 中识别出的实体中，找出所有（源实体，目标实体）对，这些实体对之间必须存在*明确的关系*。
对每一对相关的实体，提取以下信息：
- source_entity：源实体的名称，与步骤 1 中识别的一致
- target_entity：目标实体的名称，与步骤 1 中识别的一致
- relationship_description：解释你为何认为源实体与目标实体之间存在关联
- relationship_strength：表示源实体与目标实体之间关系强度的数值评分
每个关系的格式为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. 以中文输出步骤 1 和步骤 2 中识别出的所有实体和关系，返回一个列表。使用 **{record_delimiter}** 作为列表分隔符。

4. 完成后，输出 {completion_delimiter}

######################
-输出规则-
######################
对于实体记录，精确输出 4 个字段：
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

对于关系记录，精确输出 5 个字段：
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

######################
-示例-
######################
文本：
人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。机器学习（ML）是AI的一个子领域，其中“深度学习”利用多层神经网络进行模式识别。2012年，AlexNet在ImageNet竞赛中取得突破，推动了计算机视觉的快速发展。如今，AI被广泛应用于医疗影像诊断，例如Google的DeepMind开发了用于检测乳腺癌的算法。此外，自然语言处理（NLP）技术也常见于智能客服系统中。企业如阿里巴巴和腾讯已将AI用于智能推荐和风险控制。

输出：
("entity"{tuple_delimiter}人工智能{tuple_delimiter}概念{tuple_delimiter}计算机科学的分支，旨在模拟人类智能{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}机器学习{tuple_delimiter}概念{tuple_delimiter}人工智能的子领域，通过数据学习模式{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}深度学习{tuple_delimiter}概念{tuple_delimiter}机器学习的分支，使用多层神经网络{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}计算机视觉{tuple_delimiter}主题{tuple_delimiter}人工智能的应用领域，处理图像和视频{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}自然语言处理{tuple_delimiter}主题{tuple_delimiter}人工智能的分支，处理人类语言{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}医疗影像诊断{tuple_delimiter}应用{tuple_delimiter}AI在医学影像分析中的应用{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}DeepMind{tuple_delimiter}组织{tuple_delimiter}Google旗下的人工智能公司{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}阿里巴巴{tuple_delimiter}组织{tuple_delimiter}中国科技企业{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}腾讯{tuple_delimiter}组织{tuple_delimiter}中国科技企业{tuple_delimiter}){record_delimiter}
{record_delimiter}
("relationship"{tuple_delimiter}机器学习{tuple_delimiter}人工智能{tuple_delimiter}机器学习是人工智能的子领域{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}深度学习{tuple_delimiter}机器学习{tuple_delimiter}深度学习是机器学习的子分支{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}计算机视觉{tuple_delimiter}人工智能{tuple_delimiter}计算机视觉是人工智能的应用领域{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}自然语言处理{tuple_delimiter}人工智能{tuple_delimiter}自然语言处理是人工智能的分支{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}DeepMind{tuple_delimiter}医疗影像诊断{tuple_delimiter}DeepMind开发医疗影像诊断算法{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}阿里巴巴{tuple_delimiter}人工智能{tuple_delimiter}阿里巴巴将AI用于智能推荐和风控{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}腾讯{tuple_delimiter}人工智能{tuple_delimiter}腾讯将AI用于智能推荐和风控{tuple_delimiter}7){completion_delimiter}

-真实数据-
######################
文件名：{doc_name}
文件摘要：{doc_description}
文本：{input_text}
######################
输出：
"""


PROMPTS[
    "relation_extraction"
] = """
Given a text document that is potentially relevant to a list of entities, identify all relationships among the given identified entities.

-Steps-
1. From the entities given by user, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

2. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

3. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entities: ["Type 2 Diabetes", "Insulin Resistance", "HbA1c", "Metformin", "Dietary Fiber Intake", "Fasting Glucose"]
Text:
In an outpatient follow-up, clinicians explain that insulin resistance is a core mechanism of Type 2 diabetes. They prescribe metformin as first-line therapy and track HbA1c every three months to assess long-term glycemic control. The patient also increases dietary fiber intake, and fasting glucose readings gradually decline over the next several weeks.
################
Output:
("relationship"{tuple_delimiter}"Insulin Resistance"{tuple_delimiter}"Type 2 Diabetes"{tuple_delimiter}"Insulin resistance is described as a key pathophysiologic driver of Type 2 diabetes."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Metformin"{tuple_delimiter}"Type 2 Diabetes"{tuple_delimiter}"Metformin is prescribed as a first-line treatment for Type 2 diabetes in this case."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"HbA1c"{tuple_delimiter}"Type 2 Diabetes"{tuple_delimiter}"HbA1c is monitored periodically to evaluate long-term glycemic control in Type 2 diabetes."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Dietary Fiber Intake"{tuple_delimiter}"Fasting Glucose"{tuple_delimiter}"Increased dietary fiber intake is associated with the observed reduction in fasting glucose readings."{tuple_delimiter}8){completion_delimiter}
#############################
Example 2:

Entities: ["Central Bank", "Policy Rate", "Inflation", "Household Consumption", "Unemployment", "Business Investment"]
Text:
During a period of rising inflation, the central bank raises the policy rate in consecutive meetings. Analysts report that higher borrowing costs weaken business investment and slow household consumption growth. Several quarters later, inflation moderates, but unemployment edges up as firms postpone expansion plans.
#############
Output:
("relationship"{tuple_delimiter}"Central Bank"{tuple_delimiter}"Policy Rate"{tuple_delimiter}"The central bank is the institution that sets and adjusts the policy rate."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Policy Rate"{tuple_delimiter}"Business Investment"{tuple_delimiter}"Higher policy rates raise financing costs and are linked to weaker business investment."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Policy Rate"{tuple_delimiter}"Household Consumption"{tuple_delimiter}"Tighter monetary policy is associated with slower household consumption growth."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Inflation"{tuple_delimiter}"Unemployment"{tuple_delimiter}"As inflation moderates after tightening, unemployment is reported to rise in subsequent quarters."{tuple_delimiter}7){completion_delimiter}
#############################
Example 3:

Entities: ["Mangrove Forest", "Coastal Erosion", "Juvenile Fish Habitat", "Shrimp Farming Expansion", "Salinity Intrusion", "Storm Surge Damage"]
Text:
In a delta region, mangrove forest belts reduce wave energy and protect villages from storm surges. Ecologists note that mangroves also provide nursery habitat for juvenile fish. Over the past decade, shrimp farming expansion has replaced many mangrove areas, which coincides with stronger coastal erosion and increased salinity intrusion in nearby farmland.
#############
Output:
("relationship"{tuple_delimiter}"Mangrove Forest"{tuple_delimiter}"Storm Surge Damage"{tuple_delimiter}"Mangrove belts reduce wave energy and therefore mitigate storm surge damage."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Mangrove Forest"{tuple_delimiter}"Juvenile Fish Habitat"{tuple_delimiter}"Mangrove ecosystems are identified as nursery habitat for juvenile fish populations."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Shrimp Farming Expansion"{tuple_delimiter}"Mangrove Forest"{tuple_delimiter}"Expansion of shrimp farming is associated with replacement and loss of mangrove areas."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Shrimp Farming Expansion"{tuple_delimiter}"Coastal Erosion"{tuple_delimiter}"The period of shrimp farming expansion coincides with more severe coastal erosion in the region."{tuple_delimiter}8){completion_delimiter}
#############################
-Real Data-
######################
Entities: {entities}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "relation_extraction_zh"
] = """
-目标-
给定一份可能与给定实体列表相关的文本文档，识别出这些已指定实体之间的所有关系。

-步骤-

1.从用户给出的实体中，找出所有（源实体，目标实体）对，这些实体对之间必须存在明确的关系。
对每一对相关的实体，提取以下信息：

source_entity：源实体的名称，与步骤 1 中识别的一致

target_entity：目标实体的名称，与步骤 1 中识别的一致

relationship_description：解释你为何认为源实体与目标实体之间存在关联

relationship_strength：表示源实体与目标实体之间关系强度的数值评分
每个关系的格式为 ("relationship"{tuple_delimiter}<源实体>{tuple_delimiter}<目标实体>{tuple_delimiter}<关系描述>{tuple_delimiter}<关系强度>)

2.以中文输出步骤 1 中识别出的所有关系，返回一个列表。使用 {record_delimiter} 作为列表分隔符。

3.完成后，输出 {completion_delimiter}

######################
-示例-
######################
示例 1：

实体：["2型糖尿病", "胰岛素抵抗", "糖化血红蛋白", "二甲双胍", "膳食纤维摄入量", "空腹血糖"]
文本：
在一次门诊随访中，临床医生解释称胰岛素抵抗是2型糖尿病的核心机制。他们处方二甲双胍作为一线治疗，并每三个月检测一次糖化血红蛋白以评估长期血糖控制情况。患者还增加了膳食纤维摄入量，在接下来的几周内，空腹血糖读数逐渐下降。
################
输出：
("relationship"{tuple_delimiter}"胰岛素抵抗"{tuple_delimiter}"2型糖尿病"{tuple_delimiter}"胰岛素抵抗被描述为2型糖尿病的关键病理生理驱动因素。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"二甲双胍"{tuple_delimiter}"2型糖尿病"{tuple_delimiter}"在该病例中，二甲双胍被处方为2型糖尿病的一线治疗药物。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"糖化血红蛋白"{tuple_delimiter}"2型糖尿病"{tuple_delimiter}"定期监测糖化血红蛋白以评估2型糖尿病的长期血糖控制情况。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"膳食纤维摄入量"{tuple_delimiter}"空腹血糖"{tuple_delimiter}"膳食纤维摄入量的增加与观察到的空腹血糖读数下降有关。"{tuple_delimiter}8){completion_delimiter}
#############################
示例 2：

实体：["中央银行", "政策利率", "通货膨胀", "居民消费", "失业率", "企业投资"]
文本：
在通货膨胀上升期间，中央银行在连续会议上提高政策利率。分析师报告称，较高的借贷成本削弱了企业投资并减缓了居民消费增长。几个季度后，通货膨胀有所缓和，但由于企业推迟扩张计划，失业率略有上升。
#############
输出：
("relationship"{tuple_delimiter}"中央银行"{tuple_delimiter}"政策利率"{tuple_delimiter}"中央银行是制定和调整政策利率的机构。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"政策利率"{tuple_delimiter}"企业投资"{tuple_delimiter}"较高的政策利率提高了融资成本，并与企业投资减弱相关。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"政策利率"{tuple_delimiter}"居民消费"{tuple_delimiter}"紧缩的货币政策与居民消费增长放缓有关。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"通货膨胀"{tuple_delimiter}"失业率"{tuple_delimiter}"政策紧缩后通货膨胀趋缓，随后几个季度报告失业率上升。"{tuple_delimiter}7){completion_delimiter}
#############################
示例 3：

实体：["红树林", "海岸侵蚀", "幼鱼栖息地", "养虾业扩张", "盐度入侵", "风暴潮破坏"]
文本：
在一个三角洲地区，红树林带减少波浪能量并保护村庄免受风暴潮侵袭。生态学家指出，红树林还为幼鱼提供育苗栖息地。过去十年中，养虾业的扩张取代了许多红树林区域，这与附近农田更强烈的海岸侵蚀和盐度入侵加剧同时发生。
#############
输出：
("relationship"{tuple_delimiter}"红树林"{tuple_delimiter}"风暴潮破坏"{tuple_delimiter}"红树林带减少波浪能量，从而减轻风暴潮破坏。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"红树林"{tuple_delimiter}"幼鱼栖息地"{tuple_delimiter}"红树林生态系统被确认为幼鱼种群的育苗栖息地。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"养虾业扩张"{tuple_delimiter}"红树林"{tuple_delimiter}"养虾业的扩张与红树林区域的取代和丧失有关。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"养虾业扩张"{tuple_delimiter}"海岸侵蚀"{tuple_delimiter}"养虾业扩张时期与该地区更严重的海岸侵蚀同时发生。"{tuple_delimiter}8){completion_delimiter}
#############################
-真实数据-
######################
实体：{entities}
文本：{input_text}
######################
输出：
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


PROMPTS[
    "entiti_continue_extraction"
] = """There are entities from the original text were missed in the last extraction. 
Now please extract all the entities that were missed before. 
Do not infer or imagine entities that are not explicitly mentioned in the original text.
Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """Double-check if any entities from the original text were missed. 
Do not infer or imagine entities that are not explicitly mentioned in the original text.
Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS[
    "summary_clusters"
] = """You are tasked with analyzing a set of entity descriptions and a given list of meta attributes. Your goal is to summarize at least one attribute entity for the entity set in the given entity descriptions. And the summarized attribute entity must match the type of at least one meta attribute in the given meta attribute list (e.g., if a meta attribute is "company", the attribute entity could be "Amazon" or "Meta", which is a kind of meta attribute "company"). And it shoud be directly relevant to the entities described in the entity description set. The relationship between the entity set and the generated attribute entity should be clear and logical.

-Steps-
1. Identify at least one attribute entity for the given entity description list. For each attribute entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type/category of the entity; choose from the given meta attribute list
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From each given entity, identify all pairs of (source_entity, target_entity) that are *clearly related* to the attribute entities identified in step 1. And there should be no relations between the attribute entities.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as given in entity list
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}


######################
-Example-
######################
Input:
Meta attribute list: ["company", "location"]
Entity description list: [("Instagram", "Instagram is a software developed by Meta, which captures and shares the world's moments. Follow friends and family to see what they're up to, and discover accounts from all over the world that are sharing things you love."), ("Facebook", "Facebook is a social networking platform launched in 2004 that allows users to connect, share updates, and engage with communities. Owned by Meta, it is one of the largest social media platforms globally, offering tools for communication, business, and advertising."), ("WhatsApp", "WhatsApp Messenger: A messaging app of Meta for simple, reliable, and secure communication. Connect with friends and family, send messages, make voice and video calls, share media, and stay in touch with loved ones, no matter where they are")]
#######
Output:
("entity"{tuple_delimiter}"Meta"{tuple_delimiter}"company"{tuple_delimiter}"Meta, formerly known as Facebook, Inc., is an American multinational technology conglomerate. It is known for its various online social media services."){record_delimiter}
("relationship"{tuple_delimiter}"Instagram"{tuple_delimiter}"Meta"{tuple_delimiter}"Instagram is a software developed by Meta."{tuple_delimiter}8.5){record_delimiter}
("relationship"{tuple_delimiter}"Facebook"{tuple_delimiter}"Meta"{tuple_delimiter}"Facebook is owned by Meta."{tuple_delimiter}9.0){record_delimiter}
("relationship"{tuple_delimiter}"WhatsApp"{tuple_delimiter}"Meta"{tuple_delimiter}"WhatsApp Messenger is a messaging app of Meta."{tuple_delimiter}8.0){record_delimiter}
#############################
-Real Data-
######################
Input:
Meta attribute list: {meta_attribute_list}
Entity description list: {entity_description_list}
#######
Output:
"""

PROMPTS[
    "structure_summary_generation"
] = """-Goal-
Generate a concise parent summary from child summaries for a structured document node.

-Rules-
1. Use only information from the provided child summaries.
2. Keep the summary factual, compact, and coherent.
3. If child summaries are empty, return "".

-Real Data-
Document Name: {doc_name}
Parent Unit ID: {unit_id}
Parent Title: {unit_title}
Child Summaries:
{child_summaries}

Output:
"""

PROMPTS[
    "structure_entity_extraction"
] = """-Goal-
Given a summary of a document section, identify semantic entities and relationships.

-Steps-
1. Identify entities from the summary.
- entity_name: Name of the entity, capitalized
- entity_type: Type/category of the entity
- entity_description: Comprehensive description grounded in the unit summary
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Identify related entity pairs from step 1.
- source_entity: entity name from step 1
- target_entity: entity name from step 1
- relationship_description: why they are related in this structure unit
- relationship_strength: numeric score of relationship strength
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return as one list separated by {record_delimiter}.
4. Finish with {completion_delimiter}.

-Real Data-
Document Name: {doc_name}
Unit ID: {unit_id}
Unit Title: {unit_title}
Summary: {input_text}
Output:
"""

PROMPTS[
    "cross_unit_relation_extraction"
] = """-Goal-
Given a parent structure unit and a child structure unit, extract only cross-unit relationships.

-Steps-
1. Use the provided entities from both units.
2. Extract only clear relationships that connect one parent entity with one child entity.
3. Do not create entity records.
4. Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
5. Return one list separated by {record_delimiter} and finish with {completion_delimiter}.

-Real Data-
Parent Unit ID: {parent_unit_id}
Parent Summary: {parent_summary}
Parent Entities: {parent_entities}

Child Unit ID: {child_unit_id}
Child Summary: {child_summary}
Child Entities: {child_entities}

Output:
"""


# TYPE的定义
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["concept", "topic", "mechanism", "application","fact", "rule", "procedure", "example", "person", "organization", "event", "location"]
PROMPTS["DEFAULT_ENTITY_TYPES_ZH"] = ["概念", "主题", "应用", "事实", "算法", "程序", "例子", "人物", "组织", "事件", "地点"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<||>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS[
    "local_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "global_map_rag_points"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1...", "score": score_value}},
        {{"description": "Description of point 2...", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1", "score": score_value}},
        {{"description": "Description of point 2", "score": score_value}}
    ]
}}
"""

PROMPTS[
    "global_reduce_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "naive_rag_response"
] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""

PROMPTS["dual_query_decomposition"] = """You are a query planner for a structured-document retrieval system.

Break the user query into three parts:
1. detail_subquestions: short subquestions that should be answered from specific facts.
2. macro_subquestions: broader, thematic, general subquestions that should be answered from summary evidence.
3. keywords: concise search keywords or phrases. Be specific and include key terms that are likely to be useful for retrieval in a knowledge graph. These keywords are going to match against entity names, so they should be chosen to maximize recall of relevant information while minimizing irrelevant matches.
4. you may also include the original user query in detail/macro subquestions if you think it is useful for retrieval.

Return only valid JSON with this shape:
{{
    "detail_subquestions": ["..."],
    "macro_subquestions": ["..."],
    "keywords": ["..."]
}}

######################
-Example-
######################
User query: "What were the main causes of World War I, and how did the assassination of Archduke Franz Ferdinand trigger the conflict?"

Output:
{{
    "detail_subquestions": [
        "When was Archduke Franz Ferdinand assassinated?",
        "Who assassinated Archduke Franz Ferdinand?",
        "What alliances existed in Europe before WWI?",
        "What was the July Crisis?"
    ],
    "macro_subquestions": [
        "What were the primary long-term causes of World War I?",
        "How did the assassination trigger the broader war?",
        "What role did alliance systems play in escalating the conflict?"
    ],
    "keywords": [
        "World War I",
        "Archduke Franz Ferdinand",
        "Assassination",
        "alliance system 1914",
        "July Crisis",
    ]
}}

######################
-Real Query-
######################
User query:
{query}
"""

PROMPTS["dual_query_decomposition_with_keyword_relations"] = """You are a query planner for a document retrieval system.

Break the user query into four parts:
1. detail_subquestions: subquestions that should be answered from specific facts. You may include the original query if it is specific.
2. macro_subquestions: broader, thematic, general subquestions. You may include the original query if it is thematic and broad.
3. keywords: standalone entity keywords that should be retrieved directly.
4. keyword_relations: binary keyword pairs in the form of head->tail that indicate a relation path to retrieve.

Important for keyword_relations:
- Use pairs that reflect the relation logic in the query.
- Pairs may be same granularity (name->organization) or cross-level (theme->topic, topic->example).
- If a query involves multiple logical steps, decompose it to multiple binary relations.
- If part of the query cannot be expressed as a binary relation, put it in standalone keywords.

Return only valid JSON with this shape:
{{
    "detail_subquestions": ["..."],
    "macro_subquestions": ["..."],
    "keywords": ["..."],
    "keyword_relations": [
        {{"head": "...", "tail": "..."}}
    ]
}}

Rules:
- Keep keywords concise and retrieval-oriented.
- Be as comprehensive as possible, but do not include unrelated subquestions or keywords.
- Do not include explanations, only JSON in the provided format. If any type of subquestion or keyword is not applicable, return an empty list for that type.

--- Example ---

User query: "How did rising food prices contribute to the French Revolution and eventually lead to the Reign of Terror?"

Output:
{{
    "detail_subquestions": [
        "How did rising food prices contribute to the French Revolution?",
        "How did the French Revolution lead to the Reign of Terror?"
    ],
    "macro_subquestions": [
        "What is the background of the French revolution?"
    ],
    "keywords": [
        "French Revolution",
        "food prices",
        "Reign of Terror",
        "economic hardship"
    ],
    "keyword_relations": [
        {{"head": "food prices", "tail": "French Revolution"}},
        {{"head": "French Revolution", "tail": "Reign of Terror"}}
    ]
}}

--- End of Example ---

Now process the following user query. Return only valid JSON, no other text.

User query:
{query}
"""

PROMPTS["dual_macro_doc_selection"] = """You are selecting which documents in a knowledge base are likely to contain information relevant to a question.

You will be given:
1. A macro question.
2. A list of documents, each with a doc_id, doc_name, and doc_description.

Your job:
- Select the doc_ids whose description suggests they may contain information useful for answering the macro subquestion.
- If none seem relevant, return an empty list.

Return only valid JSON with this shape:
{{
    "doc_ids": ["..."]
}}

Rules:
- `doc_ids` must only contain ids that appear in the input list.
- Be inclusive rather than exclusive: if in doubt, include the document.
- Only return JSON, do not include any other text.

Question: {macro_subquestion}

Documents:
{documents}
"""

PROMPTS["dual_macro_traversal"] = """You are navigating a structured document tree to collect macro evidence for a question.

You will be given:
1. A question.
2. The current organized useful information collected so far.
3. The current level child sections from a structure tree.

Your job at this step:
1. Decide whether the current level summaries already provide sufficient evidence to answer the question.
2. If not sufficient, choose which section ids from the current level are most likely to contain relevant information in their child sections.
3. Update the organized useful information using the evidence seen so far.

Return only valid JSON with this shape:
{{
    "sufficient": true,
    "section_ids": ["..."],
    "useful_information": "..."
}}

Rules:
- `section_ids` must only contain ids that appear in the current level input.
- If the evidence is already sufficient, set `sufficient` to true and `section_ids` to [].
- If the evidence is not sufficient and nothing looks relevant, set `sufficient` to false and `section_ids` to [].
- Keep `useful_information` concise but cumulative and well organized.
- Do not invent evidence that is not supported by the provided summaries.

Document: {doc_name}
Document summary: {doc_description}
Question: {macro_subquestion}

Useful information so far:
{useful_information}

Current level sections:
{current_sections}
"""

PROMPTS["dual_rag_response"] = """You are a helpful assistant.
Use the provided evidence to answer the question. The evidence may include detail evidence from vector retrieval, keyword matches, and macro evidence.

Do not include information where the supporting evidence for it is not provided. Directly return the answer with no additional commentary.

---Target response length and format---
{response_type}

---Evidence---
{context_data}

Write the final answer directly and keep it grounded in the evidence.
"""

PROMPTS["dual_leaf_detail_question_answer"] = """You are a helpful assistant for structured document retrieval.

You will be given one leaf section from a structured document, its text content, and a list of detail questions.

Task:
1. Answer every detail question using only the section text.
2. If a question is not answered by this section, answer with "Not found in this section.".
3. Return only valid JSON with this shape:
{{
    "section_id": "...",
    "section_title": "...",
    "answers": [
        {{"question": "...", "answer": "..."}}
    ],
    "useful_information": "A concise synthesis of the answers grounded in the section text."
}}

Rules:
- Keep each answer concise, factual, and grounded in the section text.
- Do not invent facts that are not explicitly supported by the text.
- If all answers are "Not found in this section.", set "useful_information" to an empty string.
- Return JSON only. Do not include markdown or extra commentary.

Section ID: {section_id}
Section Title: {section_title}
Section Text:
{section_text}

Detail Questions:
{detail_questions}

Output:
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]
