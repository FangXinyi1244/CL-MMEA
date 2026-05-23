"""
Neo4j 数据导入脚本 - 用于MMEA项目数据集（DBP15K, MMKB）
将知识图谱数据导入Neo4j图数据库，便于前端演示系统调用

使用方法：
    # 1. 启动Neo4j数据库
    # 2. 修改下面的连接配置
    # 3. 运行脚本
    python neo4j_import.py --dataset dbp15k --language zh_en
    python neo4j_import.py --dataset mmkb --language FB15K_DB15K
"""

import os
import sys
import argparse
from tqdm import tqdm
from neo4j import GraphDatabase, IN_MEMORY


class Neo4jImporter:
    """Neo4j数据导入器"""
    
    def __init__(self, uri, user, password):
        """
        初始化Neo4j连接
        
        Args:
            uri: Neo4j连接URI，如 "bolt://localhost:7687"
            user: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.session = self.driver.session()
        print(f"✓ 已连接到Neo4j: {uri}")
    
    def close(self):
        """关闭连接"""
        if self.session:
            self.session.close()
        if self.driver:
            self.driver.close()
        print("✓ 连接已关闭")
    
    def clear_database(self):
        """清空数据库"""
        print("⚠ 清空数据库...")
        self.session.run("MATCH (n) DETACH DELETE n")
        print("✓ 数据库已清空")
    
    def create_constraints(self):
        """创建约束和索引"""
        print("创建约束和索引...")
        
        # 创建实体ID唯一性约束
        try:
            self.session.run(
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.entityId IS UNIQUE"
            )
        except:
            # 兼容旧版本Neo4j
            try:
                self.session.run(
                    "CREATE CONSTRAINT ON (e:Entity) ASSERT e.entityId IS UNIQUE"
                )
            except Exception as e:
                print(f"  约束可能已存在: {e}")
        
        # 创建URI索引
        try:
            self.session.run(
                "CREATE INDEX entity_uri_index IF NOT EXISTS "
                "FOR (e:Entity) ON (e.uri)"
            )
        except:
            print("  索引可能已存在")
        
        print("✓ 约束和索引创建完成")
    
    def load_entity_ids(self, file_path):
        """
        加载实体ID映射文件
        
        Args:
            file_path: 文件路径，格式为 "ID\tURI"
            
        Returns:
            dict: {id: {"uri": URI, "name": name}}
        """
        entities = {}
        if not os.path.exists(file_path):
            print(f"  ⚠ 文件不存在: {file_path}")
            return entities
        
        print(f"  加载实体: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="  读取实体"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ent_id = int(parts[0])
                    uri = parts[1]
                    # 从URI提取实体名称
                    name = uri.split('/')[-1].replace('_', ' ')
                    entities[ent_id] = {
                        'uri': uri,
                        'name': name
                    }
        return entities
    
    def load_triples(self, file_path):
        """
        加载三元组文件
        
        Args:
            file_path: 文件路径，格式为 "head\trelation\ttail"
            
        Returns:
            list: [(head_id, relation_id, tail_id), ...]
        """
        triples = []
        if not os.path.exists(file_path):
            print(f"  ⚠ 文件不存在: {file_path}")
            return triples
        
        print(f"  加载三元组: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="  读取三元组"):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    head = int(parts[0])
                    rel = int(parts[1])
                    tail = int(parts[2])
                    triples.append((head, rel, tail))
        return triples
    
    def load_alignment(self, file_path):
        """
        加载实体对齐文件
        
        Args:
            file_path: 文件路径，格式为 "ID1\tID2"
            
        Returns:
            list: [(id1, id2), ...]
        """
        alignments = []
        if not os.path.exists(file_path):
            print(f"  ⚠ 文件不存在: {file_path}")
            return alignments
        
        print(f"  加载对齐关系: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="  读取对齐"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    id1 = int(parts[0])
                    id2 = int(parts[1])
                    alignments.append((id1, id2))
        return alignments
    
    def load_attributes(self, file_path):
        """
        加载实体属性文件
        
        Args:
            file_path: 文件路径，格式为 "entity_uri\tattr1\tattr2..."
            
        Returns:
            dict: {entity_uri: [attr1, attr2, ...]}
        """
        attributes = {}
        if not os.path.exists(file_path):
            print(f"  ⚠ 文件不存在: {file_path}")
            return attributes
        
        print(f"  加载属性: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="  读取属性"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_uri = parts[0]
                    attrs = parts[1:]
                    attributes[entity_uri] = attrs
        return attributes
    
    def import_entities_batch(self, entities, dataset_label, batch_size=1000):
        """
        批量导入实体节点
        
        Args:
            entities: 实体字典 {id: {uri, name}}
            dataset_label: 数据集标签（如 "DBP_ZH", "DBP_EN"）
            batch_size: 批处理大小
        """
        print(f"导入实体节点 ({dataset_label})...")
        
        # 将实体分批
        entity_list = list(entities.items())
        total = len(entity_list)
        
        for i in tqdm(range(0, total, batch_size), desc="  导入实体"):
            batch = entity_list[i:i+batch_size]
            
            # 构建Cypher查询
            query = """
            UNWIND $batch AS item
            MERGE (e:Entity {entityId: item.id})
            SET e.uri = item.uri,
                e.name = item.name,
                e.dataset = item.dataset
            SET e:%s
            """ % dataset_label
            
            params = {
                'batch': [
                    {
                        'id': ent_id,
                        'uri': data['uri'],
                        'name': data['name'],
                        'dataset': dataset_label
                    }
                    for ent_id, data in batch
                ]
            }
            
            self.session.run(query, **params)
        
        print(f"✓ 导入 {total} 个实体")
    
    def import_triples_batch(self, triples, entities_dict, rel_prefix="", batch_size=1000):
        """
        批量导入三元组关系
        
        Args:
            triples: 三元组列表 [(head, rel, tail), ...]
            entities_dict: 实体字典，用于验证实体存在
            rel_prefix: 关系前缀
            batch_size: 批处理大小
        """
        print(f"导入三元组关系...")
        
        # 构建批处理数据
        valid_triples = []
        for head, rel, tail in triples:
            if head in entities_dict and tail in entities_dict:
                valid_triples.append((head, rel, tail))
        
        total = len(valid_triples)
        print(f"  有效三元组: {total}/{len(triples)}")
        
        for i in tqdm(range(0, total, batch_size), desc="  导入关系"):
            batch = valid_triples[i:i+batch_size]
            
            query = """
            UNWIND $batch AS item
            MATCH (h:Entity {entityId: item.head})
            MATCH (t:Entity {entityId: item.tail})
            MERGE (h)-[r:REL {relationId: item.rel}]->(t)
            SET r.rel_prefix = item.prefix
            """
            
            params = {
                'batch': [
                    {
                        'head': head,
                        'rel': rel,
                        'tail': tail,
                        'prefix': rel_prefix
                    }
                    for head, rel, tail in batch
                ]
            }
            
            self.session.run(query, **params)
        
        print(f"✓ 导入 {total} 个关系")
    
    def import_alignment_batch(self, alignments, batch_size=1000):
        """
        批量导入实体对齐关系
        
        Args:
            alignments: 对齐列表 [(id1, id2), ...]
            batch_size: 批处理大小
        """
        print(f"导入实体对齐关系...")
        
        total = len(alignments)
        
        for i in tqdm(range(0, total, batch_size), desc="  导入对齐"):
            batch = alignments[i:i+batch_size]
            
            query = """
            UNWIND $batch AS item
            MATCH (e1:Entity {entityId: item.id1})
            MATCH (e2:Entity {entityId: item.id2})
            MERGE (e1)-[r:ALIGNS_WITH]->(e2)
            SET r.confidence = 1.0
            """
            
            params = {
                'batch': [
                    {'id1': id1, 'id2': id2}
                    for id1, id2 in batch
                ]
            }
            
            self.session.run(query, **params)
        
        print(f"✓ 导入 {total} 个对齐关系")
    
    def import_attributes_batch(self, attributes, entities_dict, batch_size=1000):
        """
        批量导入实体属性
        
        Args:
            attributes: 属性字典 {uri: [attrs, ...]}
            entities_dict: 实体字典 {id: {uri, name}}
            batch_size: 批处理大小
        """
        print(f"导入实体属性...")
        
        # 构建URI到实体ID的映射
        uri_to_id = {data['uri']: ent_id for ent_id, data in entities_dict.items()}
        
        attr_list = []
        for uri, attrs in attributes.items():
            if uri in uri_to_id:
                ent_id = uri_to_id[uri]
                for attr in attrs:
                    attr_name = attr.split('/')[-1].replace('_', ' ')
                    attr_list.append((ent_id, attr_name))
        
        total = len(attr_list)
        print(f"  有效属性: {total}")
        
        for i in tqdm(range(0, total, batch_size), desc="  导入属性"):
            batch = attr_list[i:i+batch_size]
            
            query = """
            UNWIND $batch AS item
            MATCH (e:Entity {entityId: item.entId})
            MERGE (a:Attribute {name: item.attrName})
            MERGE (e)-[r:HAS_ATTR]->(a)
            """
            
            params = {
                'batch': [
                    {'entId': ent_id, 'attrName': attr_name}
                    for ent_id, attr_name in batch
                ]
            }
            
            self.session.run(query, **params)
        
        print(f"✓ 导入 {total} 个属性关系")
    
    def import_dataset(self, data_dir, dataset_name, language_pair, clear=False):
        """
        导入整个数据集
        
        Args:
            data_dir: 数据目录
            dataset_name: 数据集名称 ("DBP15K" 或 "MMKB")
            language_pair: 语言对 ("zh_en", "ja_en", "fr_en" 或 "FB15K_DB15K" 等)
            clear: 是否清空数据库
        """
        print("\n" + "="*60)
        print(f"导入数据集: {dataset_name}/{language_pair}")
        print("="*60)
        
        if clear:
            self.clear_database()
        
        self.create_constraints()
        
        base_dir = os.path.join(data_dir, dataset_name, language_pair)
        
        if not os.path.exists(base_dir):
            print(f"❌ 数据目录不存在: {base_dir}")
            return
        
        # 1. 加载并导入实体（知识图谱1）
        print("\n[1/5] 处理知识图谱 1 的实体...")
        ent_file_1 = os.path.join(base_dir, "ent_ids_1")
        entities_1 = self.load_entity_ids(ent_file_1)
        if entities_1:
            label_1 = f"{dataset_name}_{language_pair.split('_')[0].upper()}"
            self.import_entities_batch(entities_1, label_1)
        
        # 2. 加载并导入实体（知识图谱2）
        print("\n[2/5] 处理知识图谱 2 的实体...")
        ent_file_2 = os.path.join(base_dir, "ent_ids_2")
        entities_2 = self.load_entity_ids(ent_file_2)
        if entities_2:
            # 重新编号实体ID，避免冲突（KG2的ID需要偏移）
            max_id_1 = max(entities_1.keys()) if entities_1 else 0
            entities_2_offset = {k + max_id_1 + 1: v for k, v in entities_2.items()}
            label_2 = f"{dataset_name}_{language_pair.split('_')[1].upper()}"
            self.import_entities_batch(entities_2_offset, label_2)
            # 更新entities_2的key
            entities_2 = entities_2_offset
        
        # 合并实体字典
        all_entities = {**entities_1, **entities_2}
        
        # 3. 加载并导入三元组（KG1）
        print("\n[3/5] 处理知识图谱 1 的三元组...")
        triple_file_1 = os.path.join(base_dir, "triples_1")
        triples_1 = self.load_triples(triple_file_1)
        if triples_1:
            self.import_triples_batch(triples_1, all_entities, rel_prefix="KG1")
        
        # 4. 加载并导入三元组（KG2）
        print("\n[4/5] 处理知识图谱 2 的三元组...")
        triple_file_2 = os.path.join(base_dir, "triples_2")
        triples_2 = self.load_triples(triple_file_2)
        if triples_2:
            # 三元组中的实体ID也需要偏移
            if entities_1 and entities_2:
                max_id_1 = max(entities_1.keys())
                triples_2_offset = [(h + max_id_1 + 1, r, t + max_id_1 + 1) for h, r, t in triples_2]
                self.import_triples_batch(triples_2_offset, all_entities, rel_prefix="KG2")
        
        # 5. 加载并导入对齐关系
        print("\n[5/5] 处理实体对齐关系...")
        ill_file = os.path.join(base_dir, "ill_ent_ids")
        alignments = self.load_alignment(ill_file)
        if alignments:
            # 对齐关系中的实体ID也需要偏移
            if entities_1 and entities_2:
                max_id_1 = max(entities_1.keys())
                alignments_offset = [(id1, id2 + max_id_1 + 1) for id1, id2 in alignments]
                self.import_alignment_batch(alignments_offset)
        
        # 6. 可选：导入属性
        print("\n[可选] 处理实体属性...")
        attr_file_1 = os.path.join(base_dir, "training_attrs_1")
        attrs_1 = self.load_attributes(attr_file_1)
        if attrs_1:
            self.import_attributes_batch(attrs_1, entities_1)
        
        print("\n" + "="*60)
        print(f"✅ 数据集 {dataset_name}/{language_pair} 导入完成!")
        print("="*60)
    
    def get_statistics(self):
        """获取数据库统计信息"""
        print("\n数据库统计信息:")
        print("-" * 60)
        
        # 节点统计
        result = self.session.run("MATCH (n) RETURN count(n) AS count")
        node_count = result.single()['count']
        print(f"总节点数: {node_count}")
        
        # 实体节点统计
        result = self.session.run("MATCH (e:Entity) RETURN count(e) AS count")
        entity_count = result.single()['count']
        print(f"实体节点数: {entity_count}")
        
        # 关系统计
        result = self.session.run("MATCH ()-[r]->() RETURN count(r) AS count")
        rel_count = result.single()['count']
        print(f"总关系数: {rel_count}")
        
        # 对齐关系统计
        result = self.session.run("MATCH ()-[r:ALIGNS_WITH]->() RETURN count(r) AS count")
        align_count = result.single()['count']
        print(f"对齐关系数: {align_count}")
        
        # 按标签统计实体
        result = self.session.run(
            "MATCH (e:Entity) RETURN labels(e) AS labels, count(e) AS count"
        )
        print("\n实体分布:")
        for record in result:
            print(f"  {record['labels']}: {record['count']}")
        
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="MMEA数据集Neo4j导入工具")
    
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687",
                        help="Neo4j连接URI")
    parser.add_argument("--user", type=str, default="neo4j",
                        help="Neo4j用户名")
    parser.add_argument("--password", type=str, default="password",
                        help="Neo4j密码")
    
    parser.add_argument("--dataset", type=str, default="dbp15k",
                        choices=["dbp15k", "mmkb"],
                        help="数据集类型")
    parser.add_argument("--language", type=str, default="zh_en",
                        help="语言对 (dbp15k: zh_en/ja_en/fr_en, mmkb: FB15K_DB15K/FB15K_YAGO15K)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据根目录")
    
    parser.add_argument("--clear", action="store_true",
                        help="导入前清空数据库")
    parser.add_argument("--stats", action="store_true",
                        help="仅显示统计信息")
    
    args = parser.parse_args()
    
    # 初始化导入器
    importer = Neo4jImporter(args.uri, args.user, args.password)
    
    try:
        if args.stats:
            importer.get_statistics()
        else:
            # 导入数据集
            dataset_name = "DBP15K" if args.dataset == "dbp15k" else "mmkb-datasets"
            importer.import_dataset(
                data_dir=args.data_dir,
                dataset_name=dataset_name,
                language_pair=args.language,
                clear=args.clear
            )
            # 显示统计信息
            importer.get_statistics()
    finally:
        importer.close()


if __name__ == "__main__":
    main()
