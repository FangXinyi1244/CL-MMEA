"""
针对MMEA演示系统的Neo4j导入脚本（最终版）
策略：最大化利用现有数据，优化展示效果
"""

import os
import sys
import argparse
import re
import urllib.parse
from tqdm import tqdm
from neo4j import GraphDatabase


class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.session = self.driver.session()
        print(f"✓ 已连接到Neo4j: {uri}")
    
    def close(self):
        if self.session:
            self.session.close()
        if self.driver:
            self.driver.close()
        print("✓ 连接已关闭")
    
    def clear_database(self):
        print("⚠ 清空数据库...")
        self.session.run("MATCH (n) DETACH DELETE n")
        print("✓ 数据库已清空")
    
    def create_constraints(self):
        print("创建约束和索引...")
        try:
            self.session.run(
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.entityId IS UNIQUE"
            )
        except:
            try:
                self.session.run(
                    "CREATE CONSTRAINT ON (e:Entity) ASSERT e.entityId IS UNIQUE"
                )
            except:
                pass
        
        # 创建URI索引
        try:
            self.session.run(
                "CREATE INDEX entity_uri IF NOT EXISTS "
                "FOR (e:Entity) ON (e.uri)"
            )
        except:
            pass
        
        print("✓ 约束和索引创建完成")
    
    def uri_to_name(self, uri):
        """从URI提取可读名称"""
        name = uri.split('/')[-1].split('#')[-1]
        name = urllib.parse.unquote(name)
        name = name.replace('_', ' ')
        return name
    
    def uri_to_prop_name(self, uri):
        """从属性URI提取属性名"""
        name = uri.split('/')[-1].split('#')[-1]
        name = urllib.parse.unquote(name)
        return name
    
    def rel_id_to_type(self, rel_id):
        """将关系ID转换为关系类型"""
        return f"REL_{rel_id}"
    
    def load_entity_ids(self, file_path):
        """加载实体ID映射"""
        entities = {}
        if not os.path.exists(file_path):
            return entities
        
        print(f"  加载实体: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="  读取实体"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ent_id = int(parts[0])
                    uri = parts[1]
                    name = self.uri_to_name(uri)
                    entities[ent_id] = {
                        'uri': uri,
                        'name': name
                    }
        return entities
    
    def load_triples(self, file_path):
        """加载三元组"""
        triples = []
        if not os.path.exists(file_path):
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
        """加载对齐关系"""
        alignments = []
        if not os.path.exists(file_path):
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
        """加载属性"""
        attributes = {}
        if not os.path.exists(file_path):
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
        """导入实体节点"""
        print(f"导入实体节点 ({dataset_label})...")
        entity_list = list(entities.items())
        total = len(entity_list)
        
        for i in tqdm(range(0, total, batch_size), desc="  导入实体"):
            batch = entity_list[i:i+batch_size]
            
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
    
    def import_triples_batch(self, triples, entities_dict, rel_prefix="KG1", batch_size=1000):
        """导入三元组关系"""
        print(f"导入三元组关系 ({rel_prefix})...")
        
        # 验证实体存在
        valid_triples = []
        for head, rel, tail in triples:
            if head in entities_dict and tail in entities_dict:
                valid_triples.append((head, rel, tail))
        
        total = len(valid_triples)
        print(f"  有效三元组: {total}/{len(triples)}")
        
        if total == 0:
            return
        
        # 按关系类型分组
        from collections import defaultdict
        rel_groups = defaultdict(list)
        
        for head, rel, tail in valid_triples:
            rel_type = self.rel_id_to_type(rel)
            rel_groups[rel_type].append((head, tail, rel, rel_prefix))
        
        print(f"  关系类型数: {len(rel_groups)}")
        
        # 导入每种关系类型
        for rel_type, rel_list in tqdm(rel_groups.items(), desc="  导入关系"):
            rel_count = len(rel_list)
            
            for i in range(0, rel_count, batch_size):
                batch = rel_list[i:i+batch_size]
                
                query = f"""
                UNWIND $batch AS item
                MATCH (h:Entity {{entityId: item.head}})
                MATCH (t:Entity {{entityId: item.tail}})
                MERGE (h)-[r:{rel_type}]->(t)
                SET r.relationId = item.rel,
                    r.prefix = item.prefix
                """
                
                params = {
                    'batch': [
                        {'head': h, 'tail': t, 'rel': r, 'prefix': p}
                        for h, t, r, p in batch
                    ]
                }
                
                self.session.run(query, **params)
        
        print(f"✓ 导入 {total} 个关系")
    
    def import_alignment_batch(self, alignments, batch_size=1000):
        """导入对齐关系"""
        print(f"导入实体对齐关系...")
        total = len(alignments)
        
        if total == 0:
            return
        
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
                'batch': [{'id1': id1, 'id2': id2} for id1, id2 in batch]
            }
            
            self.session.run(query, **params)
        
        print(f"✓ 导入 {total} 个对齐关系")
    
    def import_attributes_as_properties(self, attributes, entities_dict, batch_size=500):
        """导入属性为节点属性"""
        print(f"导入实体属性（存储为数组）...")
        
        uri_to_id = {data['uri']: ent_id for ent_id, data in entities_dict.items()}
        
        entity_props = {}
        for uri, attrs in attributes.items():
            if uri in uri_to_id:
                ent_id = uri_to_id[uri]
                if ent_id not in entity_props:
                    entity_props[ent_id] = []
                
                for attr_uri in attrs:
                    prop_name = self.uri_to_prop_name(attr_uri)
                    if prop_name not in entity_props[ent_id]:
                        entity_props[ent_id].append(prop_name)
        
        total = len(entity_props)
        print(f"  有属性的实体数: {total}")
        
        if total == 0:
            return
        
        # 存储为数组
        count = 0
        for ent_id, prop_list in tqdm(entity_props.items(), desc="  更新属性", total=total):
            query = """
            MATCH (e:Entity {entityId: $id})
            SET e.propertyList = $props
            """
            try:
                self.session.run(query, id=ent_id, props=prop_list)
                count += 1
            except Exception as e:
                print(f"  ⚠ 实体 {ent_id} 属性更新失败: {e}")
        
        print(f"✓ 更新 {count} 个实体的属性")
    
    def import_dataset(self, data_dir, dataset_name, language_pair, clear=False):
        """导入整个数据集"""
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
        
        # 1. 实体（KG1）
        print("\n[1/6] 处理知识图谱 1 的实体...")
        ent_file_1 = os.path.join(base_dir, "ent_ids_1")
        entities_1 = self.load_entity_ids(ent_file_1)
        if entities_1:
            label_1 = f"{dataset_name}_{language_pair.split('_')[0].upper()}"
            self.import_entities_batch(entities_1, label_1)
        
        # 2. 实体（KG2）
        print("\n[2/6] 处理知识图谱 2 的实体...")
        ent_file_2 = os.path.join(base_dir, "ent_ids_2")
        entities_2 = self.load_entity_ids(ent_file_2)
        if entities_2:
            max_id_1 = max(entities_1.keys()) if entities_1 else 0
            entities_2_offset = {k + max_id_1 + 1: v for k, v in entities_2.items()}
            label_2 = f"{dataset_name}_{language_pair.split('_')[1].upper()}"
            self.import_entities_batch(entities_2_offset, label_2)
            entities_2 = entities_2_offset
        
        all_entities = {**entities_1, **entities_2}
        
        # 3. 三元组（KG1）
        print("\n[3/6] 处理知识图谱 1 的三元组...")
        triple_file_1 = os.path.join(base_dir, "triples_1")
        triples_1 = self.load_triples(triple_file_1)
        if triples_1:
            self.import_triples_batch(triples_1, all_entities, rel_prefix="KG1")
        
        # 4. 三元组（KG2）
        print("\n[4/6] 处理知识图谱 2 的三元组...")
        triple_file_2 = os.path.join(base_dir, "triples_2")
        triples_2 = self.load_triples(triple_file_2)
        if triples_2:
            if entities_1 and entities_2:
                max_id_1 = max(entities_1.keys())
                triples_2_offset = [(h + max_id_1 + 1, r, t + max_id_1 + 1) for h, r, t in triples_2]
                self.import_triples_batch(triples_2_offset, all_entities, rel_prefix="KG2")
        
        # 5. 对齐关系
        print("\n[5/6] 处理实体对齐关系...")
        ill_file = os.path.join(base_dir, "ill_ent_ids")
        alignments = self.load_alignment(ill_file)
        if alignments:
            if entities_1 and entities_2:
                max_id_1 = max(entities_1.keys())
                alignments_offset = [(id1, id2 + max_id_1 + 1) for id1, id2 in alignments]
                self.import_alignment_batch(alignments_offset)
        
        # 6. 属性
        print("\n[6/6] 处理实体属性...")
        attr_file_1 = os.path.join(base_dir, "training_attrs_1")
        attrs_1 = self.load_attributes(attr_file_1)
        if attrs_1:
            self.import_attributes_as_properties(attrs_1, entities_1)
        
        attr_file_2 = os.path.join(base_dir, "training_attrs_2")
        attrs_2 = self.load_attributes(attr_file_2)
        if attrs_2:
            self.import_attributes_as_properties(attrs_2, entities_2)
        
        # 统计
        self.get_statistics()
        
        print("\n" + "="*60)
        print(f"✅ 数据集 {dataset_name}/{language_pair} 导入完成!")
        print("="*60)
    
    def get_statistics(self):
        """统计信息"""
        print("\n数据库统计信息:")
        print("-" * 60)
        
        result = self.session.run("MATCH (n) RETURN count(n) AS count")
        print(f"总节点数: {result.single()['count']}")
        
        result = self.session.run("MATCH (e:Entity) RETURN count(e) AS count")
        print(f"实体节点数: {result.single()['count']}")
        
        result = self.session.run("MATCH ()-[r]->() RETURN count(r) AS count")
        print(f"总关系数: {result.single()['count']}")
        
        result = self.session.run("MATCH ()-[r:ALIGNS_WITH]->() RETURN count(r) AS count")
        print(f"对齐关系数: {result.single()['count']}")
        
        print("\n实体分布:")
        result = self.session.run(
            "MATCH (e:Entity) RETURN e.dataset AS dataset, count(e) AS count ORDER BY dataset"
        )
        for record in result:
            print(f"  {record['dataset']}: {record['count']}")
        
        print("\n关系类型分布 (TOP 10):")
        result = self.session.run(
            "MATCH ()-[r]->() RETURN type(r) AS relType, count(r) AS count "
            "ORDER BY count DESC LIMIT 10"
        )
        for record in result:
            print(f"  {record['relType']}: {record['count']}")
        
        print("-" * 60)
    
    def print_demo_queries(self):
        """打印演示用的查询"""
        print("\n" + "="*60)
        print("演示查询（复制到Neo4j Browser或Dashboard）:")
        print("="*60)
        
        print("\n1. 【图可视化】查看整体结构（采样50个节点）:")
        print("MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 50")
        
        print("\n2. 【柱状图】查看两个知识图谱的实体数量:")
        print("MATCH (e:Entity) RETURN e.dataset AS 数据集, count(e) AS 实体数量")
        
        print("\n3. 【图可视化】查看对齐关系:")
        print("MATCH (e1)-[r:ALIGNS_WITH]->(e2) RETURN e1, r, e2 LIMIT 100")
        
        print("\n4. 【表格】查看实体详情（含属性）:")
        print("MATCH (e:Entity) "
              "RETURN e.name AS 名称, e.dataset AS 数据集, e.propertyList AS 属性列表 "
              "LIMIT 20")
        
        print("\n5. 【图可视化】查看特定实体的邻居:")
        print("MATCH (e:Entity {name: '阿卜杜拉·居尔'})-[r]-(n) RETURN e, r, n")
        
        print("\n6. 【统计】查看关系类型分布:")
        print("MATCH ()-[r]->() RETURN type(r) AS 关系类型, count(r) AS 数量 "
              "ORDER BY 数量 DESC LIMIT 20")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="MMEA数据集Neo4j导入工具（演示版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导入DBP15K数据集
  python neo4j_import.py --dataset dbp15k --language zh_en --clear
  
  # 查看统计
  python neo4j_import.py --stats
        """
    )
    
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687",
                        help="Neo4j URI")
    parser.add_argument("--user", type=str, default="neo4j",
                        help="用户名")
    parser.add_argument("--password", type=str, default="password",
                        help="密码")
    
    parser.add_argument("--dataset", type=str, default="dbp15k",
                        choices=["dbp15k", "mmkb"],
                        help="数据集")
    parser.add_argument("--language", type=str, default="zh_en",
                        help="语言对")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据目录")
    
    parser.add_argument("--clear", action="store_true",
                        help="清空数据库")
    parser.add_argument("--stats", action="store_true",
                        help="仅查看统计")
    
    args = parser.parse_args()
    
    importer = Neo4jImporter(args.uri, args.user, args.password)
    
    try:
        if args.stats:
            importer.get_statistics()
            importer.print_demo_queries()
        else:
            dataset_name = "DBP15K" if args.dataset == "dbp15k" else "mmkb-datasets"
            importer.import_dataset(
                data_dir=args.data_dir,
                dataset_name=dataset_name,
                language_pair=args.language,
                clear=args.clear
            )
            importer.print_demo_queries()
    finally:
        importer.close()


if __name__ == "__main__":
    main()
