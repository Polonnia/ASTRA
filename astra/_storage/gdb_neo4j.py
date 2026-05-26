import asyncio
from dataclasses import dataclass
from typing import Union

from neo4j import AsyncGraphDatabase

from ..base import BaseGraphStorage
from .._utils import logger

neo4j_lock = asyncio.Lock()


def make_path_idable(path):
    return path.replace(".", "_").replace("/", "__").replace("-", "_")


@dataclass
class Neo4jStorage(BaseGraphStorage):
    def __post_init__(self):
        self.neo4j_url = self.global_config["addon_params"].get("neo4j_url", None)
        self.neo4j_auth = self.global_config["addon_params"].get("neo4j_auth", None)
        self.namespace = (
            f"{make_path_idable(self.global_config['working_dir'])}__{self.namespace}"
        )
        logger.info(f"Using the label {self.namespace} for Neo4j as identifier")
        if self.neo4j_url is None or self.neo4j_auth is None:
            raise ValueError("Missing neo4j_url or neo4j_auth in addon_params")
        self.async_driver = AsyncGraphDatabase.driver(
            self.neo4j_url, auth=self.neo4j_auth
        )

    async def _init_workspace(self):
        await self.async_driver.verify_authentication()
        await self.async_driver.verify_connectivity()

    async def index_start_callback(self):
        logger.info("Init Neo4j workspace")
        await self._init_workspace()

    async def has_node(self, node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",
                node_id=node_id,
            )
            record = await result.single()
            if record and record["exists"]:
                return True
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.entity_name = $entity_name RETURN COUNT(n) > 0 AS exists",
                entity_name=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN COUNT(r) > 0 AS exists",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def node_degree(self, node_id: str) -> int:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id "
                f"RETURN COUNT {{(n)-[]-(:{self.namespace})}} AS degree",
                node_id=node_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $src_id AND t.id = $tgt_id "
                f"RETURN COUNT {{(s)-[]-(:{self.namespace})}} + COUNT {{(t)-[]-(:{self.namespace})}} AS degree",
                src_id=src_id,
                tgt_id=tgt_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def get_node(self, node_id: str) -> Union[dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN properties(n) AS node_data",
                node_id=node_id,
            )
            record = await result.single()
            if record:
                return record["node_data"]
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.entity_name = $entity_name RETURN properties(n) AS node_data LIMIT 1",
                entity_name=node_id,
            )
            record = await result.single()
            return record["node_data"] if record else None

    async def get_all_nodes(self) -> list[dict]:
        async with self.async_driver.session() as session:
            result = await session.run(f"MATCH (n:{self.namespace}) RETURN properties(n) AS node_data")
            nodes = []
            async for record in result:
                node_data = record["node_data"]
                if node_data is None:
                    continue
                nodes.append({"id": node_data.get("id", ""), **node_data})
            return nodes

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN properties(r) AS edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["edge_data"] if record else None

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) WHERE s.id = $source_id "
                "RETURN s.id AS source, t.id AS target",
                source_id=source_node_id,
            )
            edges = []
            async for record in result:
                edges.append((record["source"], record["target"]))
            if edges:
                return edges

            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) WHERE s.entity_name = $entity_name "
                "RETURN s.id AS source, t.id AS target",
                entity_name=source_node_id,
            )
            async for record in result:
                edges.append((record["source"], record["target"]))
            return edges

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        async with self.async_driver.session() as session:
            await session.run(
                f"MERGE (n:{self.namespace} {{id: $node_id}}) "
                "SET n += $node_data",
                node_id=node_id,
                node_data=node_data,
            )

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        edge_data.setdefault("weight", 0.0)
        async with self.async_driver.session() as session:
            await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "MERGE (s)-[r:RELATED]->(t) "
                "SET r += $edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
                edge_data=edge_data,
            )

    async def index_done_callback(self):
        await self.async_driver.close()

    async def _debug_delete_all_node_edges(self):
        async with self.async_driver.session() as session:
            try:
                await session.run(f"MATCH (n:{self.namespace})-[r]-() DELETE r")
                await session.run(f"MATCH (n:{self.namespace}) DELETE n")
                logger.info(
                    f"All nodes and edges in namespace '{self.namespace}' have been deleted."
                )
            except Exception as e:
                logger.error(f"Error deleting nodes and edges: {str(e)}")
                raise
