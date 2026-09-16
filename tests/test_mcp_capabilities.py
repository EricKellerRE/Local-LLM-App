import asyncio
import unittest
from types import SimpleNamespace

from local_model_app.mcp_manager import ActivePlugin, McpPluginManager
from local_model_app.mcp_plugins import DiscoveredTool, McpPluginRegistry


class Dumpable:
    def __init__(self, **data):
        self.data = data

    def model_dump(self, **kwargs):
        return self.data


class FakeClient:
    def __init__(self) -> None:
        self.reads = []
        self.prompts = []

    async def read_resource(self, uri):
        self.reads.append(uri)
        return SimpleNamespace(contents=[Dumpable(uri=uri, text="resource text")])

    async def get_prompt(self, name, arguments):
        self.prompts.append((name, arguments))
        return SimpleNamespace(description="Rendered prompt", messages=[Dumpable(role="user", content={"text": "hello"})])


class FakeDiscoveryClient:
    async def list_resources(self, *, cursor=None):
        return SimpleNamespace(resources=[SimpleNamespace(
            name="handbook", title="Handbook", uri="docs://handbook", description="Operating handbook",
        )], next_cursor=None)

    async def list_resource_templates(self, *, cursor=None):
        return SimpleNamespace(resource_templates=[SimpleNamespace(
            name="case", title=None, uri_template="case://{id}", description="Read a case",
        )], next_cursor=None)

    async def list_prompts(self, *, cursor=None):
        argument = SimpleNamespace(name="topic", title=None, description="Review topic", required=True)
        return SimpleNamespace(prompts=[SimpleNamespace(
            name="review", title="Review", description="Review a topic", arguments=[argument],
        )], next_cursor=None)


class McpCapabilityTests(unittest.TestCase):
    @staticmethod
    def manager_for(tool, client):
        policy = SimpleNamespace(default_access="allow", allow_tools=[], deny_tools=[])
        registry = SimpleNamespace(get_plugin=lambda plugin_id: SimpleNamespace(manifest=SimpleNamespace(policy=policy)))
        manager = McpPluginManager(registry)
        manager._active[tool.plugin_id] = ActivePlugin(clients={tool.server_id: client}, tools=[tool])
        return manager

    def test_static_resource_is_read_through_host_adapter(self) -> None:
        tool = DiscoveredTool(
            exposed_name="demo__resource_handbook", plugin_id="demo.plugin", server_id="demo",
            native_name="resource:handbook", title="Handbook", description="Read handbook",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema=None, annotations={"readOnlyHint": True}, kind="resource", target="docs://handbook",
        )
        client = FakeClient()

        result = asyncio.run(self.manager_for(tool, client).call_tool(tool.exposed_name, {}))

        self.assertEqual(client.reads, ["docs://handbook"])
        self.assertEqual(result["structuredContent"]["contents"][0]["text"], "resource text")

    def test_prompt_is_rendered_through_host_adapter(self) -> None:
        tool = DiscoveredTool(
            exposed_name="demo__prompt_review", plugin_id="demo.plugin", server_id="demo",
            native_name="prompt:review", title="Review", description="Render review prompt",
            input_schema={"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
            output_schema=None, annotations={"readOnlyHint": True}, kind="prompt", target="review",
        )
        client = FakeClient()

        result = asyncio.run(self.manager_for(tool, client).call_tool(tool.exposed_name, {"topic": "stability"}))

        self.assertEqual(client.prompts, [("review", {"topic": "stability"})])
        self.assertEqual(result["structuredContent"]["description"], "Rendered prompt")

    def test_resources_templates_and_prompts_become_namespaced_capabilities(self) -> None:
        registry = McpPluginRegistry.__new__(McpPluginRegistry)
        plugin = SimpleNamespace(manifest=SimpleNamespace(id="demo.plugin", tool_namespace="demo"))
        server = SimpleNamespace(id="demo")
        client = FakeDiscoveryClient()

        resources = asyncio.run(registry.discover_connected_resources(plugin, server, client))
        prompts = asyncio.run(registry.discover_connected_prompts(plugin, server, client))

        self.assertEqual([item.kind for item in resources], ["resource", "resource_template"])
        self.assertTrue(all(item.exposed_name.startswith("demo__") for item in [*resources, *prompts]))
        self.assertEqual(resources[1].input_schema["properties"]["uri"]["pattern"], r"^case://.+?$")
        self.assertEqual(prompts[0].input_schema["required"], ["topic"])

    def test_resources_and_prompts_are_not_model_selected_tools(self) -> None:
        normal = DiscoveredTool(
            exposed_name="demo__tool", plugin_id="demo.plugin", server_id="demo",
            native_name="tool", title=None, description="Tool", input_schema={"type": "object"},
            output_schema=None, annotations={},
        )
        resource = DiscoveredTool(
            exposed_name="demo__resource", plugin_id="demo.plugin", server_id="demo",
            native_name="resource:one", title=None, description="Resource", input_schema={"type": "object"},
            output_schema=None, annotations={}, kind="resource", target="demo://one",
        )
        prompt = DiscoveredTool(
            exposed_name="demo__prompt", plugin_id="demo.plugin", server_id="demo",
            native_name="prompt:one", title=None, description="Prompt", input_schema={"type": "object"},
            output_schema=None, annotations={}, kind="prompt", target="one",
        )
        policy = SimpleNamespace(default_access="allow", allow_tools=[], deny_tools=[])
        registry = SimpleNamespace(
            plugins=[], get_plugin=lambda plugin_id: SimpleNamespace(manifest=SimpleNamespace(policy=policy)),
        )
        manager = McpPluginManager(registry)
        manager._active["demo.plugin"] = ActivePlugin(tools=[normal, resource, prompt])

        self.assertEqual(manager.tools_for_plugins(["demo.plugin"]), [normal])
        self.assertEqual(manager.resources_for_plugins(["demo.plugin"]), [resource])
        self.assertEqual(manager.prompts_for_plugins(["demo.plugin"]), [prompt])


if __name__ == "__main__":
    unittest.main()
