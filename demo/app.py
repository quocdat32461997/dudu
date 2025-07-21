import os
from functools import partial
from typing import List

import dotenv
from google import genai
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa
from langgraph.prebuilt import ToolNode

dotenv.load_dotenv()


def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    print("should continue", last_message)

    if last_message.tool_calls:
        return "tools"

    return END


def call_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=1.0,
        max_retries=2,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


def create_call_model(  # state and config are two default runtime params.
    # Other static parameters
    tools: List[str] = [],
):
    def call_model(
        # state and config are two default runtime params.
        state: MessagesState,
        config: RunnableConfig,
        # Other static parameters
        tools: List[str] = [],
    ):
        """
        Follow below script to get specific on run (https://langchain-ai.github.io/langgraph/how-tos/graph-api/#add-runtime-configuration). # noqa
            MODELS = {
                "anthropic": init_chat_model("anthropic:claude-3-5-haiku-latest"),
                "openai": init_chat_model("openai:gpt-4.1-mini"),
            }

            def call_model(state: MessagesState, config: RunnableConfig):
                model = config["configurable"].get("model", "anthropic")
                model = MODELS[model]
                response = model.invoke(state["messages"])
                return {"messages": [response]}
        """
        # Get model
        model = call_gemini_model()

        # Binding tools in run time
        if len(tools) > 0:
            model = model.bind_tools(tools)

        # Invoke model
        prompt = f"""
        System: You are a helpful product recommendation assistant. 

        You are to considered following products to recommend to the customer: 
        - Gold-toned pendant necklace: $15 | H&M brand | Gold | Product-id 100
        - Set of stackable rings: $18 | Zara | Mix of Gold and Silver | Product-id 101
        - Pairs of earings: $2,000 | Gucci | Pure Gold 18K. | Product-id 102
        - Hermes Birkin bag: $10,000 | Hermes | Aligator skin | Product-id 103
        - A white silk blouse: $ 25 | Ralph Lauren | Silk | Product-id 105
        
        You are asked to the following task: {state["messages"]}
        
        You must provide the product-id within [] and product name only. 

        Below are couple examples:

        """  # parse user's message here
        response = model.invoke(prompt)

        return {"messages": [response]}

    return partial(call_model, tools=tools)


# asyncio.run(main())
def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


class Agent:
    name: str = "agent"
    model_name: str = "gemini_chat"

    def __init__(
        self,
    ) -> None:
        self.workflow = StateGraph(MessagesState)

        _call_model = create_call_model(
            tools=[],
        )  # noqa
        tool_node = ToolNode([])
        # Add nodes
        self.workflow.add_node("call_model", _call_model)
        self.workflow.add_node("tools", tool_node)

        # Add edges
        self.workflow.add_edge(START, "call_model")
        self.workflow.add_conditional_edges(
            "call_model",
            should_continue,
            {
                "tools": "tools",  # f1040_agent.name,
                END: END,
            },
        )  # noqa

        self.workflow = self.workflow.compile()

    def get(self):
        return self.workflow


agent_obj = Agent()
agent = agent_obj.get()
