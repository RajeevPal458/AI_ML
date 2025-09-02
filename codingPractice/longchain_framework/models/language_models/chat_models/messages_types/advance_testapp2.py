from langchain.schema import HumanMessage, AIMessage, FunctionMessage

messages = [
    HumanMessage(content="Get the weather in Paris"),
    AIMessage(content="", additional_kwargs={
        "function_call": {"name": "get_weather", "arguments": '{"location": "Paris"}'}
    }),
    FunctionMessage(name="get_weather", content='{"temperature": "22C"}')
]
