import re


def get_response(input_prompt: str, agent, sys_msg: str = "You are a chemist expert"):
    messages = []
    messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": input_prompt})
    return agent.get_response(messages), messages


def get_cost(text: str):
    return re.findall(r'(\d+)', text)[-1]


def get_history_str(memory):
    history_str = ""
    for item in memory.history:
        history_str += f"- Run {item['action_id']} on {', '.join(item['action_input'])}. The output is {item['action_output']}\n"
    return history_str
