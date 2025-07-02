import inspect
from pprint import pprint

from . import (humaneval, humanevalpack, mbpp, codesearchnet, humanevalplus, mbppplus)

TASK_REGISTRY = {
    **humaneval.create_all_tasks(),
    **humanevalplus.create_all_tasks(),
    **humanevalpack.create_all_tasks(),
    "mbpp": mbpp.MBPP,
    "mbppplus": mbppplus.MBPPPlus,
    **codesearchnet.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            if not args.prompt:
                kwargs["prompt"] = "instruct"
            else:
                kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
