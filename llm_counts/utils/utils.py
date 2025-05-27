import pprint
from .constants import *

class Formatter(object):
    @classmethod
    def format_value(cls, value, category):
        """根据类别统一格式化 value."""
        if category == "params" or category == "flops":
            return num_to_string(value)
        elif category == "latency":
            return latency_to_string(value)
        elif category == "memory":
            return f"{num_to_string(value)}B"
        return value  # 如果没有匹配，返回原值

    @classmethod
    def print_format_summary_dict(
        cls, summary_dict: dict, depth: int, category="params"
    ) -> str:
        """打印时对 params / flops / latency / memory 等进行统一转换显示。"""

        def recursive_format(dictionary, current_depth):
            if current_depth <= 0:  # 深度限制
                return dictionary

            formatted_dict = {}
            for key, value in dictionary.items():
                if isinstance(value, dict):  # 如果是子字典，递归格式化
                    formatted_dict[key] = recursive_format(value, current_depth - 1)
                elif "max_bs" in key or "max_batch_total" in key:  # 格式化具体值
                    formatted_dict[key] = value
                else:
                    formatted_dict[key] = cls.format_value(value, category)
            return formatted_dict

        # 开始递归格式化
        formatted_summary = recursive_format(summary_dict, depth)

        # 打印格式化后的字典
        pprint.pprint(formatted_summary, indent=4, sort_dicts=False)
        return formatted_summary  # 返回格式化后的字典


def print_list(list):
    """print one-dimensional list

    :param list: List[int]
    :return: None
    """
    for i, x in enumerate(list):
        print(x, end="\n")


def get_dict_depth(d, depth=0):
    if not isinstance(d, dict):
        return depth
    if not d:
        return depth

    return max(get_dict_depth(v, depth + 1) for v in d.values())


def latency_to_string(latency_in_s, precision=2, return_type="string"):
    if latency_in_s is None:
        return "None" if return_type == "string" else None

    day = 24 * 60 * 60
    hour = 60 * 60
    minute = 60
    ms = 1 / 1000
    us = 1 / 1000000

    if latency_in_s // day > 0:
        value = round(latency_in_s / day, precision)
        unit = "days"
    elif latency_in_s // hour > 0:
        value = round(latency_in_s / hour, precision)
        unit = "hours"
    elif latency_in_s // minute > 0:
        value = round(latency_in_s / minute, precision)
        unit = "minutes"
    elif latency_in_s > 1:
        value = round(latency_in_s, precision)
        unit = "s"
    elif latency_in_s > ms:
        value = round(latency_in_s / ms, precision)
        unit = "ms"
    else:
        value = round(latency_in_s / us, precision)
        unit = "us"

    if return_type == "string":
        return f"{value} {unit}"
    elif return_type == "float":
        return value
    else:
        return (value, unit)


def num_to_string(num, precision=2, return_type="string"):
    if num is None:
        return "None" if return_type == "string" else None

    if num // 10**12 > 0:
        value = round(num / 10.0**12, precision)
        unit = "T"
    elif num // 10**9 > 0:
        value = round(num / 10.0**9, precision)
        unit = "G"
    elif num // 10**6 > 0:
        value = round(num / 10.0**6, precision)
        unit = "M"
    elif num // 10**3 > 0:
        value = round(num / 10.0**3, precision)
        unit = "K"
    else:
        value = num
        unit = ""

    if return_type == "string":
        return f"{value} {unit}".strip()
    elif return_type == "float":
        return value
    else:
        return (value, unit)


def get_readable_summary_dict(summary_dict: dict, title="Summary") -> str:
    log_str = f"\n{title.center(PRINT_LINE_WIDTH, '-')}\n"
    for key, value in summary_dict.items():
        if "num_tokens" in key or "num_params" in key or "flops" in key:
            log_str += f"{key}: {num_to_string(value)}\n"
        elif "gpu_hours" == key:
            log_str += f"{key}: {int(value)}\n"
        elif "memory" in key and "efficiency" not in key:
            log_str += f"{key}: {num_to_string(value)}B\n"
        elif "latency" in key:
            log_str += f"{key}: {latency_to_string(value)}\n"
        else:
            log_str += f"{key}: {value}\n"
    log_str += f"{'-' * PRINT_LINE_WIDTH}\n"
    return log_str


def within_range(val, target, tolerance):
    return abs(val - target) / target < tolerance


def average(lst):
    if not lst:
        return None
    return sum(lst) / len(lst)


def max_value(lst):
    if not lst:
        return None
    return max(lst)
