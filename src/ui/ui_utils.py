# -*- coding: utf-8 -*-
"""
UI工具模块 - 提供命令行界面的辅助函数
"""

import os
import sys


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str):
    """
    打印标题头

    Args:
        title: 标题文本
    """
    width = 60
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")


def print_separator(char: str = "-", length: int = 60):
    """
    打印分隔线

    Args:
        char: 分隔符字符
        length: 长度
    """
    print(char * length)


def print_menu(title: str, options: list[str]):
    """
    打印菜单

    Args:
        title: 菜单标题
        options: 选项列表
    """
    print_header(title)
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print_separator()


def get_input(prompt: str, input_type=str, default=None, allow_empty: bool = False):
    """
    获取用户输入

    Args:
        prompt: 提示信息
        input_type: 输入类型（str, int, float等）
        default: 默认值
        allow_empty: 是否允许空输入

    Returns:
        转换后的输入值
    """
    while True:
        try:
            user_input = input(f"{prompt}: ").strip()

            # 空输入处理
            if not user_input:
                if allow_empty:
                    return default if default is not None else ""
                elif default is not None:
                    return default
                else:
                    print("输入不能为空，请重新输入")
                    continue

            # 类型转换
            if input_type == str:
                return user_input
            elif input_type == int:
                return int(user_input)
            elif input_type == float:
                return float(user_input)
            else:
                return input_type(user_input)

        except ValueError:
            print(f"输入格式错误，请输入{input_type.__name__}类型的值")
        except KeyboardInterrupt:
            print("\n操作已取消")
            return None


def get_password(prompt: str = "请输入密码") -> str:
    """
    获取密码输入（遮蔽显示）

    Args:
        prompt: 提示信息

    Returns:
        str: 输入的密码
    """
    try:
        import stdiomask
        password = stdiomask.getpass(prompt=f"{prompt}: ")
        return password
    except ImportError:
        # 如果没有stdiomask，使用getpass模块
        import getpass
        password = getpass.getpass(f"{prompt}: ")
        return password
    except KeyboardInterrupt:
        print("\n操作已取消")
        return ""


def confirm(prompt: str, default: bool = False) -> bool:
    """
    确认对话框

    Args:
        prompt: 提示信息
        default: 默认值

    Returns:
        bool: 用户选择
    """
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{default_str}]: ").strip().lower()

    if not response:
        return default

    return response in ('y', 'yes', '是')


def print_table(headers: list[str], rows: list[list], widths: list[int] = None):
    """
    打印表格

    Args:
        headers: 表头列表
        rows: 数据行列表
        widths: 各列宽度列表
    """
    if not rows:
        print("  (无数据)")
        return

    # 自动计算列宽
    if widths is None:
        widths = []
        for i, header in enumerate(headers):
            max_width = len(str(header))
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            widths.append(max_width + 2)

    # 打印表头
    header_row = "".join(f"{str(h):<{w}}" for h, w in zip(headers, widths))
    print(f"  {header_row}")
    print_separator("-", sum(widths) + 2)

    # 打印数据行
    for row in rows:
        data_row = "".join(f"{str(cell):<{w}}" for cell, w in zip(row, widths))
        print(f"  {data_row}")


def print_success(message: str):
    """打印成功消息"""
    print(f"✓ {message}")


def print_error(message: str):
    """打印错误消息"""
    print(f"✗ {message}")


def print_info(message: str):
    """打印信息消息"""
    print(f"ℹ {message}")


def pause(message: str = "按Enter继续..."):
    """
    暂停等待用户输入

    Args:
        message: 提示信息
    """
    input(f"\n{message}")
