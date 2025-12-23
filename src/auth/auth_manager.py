# -*- coding: utf-8 -*-
"""
认证管理模块 - 处理玩家注册和登录
"""

from typing import Optional, Dict
from ..models import Player
from ..data import DataManager


class AuthManager:
    """认证管理类"""

    def __init__(self, data_manager: DataManager):
        """
        初始化认证管理器

        Args:
            data_manager: 数据管理器实例
        """
        self.data_manager = data_manager
        # 与 DataManager 共享同一份玩家字典，避免“注册/登录成功但业务层找不到玩家”的状态不一致问题
        self.players: Dict[str, Player] = data_manager.players
        self.current_player: Optional[Player] = None

    def register(self, student_id: str, password: str, nickname: str = "") -> tuple[bool, str]:
        """
        注册新玩家

        Args:
            student_id: 学号
            password: 密码
            nickname: 昵称

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        # 验证学号格式（简单验证）
        if not student_id or len(student_id) < 6:
            return False, "学号格式不正确，至少需要6位数字"

        # 验证密码强度
        if not password or len(password) < 6:
            return False, "密码至少需要6个字符"

        # 检查是否已存在
        if student_id in self.players:
            return False, "该学号已注册"

        # 创建新玩家
        try:
            player = Player.create_new(student_id, password, nickname)
            self.players[student_id] = player
            # 保存到文件
            if self.data_manager.save_players(self.players):
                return True, f"注册成功！欢迎 {player.nickname}"
            else:
                return False, "注册失败：保存数据时出错"
        except Exception as e:
            return False, f"注册失败：{str(e)}"

    def login(self, student_id: str, password: str) -> tuple[bool, str]:
        """
        玩家登录

        Args:
            student_id: 学号
            password: 密码

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        # 检查学号是否存在
        if student_id not in self.players:
            return False, "学号不存在，请先注册"

        player = self.players[student_id]

        # 验证密码
        if not player.verify_password(password):
            return False, "密码错误"

        # 登录成功
        self.current_player = player
        return True, f"登录成功！欢迎回来，{player.nickname}"

    def logout(self) -> bool:
        """
        玩家登出

        Returns:
            bool: 是否成功登出
        """
        if self.current_player is None:
            return False

        # 保存当前玩家数据
        self.data_manager.save_players(self.players)
        self.current_player = None
        return True

    def is_logged_in(self) -> bool:
        """
        检查是否已登录

        Returns:
            bool: 是否已登录
        """
        return self.current_player is not None

    def get_current_player(self) -> Optional[Player]:
        """
        获取当前登录的玩家

        Returns:
            Optional[Player]: 当前玩家，未登录则返回None
        """
        return self.current_player

    def change_password(self, old_password: str, new_password: str) -> tuple[bool, str]:
        """
        修改当前玩家密码

        Args:
            old_password: 旧密码
            new_password: 新密码

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        if not self.is_logged_in():
            return False, "请先登录"

        # 验证新密码强度
        if not new_password or len(new_password) < 6:
            return False, "新密码至少需要6个字符"

        # 修改密码
        if self.current_player.change_password(old_password, new_password):
            # 保存到文件
            if self.data_manager.save_players(self.players):
                return True, "密码修改成功"
            else:
                return False, "密码修改失败：保存数据时出错"
        else:
            return False, "旧密码错误"

    def get_all_players_info(self) -> list[dict]:
        """
        获取所有玩家信息（不包含密码）

        Returns:
            list[dict]: 玩家信息列表
        """
        info_list = []
        for player in self.players.values():
            info_list.append({
                'student_id': player.student_id,
                'nickname': player.nickname,
                'weapon_count': len(player.weapons),
                'ammo_types': len(player.ammo_inventory)
            })
        return info_list

    def save_current_player(self) -> bool:
        """
        保存当前玩家数据

        Returns:
            bool: 是否保存成功
        """
        if not self.is_logged_in():
            return False
        return self.data_manager.save_players(self.players)
