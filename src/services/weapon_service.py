# -*- coding: utf-8 -*-
"""
武器管理服务模块 - 处理武器相关的业务逻辑
"""

from typing import List, Optional, Dict
from ..models import Weapon, Player
from ..data import DataManager


class WeaponService:
    """武器管理服务类"""

    def __init__(self, data_manager: DataManager):
        """
        初始化武器服务

        Args:
            data_manager: 数据管理器实例
        """
        self.data_manager = data_manager
        self.available_weapons: List[Weapon] = []

    def load_weapons_from_excel(self, excel_path: str) -> tuple[bool, str]:
        """
        从Excel文件加载武器库

        Args:
            excel_path: Excel文件路径

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        try:
            weapons = self.data_manager.load_weapons_from_excel(excel_path)
            if weapons:
                self.available_weapons = weapons
                # 保存到JSON
                self.data_manager.save_weapons(weapons)
                return True, f"成功加载 {len(weapons)} 件武器"
            else:
                return False, "未能从Excel文件中读取到武器数据"
        except Exception as e:
            return False, f"加载失败: {str(e)}"

    def load_weapons_from_cache(self) -> tuple[bool, str]:
        """
        从缓存的JSON文件加载武器库

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        try:
            weapons = self.data_manager.load_weapons()
            if weapons:
                self.available_weapons = weapons
                return True, f"从缓存加载 {len(weapons)} 件武器"
            else:
                return False, "缓存中没有武器数据"
        except Exception as e:
            return False, f"加载失败: {str(e)}"

    def get_all_weapons(self) -> List[Weapon]:
        """
        获取所有可用武器

        Returns:
            List[Weapon]: 武器列表
        """
        return self.available_weapons.copy()

    def get_weapon_by_name(self, name: str) -> Optional[Weapon]:
        """
        根据名称获取武器

        Args:
            name: 武器名称

        Returns:
            Optional[Weapon]: 找到的武器，未找到则返回None
        """
        for weapon in self.available_weapons:
            if weapon.name == name:
                # 返回副本，避免修改原数据
                return Weapon.from_dict(weapon.to_dict())
        return None

    def search_weapons(self, keyword: str) -> List[Weapon]:
        """
        搜索武器（按名称或类型）

        Args:
            keyword: 搜索关键词

        Returns:
            List[Weapon]: 匹配的武器列表
        """
        keyword = keyword.lower()
        results = []
        for weapon in self.available_weapons:
            if (keyword in weapon.name.lower() or
                keyword in weapon.weapon_type.lower()):
                results.append(weapon)
        return results

    def filter_weapons_by_type(self, weapon_type: str) -> List[Weapon]:
        """
        按类型筛选武器

        Args:
            weapon_type: 武器类型

        Returns:
            List[Weapon]: 符合条件的武器列表
        """
        return [w for w in self.available_weapons if w.weapon_type == weapon_type]

    def sort_weapons(self, weapons: List[Weapon], by: str = 'damage',
                     reverse: bool = True) -> List[Weapon]:
        """
        对武器列表进行排序

        Args:
            weapons: 武器列表
            by: 排序依据（damage, fire_rate, magazine_size, effective_range, dps）
            reverse: 是否降序

        Returns:
            List[Weapon]: 排序后的武器列表
        """
        if by == 'damage':
            return sorted(weapons, key=lambda w: w.damage, reverse=reverse)
        elif by == 'fire_rate':
            return sorted(weapons, key=lambda w: w.fire_rate, reverse=reverse)
        elif by == 'magazine_size':
            return sorted(weapons, key=lambda w: w.magazine_size, reverse=reverse)
        elif by == 'effective_range':
            return sorted(weapons, key=lambda w: w.effective_range, reverse=reverse)
        elif by == 'dps':
            return sorted(weapons, key=lambda w: w.get_dps(), reverse=reverse)
        else:
            return weapons

    def add_weapon_to_player(self, player: Player, weapon_name: str) -> tuple[bool, str]:
        """
        为玩家添加武器

        Args:
            player: 玩家对象
            weapon_name: 武器名称

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        # 检查武器是否存在
        weapon = self.get_weapon_by_name(weapon_name)
        if weapon is None:
            return False, f"武器 '{weapon_name}' 不存在"

        # 先区分失败原因，便于给出准确提示
        if len(player.weapons) >= getattr(player, "MAX_WEAPONS", 10):
            return False, "背包已满，无法继续添加武器"
        if player.get_weapon(weapon.name) is not None:
            return False, f"玩家已拥有武器: {weapon.name}"

        # 添加到玩家武器库
        if player.add_weapon(weapon):
            return True, f"成功添加武器: {weapon.name}"
        else:
            return False, "添加失败"

    def remove_weapon_from_player(self, player: Player, weapon_name: str) -> tuple[bool, str]:
        """
        从玩家武器库移除武器

        Args:
            player: 玩家对象
            weapon_name: 武器名称

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        removed = player.remove_weapon(weapon_name)
        if removed:
            return True, f"成功移除武器: {weapon_name}"
        else:
            return False, f"玩家没有武器: {weapon_name}"

    def get_player_weapons_stats(self, player: Player) -> Dict:
        """
        获取玩家武器统计信息

        Args:
            player: 玩家对象

        Returns:
            Dict: 统计信息
        """
        weapons = player.get_all_weapons()
        if not weapons:
            return {
                'total': 0,
                'types': {},
                'total_ammo': player.get_total_ammo(),
                'highest_damage': None,
                'highest_dps': None
            }

        # 统计各类型数量
        type_count = {}
        for w in weapons:
            type_count[w.weapon_type] = type_count.get(w.weapon_type, 0) + 1

        # 找出最高伤害和最高DPS的武器
        highest_damage = max(weapons, key=lambda w: w.damage)
        highest_dps = max(weapons, key=lambda w: w.get_dps())

        return {
            'total': len(weapons),
            'types': type_count,
            'total_ammo': player.get_total_ammo(),
            'highest_damage': {
                'name': highest_damage.name,
                'damage': highest_damage.damage
            },
            'highest_dps': {
                'name': highest_dps.name,
                'dps': round(highest_dps.get_dps(), 2)
            }
        }

    def get_weapon_types(self) -> List[str]:
        """
        获取所有武器类型

        Returns:
            List[str]: 武器类型列表
        """
        types = set()
        for weapon in self.available_weapons:
            types.add(weapon.weapon_type)
        return sorted(list(types))

    def get_ammo_types(self) -> List[str]:
        """
        获取所有弹药类型

        Returns:
            List[str]: 弹药类型列表
        """
        types = set()
        for weapon in self.available_weapons:
            types.add(weapon.ammo_type)
        return sorted(list(types))
