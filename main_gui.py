# -*- coding: utf-8 -*-
# NOTE: renamed from main_gui_complete.py to main_gui.py for cleaner naming.
"""
PUBG武器管理系统 - 完整版图形界面
包含武器管理 + 音频识别功能

重构版本: 使用模块化设计
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from pathlib import Path
import sys

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data import DataManager
from src.auth import AuthManager
from src.services import WeaponService
from src.models import Weapon
from src.audio import ModelLoader, AudioRecognizer


class AudioRecognitionWindow:
    """音频识别窗口(重构版)"""

    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("武器声音识别")
        self.window.geometry("700x600")

        self.base_dir = Path(__file__).parent

        # 使用模块化的加载器和识别器
        self.model_loader = ModelLoader(self.base_dir)
        self.recognizer = AudioRecognizer(self.model_loader)

        # 加载模型
        self.load_model()

        # 创建UI
        self.create_widgets()

        # 居中显示
        self.center_window()

    def center_window(self):
        """窗口居中"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def load_model(self):
        """加载训练好的模型"""
        success, msg = self.model_loader.load_model()
        if not success:
            messagebox.showwarning("警告", msg)
        else:
            print(msg)

    def create_widgets(self):
        """创建界面元素"""
        # 标题
        title_frame = ttk.Frame(self.window)
        title_frame.pack(pady=10, fill='x')

        ttk.Label(
            title_frame,
            text="🎯 武器声音识别系统",
            font=("Arial", 16, "bold")
        ).pack()

        ttk.Label(
            title_frame,
            text=self.model_loader.get_model_info(),
            font=("Arial", 10),
            foreground="gray"
        ).pack()

        # 分隔线
        ttk.Separator(self.window, orient='horizontal').pack(fill='x', pady=10)

        # 功能选择区域
        function_frame = ttk.LabelFrame(self.window, text="选择功能", padding=10)
        function_frame.pack(fill='x', padx=20, pady=10)

        ttk.Button(
            function_frame,
            text="📁 选择音频文件识别",
            command=self.select_audio_file,
            width=30
        ).pack(pady=5)

        ttk.Button(
            function_frame,
            text="📊 手动输入特征识别",
            command=self.manual_input,
            width=30
        ).pack(pady=5)

        ttk.Button(
            function_frame,
            text="🔍 批量测试模型",
            command=self.batch_test,
            width=30
        ).pack(pady=5)

        # 结果显示区域
        result_frame = ttk.LabelFrame(self.window, text="识别结果", padding=10)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # 文本框显示结果
        self.result_text = tk.Text(
            result_frame,
            height=15,
            width=60,
            font=("Consolas", 10),
            wrap=tk.WORD
        )
        self.result_text.pack(side='left', fill='both', expand=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.result_text.config(yscrollcommand=scrollbar.set)

        # 底部按钮
        bottom_frame = ttk.Frame(self.window)
        bottom_frame.pack(fill='x', padx=20, pady=10)

        ttk.Button(
            bottom_frame,
            text="清除结果",
            command=self.clear_results
        ).pack(side='left', padx=5)

        ttk.Button(
            bottom_frame,
            text="关闭",
            command=self.window.destroy
        ).pack(side='right', padx=5)

    def select_audio_file(self):
        """选择音频文件进行识别"""
        if not self.model_loader.is_loaded():
            messagebox.showerror("错误", "模型未加载，无法进行识别")
            return

        # 选择文件
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[
                ("音频文件", "*.wav *.mp3 *.flac"),
                ("所有文件", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            self.append_result(f"\n正在识别: {Path(file_path).name}\n")
            self.append_result("=" * 60 + "\n")

            # 使用识别器进行识别
            weapon, confidence = self.recognizer.predict_from_file(file_path)

            self.append_result(f"✅ 识别结果: {weapon}\n")
            self.append_result(f"📊 置信度: {confidence:.2%}\n")
            self.append_result("=" * 60 + "\n")

        except Exception as e:
            messagebox.showerror("错误", f"识别失败: {e}")
            self.append_result(f"❌ 错误: {e}\n")

    def manual_input(self):
        """手动输入特征"""
        if not self.model_loader.is_loaded():
            messagebox.showerror("错误", "模型未加载，无法进行识别")
            return

        # 创建输入对话框
        dialog = tk.Toplevel(self.window)
        dialog.title("手动输入特征")
        dialog.geometry("400x500")

        ttk.Label(dialog, text="请输入音频特征值:", font=("Arial", 12, "bold")).pack(pady=10)

        # 特征输入框
        entries = {}
        feature_names = [
            'duration', 'rms_mean', 'rms_std', 'zcr_mean', 'zcr_std',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'spectral_rolloff_mean'
        ]

        for name in feature_names:
            frame = ttk.Frame(dialog)
            frame.pack(fill='x', padx=20, pady=2)
            ttk.Label(frame, text=f"{name}:", width=25).pack(side='left')
            entry = ttk.Entry(frame, width=15)
            entry.pack(side='left', padx=5)
            entry.insert(0, "0.0")
            entries[name] = entry

        def predict_manual():
            try:
                features = {name: float(entry.get()) for name, entry in entries.items()}
                weapon, confidence = self.recognizer.predict(features)

                self.append_result(f"\n手动输入识别结果:\n")
                self.append_result("=" * 60 + "\n")
                self.append_result(f"✅ 识别结果: {weapon}\n")
                self.append_result(f"📊 置信度: {confidence:.2%}\n")
                self.append_result("=" * 60 + "\n")

                dialog.destroy()
            except Exception as e:
                messagebox.showerror("错误", f"识别失败: {e}")

        ttk.Button(dialog, text="识别", command=predict_manual).pack(pady=10)

    def batch_test(self):
        """批量测试"""
        if not self.model_loader.is_loaded():
            messagebox.showerror("错误", "模型未加载")
            return

        # 选择目录
        directory = filedialog.askdirectory(title="选择音频文件目录")
        if not directory:
            return

        directory = Path(directory)
        audio_files = list(directory.glob("*.wav")) + list(directory.glob("*.mp3"))

        if not audio_files:
            messagebox.showinfo("提示", "未找到音频文件")
            return

        self.append_result(f"\n批量测试开始 (共{len(audio_files)}个文件)\n")
        self.append_result("=" * 60 + "\n")

        total = 0

        for audio_file in audio_files[:10]:  # 限制前10个
            try:
                weapon, confidence = self.recognizer.predict_from_file(str(audio_file))
                self.append_result(f"{audio_file.name}: {weapon} ({confidence:.2%})\n")
                total += 1
            except Exception as e:
                self.append_result(f"{audio_file.name}: 失败 ({e})\n")

        self.append_result("=" * 60 + "\n")
        self.append_result(f"测试完成: {total}/{len(audio_files[:10])}\n")

    def append_result(self, text):
        """追加结果到文本框"""
        self.result_text.insert(tk.END, text)
        self.result_text.see(tk.END)
        self.window.update()

    def clear_results(self):
        """清除结果"""
        self.result_text.delete(1.0, tk.END)


class LoginWindow:
    """登录/注册窗口"""

    def __init__(self, parent, auth_manager, on_success):
        self.parent = parent
        self.auth_manager = auth_manager
        self.on_success = on_success
        self.window = tk.Toplevel(parent)
        self.window.title("PUBG武器管理系统 - 登录")
        self.window.geometry("400x300")
        self.window.resizable(False, False)

        self.center_window()
        self.create_widgets()

        self.window.transient(parent)
        self.window.grab_set()

    def center_window(self):
        """窗口居中"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        """创建界面元素"""
        title_label = tk.Label(
            self.window,
            text="PUBG武器管理系统",
            font=("Arial", 16, "bold"),
            fg="#2c3e50"
        )
        title_label.pack(pady=20)

        input_frame = ttk.Frame(self.window)
        input_frame.pack(pady=10, padx=20, fill='x')

        ttk.Label(input_frame, text="学号:").grid(row=0, column=0, sticky='w', pady=5)
        self.student_id_entry = ttk.Entry(input_frame, width=30)
        self.student_id_entry.grid(row=0, column=1, pady=5, padx=10)

        ttk.Label(input_frame, text="密码:").grid(row=1, column=0, sticky='w', pady=5)
        self.password_entry = ttk.Entry(input_frame, width=30, show="*")
        self.password_entry.grid(row=1, column=1, pady=5, padx=10)

        ttk.Label(input_frame, text="昵称:").grid(row=2, column=0, sticky='w', pady=5)
        self.nickname_entry = ttk.Entry(input_frame, width=30)
        self.nickname_entry.grid(row=2, column=1, pady=5, padx=10)

        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=20)

        login_btn = ttk.Button(button_frame, text="登录", command=self.login, width=12)
        login_btn.pack(side='left', padx=5)

        register_btn = ttk.Button(button_frame, text="注册", command=self.register, width=12)
        register_btn.pack(side='left', padx=5)

        self.window.bind('<Return>', lambda e: self.login())

    def login(self):
        """登录"""
        student_id = self.student_id_entry.get().strip()
        password = self.password_entry.get()

        if not student_id or not password:
            messagebox.showwarning("警告", "请输入学号和密码")
            return

        success, msg = self.auth_manager.login(student_id, password)
        if success:
            messagebox.showinfo("成功", f"欢迎回来，{msg}！")
            self.window.destroy()
            self.on_success()
        else:
            messagebox.showerror("错误", msg)

    def register(self):
        """注册"""
        student_id = self.student_id_entry.get().strip()
        password = self.password_entry.get()
        nickname = self.nickname_entry.get().strip()

        if not student_id or not password or not nickname:
            messagebox.showwarning("警告", "请填写所有字段")
            return

        success, msg = self.auth_manager.register(student_id, password, nickname)
        if success:
            messagebox.showinfo("成功", "注册成功！请登录")
            self.nickname_entry.delete(0, tk.END)
        else:
            messagebox.showerror("错误", msg)


class WeaponManagementApp:
    """武器管理主应用"""

    def __init__(self, root):
        self.root = root
        self.root.title("PUBG武器管理与声音识别系统")
        self.root.geometry("1200x700")

        self.base_dir = Path(__file__).parent
        self.data_manager = DataManager(self.base_dir)
        self.auth_manager = AuthManager(self.data_manager)
        self.weapon_service = WeaponService(self.data_manager)

        self._load_weapons()
        self.show_login()

    def _load_weapons(self):
        """加载武器数据"""
        success, msg = self.weapon_service.load_weapons_from_cache()
        if not success:
            excel_path = self.base_dir / "Arms.xlsx"
            if excel_path.exists():
                success, msg = self.weapon_service.load_weapons_from_excel(str(excel_path))

    def show_login(self):
        """显示登录窗口"""
        LoginWindow(self.root, self.auth_manager, self.on_login_success)

    def on_login_success(self):
        """登录成功后"""
        self.create_main_interface()

    def create_main_interface(self):
        """创建主界面"""
        for widget in self.root.winfo_children():
            widget.destroy()

        self.create_header()

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.create_weapon_list(main_frame)
        self.create_control_panel(main_frame)

    def create_header(self):
        """创建顶部信息栏"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=10, pady=5)

        player = self.auth_manager.current_player
        user_info = f"用户: {player.nickname} ({player.student_id})"
        ttk.Label(header_frame, text=user_info, font=("Arial", 12)).pack(side='left')

        # 音频识别按钮
        audio_btn = ttk.Button(
            header_frame,
            text="🎤 音频识别",
            command=self.open_audio_recognition
        )
        audio_btn.pack(side='right', padx=5)

        logout_btn = ttk.Button(header_frame, text="登出", command=self.logout)
        logout_btn.pack(side='right')

    def create_weapon_list(self, parent):
        """创建武器列表"""
        left_frame = ttk.Frame(parent)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        ttk.Label(left_frame, text="武器库", font=("Arial", 14, "bold")).pack(pady=5)

        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill='x', pady=5)

        ttk.Label(search_frame, text="搜索:").pack(side='left')
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side='left', fill='x', expand=True, padx=5)
        self.search_entry.bind('<KeyRelease>', lambda e: self.filter_weapons())

        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side='right', fill='y')

        columns = ('名称', '类型', '伤害', '弹药', 'DPS')
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show='tree headings',
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.tree.yview)

        self.tree.column('#0', width=0, stretch=False)
        self.tree.column('名称', width=150)
        self.tree.column('类型', width=120)
        self.tree.column('伤害', width=80)
        self.tree.column('弹药', width=100)
        self.tree.column('DPS', width=80)

        for col in columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by_column(c))

        self.tree.pack(fill='both', expand=True)
        self.refresh_weapon_list()

    def create_control_panel(self, parent):
        """创建控制面板"""
        right_frame = ttk.Frame(parent, width=300)
        right_frame.pack(side='right', fill='y')
        right_frame.pack_propagate(False)

        ttk.Label(right_frame, text="操作面板", font=("Arial", 14, "bold")).pack(pady=10)

        btn_width = 25

        ttk.Label(right_frame, text="武器管理", font=("Arial", 11, "bold")).pack(pady=(10, 5))
        ttk.Button(right_frame, text="查看武器详情", command=self.show_weapon_detail, width=btn_width).pack(pady=2)
        ttk.Button(right_frame, text="添加到背包", command=self.add_to_inventory, width=btn_width).pack(pady=2)
        ttk.Button(right_frame, text="从背包移除", command=self.remove_from_inventory, width=btn_width).pack(pady=2)

        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=15)

        ttk.Label(right_frame, text="我的背包", font=("Arial", 11, "bold")).pack(pady=(10, 5))
        ttk.Button(right_frame, text="查看背包", command=self.show_inventory, width=btn_width).pack(pady=2)
        ttk.Button(right_frame, text="武器排序", command=self.sort_inventory, width=btn_width).pack(pady=2)

        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=15)

        ttk.Label(right_frame, text="弹药管理", font=("Arial", 11, "bold")).pack(pady=(10, 5))
        ttk.Button(right_frame, text="添加弹药", command=self.add_ammo, width=btn_width).pack(pady=2)
        ttk.Button(right_frame, text="查看弹药", command=self.show_ammo, width=btn_width).pack(pady=2)

        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=15)

        ttk.Label(right_frame, text="系统", font=("Arial", 11, "bold")).pack(pady=(10, 5))
        ttk.Button(right_frame, text="刷新武器库", command=self.refresh_weapons, width=btn_width).pack(pady=2)

    def open_audio_recognition(self):
        """打开音频识别窗口"""
        AudioRecognitionWindow(self.root)

    def refresh_weapon_list(self):
        """刷新武器列表"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        weapons = self.weapon_service.get_all_weapons()
        for weapon in weapons:
            self.tree.insert('', 'end', values=(
                weapon.name,
                weapon.weapon_type,
                weapon.damage,
                weapon.ammo_type,
                f"{weapon.get_dps():.1f}"
            ))

    def filter_weapons(self):
        """过滤武器"""
        search_text = self.search_entry.get().lower()

        for item in self.tree.get_children():
            self.tree.delete(item)

        weapons = self.weapon_service.get_all_weapons()
        for weapon in weapons:
            if (search_text in weapon.name.lower() or
                search_text in weapon.weapon_type.lower() or
                search_text in weapon.ammo_type.lower()):
                self.tree.insert('', 'end', values=(
                    weapon.name,
                    weapon.weapon_type,
                    weapon.damage,
                    weapon.ammo_type,
                    f"{weapon.get_dps():.1f}"
                ))

    def sort_by_column(self, col):
        """按列排序"""
        data = [(self.tree.set(item, col), item) for item in self.tree.get_children('')]
        data.sort(reverse=False)
        for index, (val, item) in enumerate(data):
            self.tree.move(item, '', index)

    def show_weapon_detail(self):
        """显示武器详情"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择一个武器")
            return

        weapon_name = self.tree.item(selected[0])['values'][0]
        weapon = self.weapon_service.get_weapon_by_name(weapon_name)

        if weapon:
            detail = f"""
武器名称: {weapon.name}
类型: {weapon.weapon_type}
伤害: {weapon.damage}
射速: {weapon.fire_rate} RPM
弹匣容量: {weapon.magazine_size}
弹药类型: {weapon.ammo_type}
有效射程: {weapon.effective_range}m
子弹速度: {weapon.bullet_speed}m/s
换弹时间: {weapon.reload_time}s
DPS: {weapon.get_dps():.2f}
            """
            messagebox.showinfo(f"武器详情 - {weapon.name}", detail.strip())

    def add_to_inventory(self):
        """添加到背包"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择一个武器")
            return

        weapon_name = self.tree.item(selected[0])['values'][0]
        weapon = self.weapon_service.get_weapon_by_name(weapon_name)

        if weapon:
            player = self.auth_manager.current_player
            if player.add_weapon(weapon):
                self.auth_manager.save_current_player()
                messagebox.showinfo("成功", f"已将 {weapon.name} 添加到背包")
            else:
                messagebox.showwarning("提示", "添加失败：背包已满或已拥有该武器")

    def remove_from_inventory(self):
        """从背包移除"""
        player = self.auth_manager.current_player
        if not player.weapons:
            messagebox.showinfo("提示", "背包为空")
            return

        weapon_names = [w.name for w in player.weapons]
        choice = simpledialog.askstring(
            "选择武器",
            f"请输入要移除的武器名称:\n{', '.join(weapon_names)}"
        )

        if choice:
            for weapon in player.weapons:
                if weapon.name == choice:
                    player.remove_weapon(weapon.name)
                    self.auth_manager.save_current_player()
                    messagebox.showinfo("成功", f"已移除 {weapon.name}")
                    return

            messagebox.showwarning("警告", "未找到该武器")

    def show_inventory(self):
        """显示背包"""
        player = self.auth_manager.current_player

        if not player.weapons:
            messagebox.showinfo("我的背包", "背包为空")
            return

        inventory_text = f"背包武器 ({len(player.weapons)}件):\n\n"
        for i, weapon in enumerate(player.weapons, 1):
            inventory_text += f"{i}. {weapon.name} ({weapon.weapon_type}) - DPS: {weapon.get_dps():.1f}\n"

        messagebox.showinfo("我的背包", inventory_text)

    def sort_inventory(self):
        """排序背包"""
        player = self.auth_manager.current_player

        if not player.weapons:
            messagebox.showinfo("提示", "背包为空")
            return

        player.weapons = player.sort_weapons('dps')
        self.auth_manager.save_current_player()

        messagebox.showinfo("成功", "背包已按DPS排序")
        self.show_inventory()

    def add_ammo(self):
        """添加弹药"""
        ammo_type = simpledialog.askstring("添加弹药", "请输入弹药类型 (如: 5.56mm, 7.62mm):")
        if not ammo_type:
            return

        amount = simpledialog.askinteger("添加弹药", f"请输入{ammo_type}的数量:", minvalue=1)
        if not amount:
            return

        player = self.auth_manager.current_player
        player.add_ammo(ammo_type, amount)
        self.auth_manager.save_current_player()

        messagebox.showinfo("成功", f"已添加 {amount} 发 {ammo_type}")

    def show_ammo(self):
        """显示弹药"""
        player = self.auth_manager.current_player

        if not player.ammo_inventory:
            messagebox.showinfo("弹药库", "弹药库为空")
            return

        ammo_text = "弹药库存:\n\n"
        for ammo_type, amount in player.ammo_inventory.items():
            ammo_text += f"{ammo_type}: {amount} 发\n"

        messagebox.showinfo("弹药库", ammo_text)

    def refresh_weapons(self):
        """刷新武器库"""
        excel_path = self.base_dir / "Arms.xlsx"
        if excel_path.exists():
            success, msg = self.weapon_service.load_weapons_from_excel(str(excel_path))
            if success:
                self.refresh_weapon_list()
                messagebox.showinfo("成功", msg)
            else:
                messagebox.showerror("错误", msg)
        else:
            messagebox.showwarning("警告", "未找到 Arms.xlsx 文件")

    def logout(self):
        """登出"""
        self.auth_manager.logout()
        for widget in self.root.winfo_children():
            widget.destroy()
        self.show_login()


def main():
    """主函数"""
    root = tk.Tk()
    app = WeaponManagementApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
