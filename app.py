# -*- coding: utf-8 -*-
"""
PUBG武器管理系统 - Web版
Flask Web应用主程序
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import sys
import os
import uuid
import secrets

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data import DataManager
from src.auth import AuthManager
from src.services import WeaponService
from src.models import Weapon
from src.audio import ModelLoader, AudioRecognizer

# 项目根目录
base_dir = Path(__file__).parent

# Web 资源目录（整理后统一放在 web/ 下）
web_dir = base_dir / "web"
template_dir = web_dir / "templates"
static_dir = web_dir / "static"

app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
app.secret_key = 'pubg_weapon_system_secret_key_2025'  # 用于session加密
upload_dir = base_dir / "uploads"
app.config['UPLOAD_FOLDER'] = str(upload_dir)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小16MB
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'mp3', 'wav', 'flac'}

# 确保上传目录存在
try:
    os.makedirs(upload_dir, exist_ok=True)
except OSError as e:
    # 在只读环境/无权限环境下，避免应用直接崩溃
    # 音频上传功能会不可用，但其它功能仍可运行
    print(f"警告: 无法创建上传目录 {app.config['UPLOAD_FOLDER']}: {e}")

# 初始化管理器
data_manager = DataManager(base_dir)
auth_manager = AuthManager(data_manager)
weapon_service = WeaponService(data_manager)

# 加载武器数据
success, msg = weapon_service.load_weapons_from_cache()
if not success:
    # 如果缓存加载失败,尝试从Excel加载
    excel_path = base_dir / "Arms.xlsx"
    if excel_path.exists():
        success, msg = weapon_service.load_weapons_from_excel(str(excel_path))
        print(msg)
    else:
        print("警告: 未找到武器数据")
else:
    print(msg)

# 加载音频识别模型
model_loader = ModelLoader(base_dir)
recognizer = AudioRecognizer(model_loader)
audio_model_loaded, audio_model_msg = model_loader.load_model()
if audio_model_loaded:
    print(audio_model_msg)
else:
    print(f"警告: {audio_model_msg}")

def _get_csrf_token() -> str:
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


def _is_json_request() -> bool:
    accept = request.headers.get('Accept', '')
    return 'application/json' in accept.lower() or request.is_json


@app.context_processor
def inject_csrf_token():
    return {'csrf_token': _get_csrf_token}


@app.before_request
def csrf_protect():
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return

    # Skip CSRF for static files
    if request.endpoint == 'static':
        return

    # For all other state-changing requests, require token.
    token = request.form.get('csrf_token') or request.headers.get('X-CSRFToken')
    if not token or token != session.get('_csrf_token'):
        if request.path.startswith('/upload_audio') or _is_json_request():
            return jsonify({'error': 'CSRF校验失败'}), 400
        flash('CSRF校验失败，请刷新页面后重试', 'error')
        return redirect(request.referrer or url_for('login'))


@app.route('/')
def index():
    """首页"""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        success, message = auth_manager.login(username, password)
        if success:
            session['username'] = username
            flash(message, 'success')
            return redirect(url_for('dashboard'))
        else:
            flash(message, 'error')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """注册页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('两次输入的密码不一致', 'error')
        elif len(username) < 6:
            flash('学号至少6位字符', 'error')
        elif len(password) < 6:
            flash('密码至少6个字符', 'error')
        else:
            success, message = auth_manager.register(username, password)
            if success:
                flash('注册成功! 请登录', 'success')
                return redirect(url_for('login'))
            else:
                flash(message, 'error')

    return render_template('register.html')


def _force_relogin(message: str = '用户数据未找到，请重新登录'):
    """
    内部使用：当 session 与本地玩家数据不一致时，清理登录态并引导重新登录。
    不作为路由暴露，避免产生“GET 改变状态”的入口。
    """
    session.pop('username', None)
    flash(message, 'error')
    return redirect(url_for('login'))


@app.route('/logout', methods=['POST'])
def logout():
    """登出（POST，CSRF保护）"""
    session.pop('username', None)
    flash('已退出登录', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    """主控制台"""
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    player = data_manager.get_player(username)

    # 获取武器库
    all_weapons = weapon_service.get_all_weapons()

    # 获取玩家背包
    backpack = player.weapons if player else []

    # 获取弹药库存
    ammo_inventory = player.ammo_inventory if player else {}

    return render_template('dashboard.html',
                         username=username,
                         all_weapons=all_weapons,
                         backpack=backpack,
                         ammo_inventory=ammo_inventory,
                         model_info=model_loader.get_model_info())


@app.route('/weapons')
def weapons():
    """武器库页面"""
    if 'username' not in session:
        return redirect(url_for('login'))

    search = request.args.get('search', '')
    sort_by = request.args.get('sort_by', 'name')

    all_weapons = weapon_service.get_all_weapons()

    # 搜索
    if search:
        all_weapons = [w for w in all_weapons if search.lower() in w.name.lower()]

    # 排序
    if sort_by == 'damage':
        all_weapons.sort(key=lambda x: x.damage, reverse=True)
    elif sort_by == 'fire_rate':
        all_weapons.sort(key=lambda x: x.fire_rate, reverse=True)
    elif sort_by == 'dps':
        all_weapons.sort(key=lambda x: x.calculate_dps(), reverse=True)

    return render_template('weapons.html', weapons=all_weapons, search=search, sort_by=sort_by)


@app.route('/backpack')
def backpack():
    """背包页面"""
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    player = data_manager.get_player(username)

    backpack_weapons = player.weapons if player else []

    return render_template('backpack.html', weapons=backpack_weapons)


@app.route('/add_weapon/<weapon_name>', methods=['POST'])
def add_weapon(weapon_name):
    """添加武器到背包"""
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    player = data_manager.get_player(username)

    if not player:
        return _force_relogin()

    weapon = weapon_service.get_weapon_by_name(weapon_name)
    if weapon:
        if player.add_weapon(weapon):
            data_manager.save_players()
            flash(f'成功添加 {weapon_name} 到背包!', 'success')
        else:
            flash('添加失败：背包已满或已拥有该武器', 'error')
    else:
        flash('武器不存在', 'error')

    # 优先回到来源页，避免每次都跳回 dashboard 打断浏览
    return redirect(request.referrer or url_for('weapons'))


@app.route('/remove_weapon/<weapon_name>', methods=['POST'])
def remove_weapon(weapon_name):
    """从背包移除武器"""
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    player = data_manager.get_player(username)

    if not player:
        return _force_relogin()

    if player.remove_weapon(weapon_name):
        data_manager.save_players()
        flash(f'成功移除 {weapon_name}', 'success')
    else:
        flash('武器不在背包中', 'error')

    return redirect(url_for('backpack'))


@app.route('/ammo')
def ammo():
    """弹药管理页面"""
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    player = data_manager.get_player(username)

    if not player:
        return _force_relogin()

    # 获取弹药库存
    ammo_inventory = player.ammo_inventory
    # 获取总弹药数（包括武器中的）
    total_ammo = player.get_total_ammo()

    return render_template('ammo.html',
                         ammo_inventory=ammo_inventory,
                         total_ammo=total_ammo,
                         weapons=player.weapons)


@app.route('/add_ammo', methods=['POST'])
def add_ammo():
    """添加弹药"""
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    player = data_manager.get_player(username)

    if not player:
        return _force_relogin()

    ammo_type = (request.form.get('ammo_type') or '').strip()
    if not ammo_type:
        flash('请选择弹药类型', 'error')
        return redirect(url_for('ammo'))

    try:
        amount = int(request.form.get('amount', 0))
    except (TypeError, ValueError):
        flash('数量必须是整数', 'error')
        return redirect(url_for('ammo'))

    if amount <= 0:
        flash('数量必须大于0', 'error')
        return redirect(url_for('ammo'))

    normalized_ammo_type = player.normalize_ammo_type(ammo_type)
    player.add_ammo(normalized_ammo_type, amount)
    data_manager.save_players()
    flash(f'成功添加 {amount} 发 {normalized_ammo_type} 弹药', 'success')

    return redirect(url_for('ammo'))


@app.route('/reload_weapon/<weapon_name>', methods=['POST'])
def reload_weapon_route(weapon_name):
    """为武器换弹"""
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    player = data_manager.get_player(username)

    if not player:
        return _force_relogin()

    used = player.reload_weapon(weapon_name)
    if used > 0:
        data_manager.save_players()
        flash(f'成功为 {weapon_name} 装填了 {used} 发弹药', 'success')
    else:
        flash('换弹失败，弹药不足或武器已满', 'error')

    return redirect(url_for('ammo'))


@app.route('/audio_recognition')
def audio_recognition():
    """音频识别页面"""
    if 'username' not in session:
        return redirect(url_for('login'))

    return render_template(
        'audio_recognition.html',
        model_info=model_loader.get_model_info(),
        model_status=audio_model_msg,
        model_loaded=audio_model_loaded,
    )


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """上传音频文件进行识别"""
    if 'username' not in session:
        return jsonify({'error': '未登录'}), 401

    if not audio_model_loaded:
        return jsonify({'error': audio_model_msg}), 503

    if 'audio_file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file:
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in app.config['ALLOWED_AUDIO_EXTENSIONS']:
            return jsonify({'error': f'不支持的文件类型: .{ext}'}), 400

        if not os.path.isdir(app.config['UPLOAD_FOLDER']):
            return jsonify({'error': '服务器未配置上传目录，无法上传文件'}), 500

        temp_name = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)

        try:
            file.save(filepath)
            # 识别音频
            weapon, confidence = recognizer.predict_from_file(filepath)
            return jsonify({
                'weapon': weapon,
                'confidence': f'{confidence:.2%}'
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # 删除临时文件（无论成功/失败）
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                pass

    return jsonify({'error': '文件上传失败'}), 400


if __name__ == '__main__':
    print("=" * 60)
    print("PUBG武器管理系统 - Web版")
    print("=" * 60)
    print(f"项目目录: {base_dir}")
    print(f"模型状态: {model_loader.get_model_info()}")
    print("=" * 60)
    print("服务器启动中...")
    print("访问地址: http://localhost:5000")
    print("局域网访问: http://你的IP:5000")
    print("=" * 60)

    # 开启调试模式,允许局域网访问
    app.run(host='0.0.0.0', port=5000, debug=True)
