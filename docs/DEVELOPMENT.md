# MoneyPrinterTurbo 本地开发启动指南

本指南说明如何在本地快速搭建并启动完整的 MoneyPrinterTurbo 知识视频（Knowledge Video）智能体开发环境。

---

## 1. 快速入门：一键统一启动

在开发知识视频工作流（十阶段工作流、RAG 检索、分镜与供应商适配器）时，系统需要同时运行 3 个独立进程：
1. **FastAPI 后端**（提供 REST API 与任务生命周期管理，监听 `8080` 端口）
2. **StageWorker**（独立的后台工作进程，负责轮询数据库、认领 Job 并执行 Stage）
3. **Streamlit WebUI**（提供图形界面与知识库/工作台管理，监听 `8501` 端口）

项目提供了统一开发启动器（Unified Dev Launcher），自动进行数据库就绪检查与迁移、端口占用探测，并安全编排这 3 个独立进程。

### Windows 推荐启动命令

在项目根目录运行：

```cmd
.\dev.bat
```

> `dev.bat` 会自动检测并使用项目环境（`.venv`、`lib\python`、`uv` 或系统 Python）。

### macOS / Linux 推荐启动命令

```bash
sh dev.sh
# 或直接通过 Python 运行：
python3 dev.py
```

### 启动器参数选项

```bash
# 仅执行前置检查（数据库连接、Alembic 迁移、端口探测），不启动服务
python dev.py --check-only

# 无界面模式（仅启动 FastAPI 后端 + StageWorker）
python dev.py --no-webui

# 自定义端口
python dev.py --port 8080 --webui-port 8501
```

---

## 2. 进程状态与日志输出

统一启动器成功启动后，会在终端打印服务状态表：

```text
=================================================================
 MoneyPrinterTurbo Unified Development Environment
=================================================================
 Database: sqlite:///storage/dev.db
 Root:     D:\MoneyPrinterTurbo\MoneyPrinterTurbo
=================================================================

-----------------------------------------------------------------
 SERVICE        STATUS     PID      DETAILS
-----------------------------------------------------------------
 Database       READY      -        sqlite:///storage/dev.db
 Backend        RUNNING    14220    http://127.0.0.1:8080 (ping: OK)
 StageWorker    RUNNING    14224    independent process (polling loop)
 WebUI          RUNNING    14228    http://127.0.0.1:8501
-----------------------------------------------------------------
 Logs:
   Backend      -> .runtime-backend.log
   StageWorker  -> .runtime-worker.log
   WebUI        -> .runtime-webui.log
=================================================================
 Press Ctrl+C to stop all services cleanly.
```

- **健康检查**：启动器会自动轮询 `http://127.0.0.1:8080/ping`，确保后端完全就绪才确认成功。
- **独立日志**：各进程的标准输出与错误重定向至对应的 `.runtime-*.log`，已被 `.gitignore` 忽略，不污染 Git 工作区。
- **故障联动退出**：若任意子进程意外崩溃或退出，启动器会立即捕获异常并打印日志尾部摘要，同时联动关闭其余存活进程。

---

## 3. 如何安全停止

在运行统一启动器的终端中直接按下：

```text
Ctrl + C
```

启动器会拦截退出信号，依次向自身拉起的子进程发送优雅终止信号（`terminate`），等待退出，并在超时后安全收尾，**绝不遗留孤儿进程**，也**绝不会全局误杀**系统中的其他无关 Python 进程。

---

## 4. 本地开发模式对比

MoneyPrinterTurbo 明确支持两种本地开发运行模式：

| 模式 | 特点 | 适用场景 | 数据库 | 启动方式 |
| :--- | :--- | :--- | :--- | :--- |
| **Quick Local Dev**（本地轻量模式） | 开箱即用，无需 Docker 或外部依赖 | 日常代码调试、单元测试、本地页面预览 | 本地 SQLite (`storage/dev.db`) | `.\dev.bat` 或 `python dev.py` |
| **Integrated Dev / Compose**（容器集成模式） | 完整模拟生产多容器隔离环境 | 生产镜像构建、端到端集成验证、团队共享依赖 | PostgreSQL 16 Alpine (`db:5432`) | `docker compose up` |

---

## 5. Docker Compose 运行方式

若需要在本地以独立容器的形式运行包含 PostgreSQL 的全套环境：

```bash
# 启动所有服务（PostgreSQL + API + StageWorker + WebUI）
docker compose up -d

# 查看各容器状态
docker compose ps

# 查看各容器日志
docker compose logs -f api
docker compose logs -f worker

# 停止所有服务
docker compose down
```

---

## 6. 配置文件说明与敏感信息保护

- **`config.example.toml`**：配置模板文件，已纳入 Git 版本控制。其中**所有 API Key、Token 字段必须保持为空字符串或示例占位符**。
- **`config.toml`**：本地实际运行配置文件。应用在首次运行时若未发现该文件，会自动从 `config.example.toml` 复制生成。
- **安全防范**：`config.toml` 已被 `.gitignore` 严格忽略，**严禁将包含真实私钥/凭据的 `config.toml` 提交至 Git 仓库**。
