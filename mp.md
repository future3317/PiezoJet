你项目里的 “mp 数据库” 指的是 Materials Project（MP） 的材料数据库。下面是调用方式和你的 API key 总结。

  1. 你的 API Key

  项目里目前硬编码在代码中：

  api_key = os.getenv("MP_API_KEY", "k1yfeLQjR4ETncm2YEFIetBspTB8pEwb")

  即：k1yfeLQjR4ETncm2YEFIetBspTB8pEwb

  建议优先通过环境变量配置，避免硬编码：

  export MP_API_KEY="k1yfeLQjR4ETncm2YEFIetBspTB8pEwb"

  2. 调用方式

  项目里用了两种调用方式。

  方式一：使用 mp-api 客户端（推荐，用于批量拉数据）

  from mp_api.client import MPRester

  api_key = os.getenv("MP_API_KEY", "k1yfeLQjR4ETncm2YEFIetBspTB8pEwb")

  with MPRester(api_key) as mpr:
      # 按材料 ID 批量查询
      data = mpr.materials.search(
          material_ids=["mp-123", "mp-456"],
          fields=["structure", "formula_pretty", "symmetry", "density", "volume"]
      )

  方式二：直接用 requests 调用 REST API

  文件位置：piezo_crystal_dataset/features/mp_api_client.py

  import requests

  api_key = os.getenv("MP_API_KEY", "k1yfeLQjR4ETncm2YEFIetBspTB8pEwb")
  base_url = "https://api.materialsproject.org"
  headers = {
      "X-API-KEY": api_key,
      "Content-Type": "application/json"
  }

  # 获取材料摘要
  response = requests.get(f"{base_url}/materials/summary/mp-123", headers=headers)
  data = response.json()

  常用 REST 端点：
  - 基本信息：/materials/summary/{material_id}
  - 弹性常数：/materials/elasticity/{material_id}
  - 介电常数：/materials/dielectric/{material_id}
  - 电子结构：/materials/electronic_structure/{material_id}
  - 压电系数：/materials/piezoelectricity/{material_id}
  - 搜索材料：/materials/summary?formula=SiO2&limit=10

  3. 核心调用经验

  你项目里已经总结过，关键几条：

  ┌────────────────┬───────────────────────────────────────────────────────────────────────────────┐
  │      项目      │                                     建议                                      │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ 批量大小       │ 每批 50 个 material_id 左右最优                                               │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ 查询方式       │ 用 mpr.materials.search(material_ids=...) 批量查，不要逐个 get_material_by_id │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ 数据分类型获取 │ 基本信息、电子结构、热力学数据分开批量获取                                    │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────┤
  │ 断点续传       │ 每批处理完保存临时文件，避免中断后重来