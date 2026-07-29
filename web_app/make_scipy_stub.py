"""
在用户站点包里写入一个极小的 `scipy` 桩，使 `import lightgbm` 能够通过，
同时不占用 ~190MB 的真实 scipy 磁盘空间（PythonAnywhere 免费版 512MB 限制下必需）。

原理：
- lightgbm 在导入时（`basic.py` 顶层）只做 `import scipy.sparse` 并引用
  `spmatrix / csr_matrix / csc_matrix / hstack`，且这些名字仅用于 *稀疏输入* 路径。
- 本应用的预测只向 `Booster.predict` 喂 **稠密 numpy 数组**，永远不会真正调用 scipy，
  因此一个只含上述空壳名字的桩即可，无需真实 scipy。

仅部署到磁盘受限环境（如 PythonAnywhere 免费版）时运行本脚本；
本地完整环境请直接 `pip install scipy`，不要运行本脚本。
"""
import os
import site
import sys


STUB_SPARSE_INIT = '''\
# 最小化 scipy.sparse 桩（仅供 lightgbm 导入通过；预测只用稠密 numpy，不触发这些名字）
class spmatrix:
    pass


class csr_matrix(spmatrix):
    def __init__(self, *a, **k):
        raise RuntimeError("scipy stub: sparse matrices are not supported in this deployment")


class csc_matrix(spmatrix):
    def __init__(self, *a, **k):
        raise RuntimeError("scipy stub: sparse matrices are not supported in this deployment")


def hstack(*a, **k):
    raise RuntimeError("scipy stub: hstack is not supported in this deployment")


def issparse(x):
    return False
'''


def main():
    user_site = site.getusersitepackages()
    if not user_site or not os.path.isdir(user_site):
        print("未找到用户站点包目录：", user_site, file=sys.stderr)
        return 1
    scipy_dir = os.path.join(user_site, "scipy")
    sparse_dir = os.path.join(scipy_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    # scipy/__init__.py（空包即可）
    with open(os.path.join(scipy_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write("")
    # scipy/sparse/__init__.py
    with open(os.path.join(sparse_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(STUB_SPARSE_INIT)
    print("scipy 桩已写入：", scipy_dir)
    # 自检：能 import lightgbm
    try:
        import lightgbm  # noqa: F401
        from lightgbm import Booster  # noqa: F401
        print("自检通过：import lightgbm / Booster 成功（无需真实 scipy）")
    except Exception as e:  # pragma: no cover
        print("自检失败：", type(e).__name__, e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
