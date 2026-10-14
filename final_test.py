# -*- coding:utf-8 -*-
"""
最终验证测试：offscreen 真实链路——预测→概率采集→存档
沙盒化 prediction_records，绝不触碰真实 prediction_history.json。
(旧版用 mock_init 伪造窗口，属性名单与现行 generate_prediction 已脱节而长期失效；
 现改为真实窗口+拦截对话框，与现网 GUI 行为一致。)
"""
import os
import sys

# 添加项目路径
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# 沙盒隔离: 任何 prediction_records 读写都落临时文件
import tempfile as _tf
import prediction_records as _pr
_pr.HISTORY_FILE = os.path.join(_tf.mkdtemp(prefix='final_sandbox_'), 'ph.json')

# 无头模式
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import torch  # noqa: F401  先 torch 后 Qt，规避 DLL 冲突(已知约定)


def test_full_flow():
    """测试完整的预测->存档流程(真实 offscreen 窗口 + mock 模型输出)"""
    print("开始完整流程测试...")

    from PyQt5.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv)

    # 拦截所有模态对话框
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.critical = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

    # mock 模型预测输出(注入概率，验证采集链路)
    from ml_models import LotteryMLModels
    def mock_predict(self, recent_data, variation=0):
        if self.lottery_type == 'ssq':
            red_pred, blue_pred = [1, 2, 3, 4, 5, 6], [1]
            self.last_red_proba = {str(i): 0.1 for i in range(33)}
            self.last_blue_proba = {str(i): 0.1 for i in range(16)}
        else:
            red_pred, blue_pred = [1, 2, 3, 4, 5], [1, 2]
            self.last_red_proba = {str(i): 0.1 for i in range(35)}
            self.last_blue_proba = {str(i): 0.1 for i in range(12)}
        return red_pred, blue_pred
    LotteryMLModels.predict = mock_predict
    # 跳过模型加载(无需真实模型文件)
    LotteryMLModels.load_models = lambda self: True

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        from lottery_predictor_app_new import LotteryPredictorApp
        win = LotteryPredictorApp()
    app.processEvents()
    print("  应用实例创建成功(真实 offscreen 窗口)")

    # 切到 SSQ + 随机森林, 1注 简单
    win.lottery_combo.setCurrentIndex(0)  # 以实际第一项为准, 下面二次确认文本
    for i in range(win.lottery_combo.count()):
        if 'SSQ' in win.lottery_combo.itemText(i) or '双色球' in win.lottery_combo.itemText(i):
            win.lottery_combo.setCurrentIndex(i)
            break
    for i in range(win.model_combo.count()):
        if 'random_forest' in win.model_combo.itemText(i):
            win.model_combo.setCurrentIndex(i)
            break
    print(f"  彩种: {win.lottery_combo.currentText()} / 模型: {win.model_combo.currentText()}")

    # 清空沙盒记录
    from prediction_records import load_records, save_records
    save_records([])
    print("  已清空沙盒预测记录")

    # 走真实 generate_prediction (内部有 try/except 兜底)
    try:
        win.generate_prediction()
        print("  预测生成方法执行完成")
    except Exception as e:
        print(f"  预测生成过程中出现异常: {e}")

    app.processEvents()

    records = load_records()
    print(f"  存档的记录数量: {len(records)}")
    if len(records) > 0:
        r = records[0]
        print(f"  第一条记录: {r.get('lottery_type')} {r.get('model_type')} "
              f"红{r.get('red_numbers')} 蓝{r.get('blue_numbers')}")
        if r.get('red_probabilities') is not None or r.get('blue_probabilities') is not None:
            print("  ✅ 成功：预测记录中包含概率信息")
            return True
        print("  ❌ 失败：预测记录中缺少概率信息")
        return False
    print("  ❌ 失败：没有存档任何预测记录")
    return False


if __name__ == "__main__":
    print("=" * 50)
    print("最终验证测试")
    print("=" * 50)
    try:
        ok = test_full_flow()
        print("\n" + "=" * 50)
        if ok:
            print("✅ 最终验证测试通过。")
            sys.exit(0)
        else:
            print("❌ 最终验证测试失败。")
            sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 测试运行出错: {e}")
        sys.exit(1)
