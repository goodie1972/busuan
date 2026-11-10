with open('D:/bobo/python/busuan/lottery_predictor_app_new.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_func = '''    def _do_auto_verify(self):
        """后台执行自动核验，不阻塞 UI"""
        if self.auto_predict_thread and self.auto_predict_thread.isRunning():
            self.update_log("自动核验延迟：智能预测进行中")
            return
        try:
            from prediction_records import verify_records
            # 同步核验所有记录（数据量小时快速完成）
            self.update_log("自动核验：检查是否有新开奖数据可核验...")
            report = verify_records(
                log_callback=lambda msg: self.log_emitter.new_log.emit(msg))
            self.refresh_prediction_records()
            self._refresh_model_weights()
            verified = report.get('verified', 0)
            pending = report.get('pending', 0)
            if verified > 0:
                self.update_log(f"自动核验完成：本次核验 {verified} 条，待核验 {pending} 条")
            else:
                self.update_log(f"自动核验：无新增可核验记录（待核验 {pending} 条）")
        except Exception as e:
            self.update_log(f"自动核验失败: {e}")'''

new_func = '''    def _do_auto_verify(self):
        """后台执行自动核验，不阻塞 UI"""
        if self.auto_predict_thread and self.auto_predict_thread.isRunning():
            self.update_log("自动核验延迟：智能预测进行中")
            return
        try:
            from prediction_records import verify_records
            # 同步核验所有记录（数据量小时快速完成）
            self.update_log("自动核验：检查是否有新开奖数据可核验...")
            report = verify_records(
                log_callback=lambda msg: self.log_emitter.new_log.emit(msg))

            # 同时核验集成模型记录
            try:
                from ensemble_engine import EnsembleEngine
                engine = EnsembleEngine(data_dir='.')

                # 获取未核验的集成预测记录
                ensemble_verified = 0
                for record in engine.recorder.records:
                    if (not record.get('verified')
                        and record.get('ensemble_id')
                        and record.get('target_period')):
                        # 查找开奖数据
                        from db_access import get_lottery_draws
                        draws = get_lottery_draws(record['lottery_type'], limit=None)
                        draw = next((d for d in draws if d['period'] == record['target_period']), None)
                        if draw:
                            result = engine.verify_ensemble(
                                record['ensemble_id'],
                                draw['red_numbers'],
                                draw['blue_numbers']
                            )
                            if 'error' not in result:
                                ensemble_verified += 1
                                lt = record['lottery_type'].upper()
                                tp = record['target_period']
                                self.update_log(f"集成核验 [{lt}] {tp}: 权重已更新")

                if ensemble_verified > 0:
                    self.update_log(f"集成模型自动核验完成：核验 {ensemble_verified} 条")
                    self._refresh_model_weights()

            except Exception as e:
                self.update_log(f"集成模型核验失败: {e}")

            self.refresh_prediction_records()
            self._refresh_model_weights()
            verified = report.get('verified', 0)
            pending = report.get('pending', 0)
            if verified > 0:
                self.update_log(f"自动核验完成：本次核验 {verified} 条，待核验 {pending} 条")
            else:
                self.update_log(f"自动核验：无新增可核验记录（待核验 {pending} 条）")
        except Exception as e:
            self.update_log(f"自动核验失败: {e}")'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open('D:/bobo/python/busuan/lottery_predictor_app_new.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Replaced successfully')
else:
    print('NOT FOUND - trying regex')
    import re
    pattern = r'(    def _do_auto_verify\\(self\\):.*?)(?=\\n    def )'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content.replace(match.group(1), new_func)
        with open('D:/bobo/python/busuan/lottery_predictor_app_new.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print('Replaced with regex')
    else:
        print('Still not found')