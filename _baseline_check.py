# -*- coding: utf-8 -*-
"""数据文件基线指纹检查（测试前后比对，确保零污染）"""
import json
import hashlib
import os
import sys

FILES = [
    'prediction_history.json',
    'investment_history.json',
    'model_weights.json',
    'scripts/ssq/ssq_history.csv',
    'scripts/dlt/dlt_history.csv',
]


def fingerprint(path):
    """计算单个文件指纹"""
    if not os.path.exists(path):
        return {'path': path, 'exists': False}
    data = open(path, 'rb').read()
    info = {
        'path': path,
        'exists': True,
        'md5': hashlib.md5(data).hexdigest(),
        'size': len(data),
    }
    if path.endswith('.json'):
        try:
            obj = json.loads(data.decode('utf-8'))
            info['items'] = len(obj)
        except Exception as exc:
            info['parse_err'] = str(exc)[:40]
    return info


def main():
    """主入口：print 打印 / save 保存基线 / verify 比对基线"""
    mode = sys.argv[1] if len(sys.argv) > 1 else 'print'
    result = [fingerprint(p) for p in FILES]

    if mode == 'save':
        with open('.test_backup/baseline.json', 'w', encoding='utf-8') as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print('BASELINE_SAVED')
        for r in result:
            print('  %-40s md5=%s size=%s items=%s' % (
                r['path'], r.get('md5', 'N/A')[:12], r.get('size', 'N/A'), r.get('items', '-')))

    elif mode == 'verify':
        baseline = json.load(open('.test_backup/baseline.json', encoding='utf-8'))
        base_map = {b['path']: b for b in baseline}
        dirty = []
        for r in result:
            b = base_map.get(r['path'], {})
            same = (r.get('md5') == b.get('md5'))
            tag = 'OK' if same else 'CHANGED'
            if not same:
                dirty.append(r['path'])
            print('  [%s] %s' % (tag, r['path']))
        if dirty:
            print('DIRTY FILES: ' + ', '.join(dirty))
            return 1
        print('ALL_CLEAN')
        return 0
    else:
        for r in result:
            print('  %-40s md5=%s size=%s items=%s' % (
                r['path'], r.get('md5', 'N/A')[:12], r.get('size', 'N/A'), r.get('items', '-')))
    return 0


if __name__ == '__main__':
    sys.exit(main())