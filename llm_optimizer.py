# -*- coding: utf-8 -*-
"""
LLM 决策助手 (LLM Optimizer)
=============================
调用大模型分析健康报告，生成可解释的优化建议。

策略:
  1. 优先调用 OpenAI 兼容 API（支持 OpenAI / DeepSeek / 本地 Ollama）
  2. API 不可用时降级到规则引擎

操作分级:
  - auto_executable=True:  update_weights, calibrate（低风险，可自动执行）
  - auto_executable=False: retrain_model, update_data（高风险，需用户确认）
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 配置 ──────────────────────────────────────────────

_DEFAULT_CONFIG = {
    'llm_provider': 'openai',       # openai / deepseek / ollama / none
    'llm_model': 'gpt-4o-mini',
    'llm_temperature': 0.3,
    'llm_max_tokens': 2000,
    'llm_timeout_seconds': 30,
    'llm_fallback_to_rules': True,
    'openai_api_key_env': 'OPENAI_API_KEY',
    'openai_base_url': None,         # None=默认, 或自定义(如 DeepSeek/Ollama)
    'ollama_host': 'http://localhost:11434',
}


def get_config() -> Dict:
    """获取 LLM 配置（合并默认值和环境变量）"""
    config = dict(_DEFAULT_CONFIG)
    # 环境变量覆盖
    if os.environ.get('LLM_PROVIDER'):
        config['llm_provider'] = os.environ['LLM_PROVIDER']
    if os.environ.get('LLM_MODEL'):
        config['llm_model'] = os.environ['LLM_MODEL']
    if os.environ.get('OPENAI_BASE_URL'):
        config['openai_base_url'] = os.environ['OPENAI_BASE_URL']
    if os.environ.get('OLLAMA_HOST'):
        config['ollama_host'] = os.environ['OLLAMA_HOST']
    return config


# ── Prompt 模板 ───────────────────────────────────────

_SYSTEM_PROMPT = """你是彩票预测系统的优化专家。请分析系统健康报告，给出诊断和建议。

## 输出要求
输出 JSON 格式（不要 markdown 代码块），包含以下字段：
- diagnosis: 系统健康度评估（一句话）
- health_score: 0-10 分
- issues: 问题列表 [{name, severity, detail}]
- recommendations: 建议列表 [{
    action: "update_weights"|"calibrate"|"retrain_model"|"update_data"|"noop",
    priority: "high"|"medium"|"low",
    auto_executable: true|false,
    reason: "为什么",
    params: {具体参数}
  }]
- explanation: 给用户的通俗解释（2-3句话）

## 判定规则
- auto_executable=true: update_weights, calibrate（低风险，可自动执行）
- auto_executable=false: retrain_model, update_data（高风险，需用户确认）
- 如果健康度>=8且无 critical 问题，建议 noop
- 数据不足时优先建议 noop，不要建议重训"""


def _build_user_prompt(report: Dict, lottery_type: str) -> str:
    """构建用户 prompt"""
    # 简化报告，避免 token 浪费
    compact = {
        'lottery_type': lottery_type,
        'health_score': report.get('health_score'),
        'priority': report.get('priority'),
        'checks': {},
    }
    for ck_name, ck_val in report.get('checks', {}).items():
        compact['checks'][ck_name] = {
            'status': ck_val.get('status'),
            'score': ck_val.get('score'),
            'message': ck_val.get('message', ''),
        }
        # 关键数据
        if ck_name == 'model_performance' and 'details' in ck_val:
            compact['checks'][ck_name]['details'] = {
                mt: {'samples': d.get('samples', 0),
                     'trend': d.get('trend', 'unknown')}
                for mt, d in ck_val['details'].items()
            }
        elif ck_name == 'feature_drift':
            compact['checks'][ck_name]['drifted'] = ck_val.get('drifted_features', [])
            compact['checks'][ck_name]['max_kl'] = ck_val.get('max_kl', 0)

    return f"## 系统健康报告\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n请给出诊断和建议。"


# ── LLM 调用 ──────────────────────────────────────────

def _call_openai_api(config: Dict, system_prompt: str, user_prompt: str) -> Optional[str]:
    """调用 OpenAI 兼容 API"""
    try:
        from openai import OpenAI

        api_key = os.environ.get(config['openai_api_key_env'])
        if not api_key:
            logger.info('No API key found, falling back to rules')
            return None

        base_url = config.get('openai_base_url')
        kwargs = {'api_key': api_key, 'timeout': config['llm_timeout_seconds']}
        if base_url:
            kwargs['base_url'] = base_url

        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=config['llm_model'],
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=config['llm_temperature'],
            max_tokens=config['llm_max_tokens'],
        )
        return response.choices[0].message.content
    except ImportError:
        logger.warning('openai package not available')
        return None
    except Exception as e:
        logger.warning(f'OpenAI API call failed: {e}')
        return None


def _call_ollama_api(config: Dict, system_prompt: str, user_prompt: str) -> Optional[str]:
    """调用本地 Ollama API（免 Key）"""
    try:
        import urllib.request
        import urllib.error

        host = config.get('ollama_host', 'http://localhost:11434')
        model = config.get('llm_model', 'qwen2.5:7b')
        url = f'{host}/api/chat'

        payload = json.dumps({
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'stream': False,
            'options': {
                'temperature': config['llm_temperature'],
            },
        }).encode('utf-8')

        req = urllib.request.Request(url, data=payload,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=config['llm_timeout_seconds']) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('message', {}).get('content', '')
    except Exception as e:
        logger.warning(f'Ollama API call failed: {e}')
        return None


def _parse_llm_response(text: str) -> Optional[Dict]:
    """解析 LLM 返回的 JSON（容错处理）"""
    if not text:
        return None

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块（去掉 markdown 代码块）
    import re
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(f'Failed to parse LLM response as JSON: {text[:200]}')
    return None


# ── 规则引擎降级 ──────────────────────────────────────

def _rule_based_analysis(report: Dict, lottery_type: str) -> Dict:
    """规则引擎降级：无 LLM 时用规则匹配生成建议"""
    checks = report.get('checks', {})
    issues = []
    recommendations = []
    score = report.get('health_score', 5.0)

    # 数据量不足
    dv = checks.get('data_volume', {})
    if dv.get('status') == 'critical':
        issues.append({
            'name': '数据严重不足',
            'severity': 'high',
            'detail': dv.get('message', '核验记录不足'),
        })
        recommendations.append({
            'action': 'noop',
            'priority': 'high',
            'auto_executable': True,
            'reason': '数据不足，建议先积累预测+核验记录（至少10条）',
            'params': {'lottery_type': lottery_type},
        })

    # 特征漂移
    fd = checks.get('feature_drift', {})
    if fd.get('status') == 'critical':
        issues.append({
            'name': '特征漂移显著',
            'severity': 'high',
            'detail': fd.get('message', ''),
        })
        recommendations.append({
            'action': 'retrain_model',
            'priority': 'high',
            'auto_executable': False,
            'reason': f'特征漂移(max KL={fd.get("max_kl", 0):.3f})，建议用近期50期数据重训',
            'params': {'lottery_type': lottery_type, 'recent_periods': 50},
        })

    # 模型表现下降
    mp = checks.get('model_performance', {})
    if mp.get('status') in ('warning', 'critical'):
        declining = []
        details = mp.get('details', {})
        for mt, d in details.items():
            if d.get('trend') == 'declining':
                declining.append(mt)
        if declining:
            issues.append({
                'name': '部分模型表现下降',
                'severity': 'medium',
                'detail': f'下降模型: {", ".join(declining)}',
            })
            recommendations.append({
                'action': 'update_weights',
                'priority': 'medium',
                'auto_executable': True,
                'reason': f'模型 {", ".join(declining)} 表现下降，重算权重',
                'params': {'lottery_type': lottery_type},
            })

    # 校准缺失
    cq = checks.get('calibration_quality', {})
    if cq.get('status') == 'warning' and '尚未校准' in cq.get('message', ''):
        dv_count = dv.get('value', 0)
        if dv_count >= 20:
            recommendations.append({
                'action': 'calibrate',
                'priority': 'medium',
                'auto_executable': True,
                'reason': '已有足够核验记录，可以训练校准模型',
                'params': {'lottery_type': lottery_type},
            })

    # 数据过期
    dfresh = checks.get('data_freshness', {})
    if dfresh.get('status') == 'critical':
        issues.append({
            'name': '开奖数据过期',
            'severity': 'high',
            'detail': dfresh.get('message', ''),
        })
        recommendations.append({
            'action': 'update_data',
            'priority': 'high',
            'auto_executable': False,
            'reason': '开奖数据已过期，建议更新',
            'params': {'lottery_type': lottery_type},
        })

    # 无需操作
    if not recommendations:
        recommendations.append({
            'action': 'noop',
            'priority': 'low',
            'auto_executable': True,
            'reason': '系统状态良好，无需优化',
            'params': {},
        })

    # 生成诊断文本
    if score >= 8:
        diagnosis = f'{lottery_type.upper()} 系统健康度 {score}/10，状态良好'
    elif score >= 5:
        diagnosis = f'{lottery_type.upper()} 系统健康度 {score}/10，存在可改进项'
    else:
        diagnosis = f'{lottery_type.upper()} 系统健康度 {score}/10，需要关注'

    # 通俗解释
    issue_names = [i['name'] for i in issues]
    if issue_names:
        explanation = f'当前 {lottery_type.upper()} 预测系统健康度 {score}/10。'
        explanation += f'主要问题: {"、".join(issue_names)}。'
        auto_count = sum(1 for r in recommendations if r['auto_executable'] and r['action'] != 'noop')
        manual_count = sum(1 for r in recommendations if not r['auto_executable'])
        if auto_count:
            explanation += f'{auto_count} 项操作可自动执行。'
        if manual_count:
            explanation += f'{manual_count} 项操作需要您确认。'
    else:
        explanation = f'当前 {lottery_type.upper()} 预测系统运行良好，无需优化。'

    return {
        'diagnosis': diagnosis,
        'health_score': score,
        'issues': issues,
        'recommendations': recommendations,
        'explanation': explanation,
        'source': 'rules',  # 标记来源
    }


# ── 主入口 ────────────────────────────────────────────

def analyze_health_report(report: Dict, lottery_type: str) -> Dict:
    """
    调用 LLM 分析健康报告，生成优化建议

    Args:
        report: optimization_monitor.generate_health_report() 的返回值
        lottery_type: 'ssq' / 'dlt'

    Returns:
        {
            diagnosis, health_score, issues, recommendations, explanation,
            source: "llm" / "rules"
        }
    """
    config = get_config()
    provider = config['llm_provider']

    # 尝试 LLM
    if provider != 'none':
        system_prompt = _SYSTEM_PROMPT
        user_prompt = _build_user_prompt(report, lottery_type)

        llm_text = None
        if provider == 'ollama':
            llm_text = _call_ollama_api(config, system_prompt, user_prompt)
        else:
            llm_text = _call_openai_api(config, system_prompt, user_prompt)

        if llm_text:
            parsed = _parse_llm_response(llm_text)
            if parsed and 'recommendations' in parsed:
                parsed['source'] = 'llm'
                # 确保 auto_executable 字段存在
                for rec in parsed.get('recommendations', []):
                    if 'auto_executable' not in rec:
                        rec['auto_executable'] = rec.get('action') in ('update_weights', 'calibrate', 'noop')
                    if 'params' not in rec:
                        rec['params'] = {'lottery_type': lottery_type}
                return parsed

    # 降级到规则引擎
    if config['llm_fallback_to_rules']:
        return _rule_based_analysis(report, lottery_type)

    # 无降级
    return {
        'diagnosis': 'LLM 不可用且未启用降级',
        'health_score': report.get('health_score', 0),
        'issues': [],
        'recommendations': [{'action': 'noop', 'priority': 'low',
                            'auto_executable': True, 'reason': '无法分析',
                            'params': {}}],
        'explanation': 'LLM 服务不可用，无法生成优化建议。',
        'source': 'none',
    }


def is_llm_available() -> bool:
    """检查 LLM 是否可用（有 API Key 或本地 Ollama）"""
    config = get_config()
    provider = config['llm_provider']
    if provider == 'none':
        return False
    if provider == 'ollama':
        # 检查 Ollama 是否运行
        try:
            import urllib.request
            host = config.get('ollama_host', 'http://localhost:11434')
            req = urllib.request.Request(f'{host}/api/tags')
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False
    # OpenAI/DeepSeek
    return bool(os.environ.get(config['openai_api_key_env']))


if __name__ == '__main__':
    from optimization_monitor import generate_health_report

    import sys
    lt = sys.argv[1] if len(sys.argv) > 1 else 'dlt'
    report = generate_health_report(lt)
    result = analyze_health_report(report, lt)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f'\n[LLM available: {is_llm_available()}, source: {result.get("source")}]')
