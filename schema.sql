-- busuan 彩票预测系统数据库 Schema v3.0
-- 替代 JSON 文件，提供事务安全、版本控制、完整性约束

-- ============================================================
-- 表1: lottery_draws - 开奖记录（替代 CSV）
-- ============================================================
CREATE TABLE IF NOT EXISTS lottery_draws (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lottery_type TEXT NOT NULL CHECK(lottery_type IN ('ssq', 'dlt')),
    period INTEGER NOT NULL,  -- 期号
    draw_date DATE NOT NULL,  -- 开奖日期
    red_numbers TEXT NOT NULL,  -- JSON array，例如 [1,5,12,23,28,33]
    blue_numbers TEXT NOT NULL, -- JSON array，SSQ=[16], DLT=[3,7]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(lottery_type, period)  -- 同一彩种同一期号唯一
);

CREATE INDEX IF NOT EXISTS idx_lottery_draws_type_period ON lottery_draws(lottery_type, period DESC);
CREATE INDEX IF NOT EXISTS idx_lottery_draws_date ON lottery_draws(lottery_type, draw_date DESC);

-- ============================================================
-- 表2: predictions - 预测记录（替代 prediction_history.json）
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predict_time TIMESTAMP NOT NULL,
    lottery_type TEXT NOT NULL CHECK(lottery_type IN ('ssq', 'dlt')),
    model_type TEXT NOT NULL,  -- 'ensemble', 'xgboost', 'lightgbm', 'catboost', 'random_forest', 'gbdt', 'expected_value'
    latest_period INTEGER NOT NULL,  -- 预测时已知的最新期号
    red_numbers TEXT NOT NULL,  -- JSON array
    blue_numbers TEXT NOT NULL, -- JSON array
    red_probabilities TEXT,     -- JSON array, nullable（旧记录无概率）
    blue_probabilities TEXT,    -- JSON array, nullable
    
    -- 核验结果（未核验时为 NULL）
    target_period INTEGER,      -- 预测目标期号（核验后填入）
    red_hits INTEGER,           -- 红球命中个数
    blue_hits INTEGER,          -- 蓝球命中个数
    prize_name TEXT,            -- 奖级名称（如"六等奖"）
    prize_amount REAL,          -- 奖金金额
    
    verified_at TIMESTAMP,      -- 核验时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 外键约束（可选，若开奖记录存在）
    FOREIGN KEY (lottery_type, target_period) REFERENCES lottery_draws(lottery_type, period)
);

CREATE INDEX IF NOT EXISTS idx_predictions_lottery_period ON predictions(lottery_type, latest_period DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_verified ON predictions(verified_at) WHERE verified_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(lottery_type, target_period) WHERE target_period IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_predictions_model_performance ON predictions(lottery_type, model_type, verified_at DESC) WHERE verified_at IS NOT NULL;

-- ============================================================
-- 表3: model_weights_history - 模型权重历史（替代 model_weights.json 单点快照）
-- ============================================================
CREATE TABLE IF NOT EXISTS model_weights_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lottery_type TEXT NOT NULL CHECK(lottery_type IN ('ssq', 'dlt')),
    updated_at TIMESTAMP NOT NULL,
    window TEXT NOT NULL,  -- JSON 配置（如 {"max_records": 200, "alpha": 0.05}）
    
    -- 模型权重 JSON（如 {"xgboost": 0.25, "lightgbm": 0.22, ...}）
    weights TEXT NOT NULL,
    
    -- 子模型权重 JSON（ensemble 内部权重）
    ensemble_sub_weights TEXT,
    
    -- 推荐模型
    recommended_model TEXT,
    
    -- 性能指标（用于回滚判断）
    avg_red_hits REAL,
    avg_blue_hits REAL,
    hit_rate REAL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_weights_lottery_time ON model_weights_history(lottery_type, updated_at DESC);

-- ============================================================
-- 表4: calibration_models - 校准模型参数
-- ============================================================
CREATE TABLE IF NOT EXISTS calibration_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lottery_type TEXT NOT NULL CHECK(lottery_type IN ('ssq', 'dlt')),
    ball_type TEXT NOT NULL CHECK(ball_type IN ('red', 'blue')),
    model_type TEXT NOT NULL,  -- 校准的原始模型类型
    
    -- 校准方法（'isotonic' or 'platt'）
    calibration_method TEXT NOT NULL CHECK(calibration_method IN ('isotonic', 'platt')),
    
    -- 校准模型序列化数据（pickle 二进制，Base64 编码后存储）
    calibrator_data BLOB NOT NULL,
    
    -- 训练元数据
    train_samples INTEGER NOT NULL,  -- 训练样本数
    train_date TIMESTAMP NOT NULL,
    calibration_score REAL,  -- 校准前后改善分数
    
    -- 校准前后误差对比
    before_error REAL,
    after_error REAL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(lottery_type, ball_type, model_type, calibration_method)
);

CREATE INDEX IF NOT EXISTS idx_calibration_lottery_ball ON calibration_models(lottery_type, ball_type, train_date DESC);

-- ============================================================
-- 表5: investment_plans - 投注计划（替代 investment_history.json）
-- ============================================================
CREATE TABLE IF NOT EXISTS investment_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lottery_type TEXT NOT NULL CHECK(lottery_type IN ('ssq', 'dlt')),
    target_period INTEGER NOT NULL,  -- 目标期号
    plan_date DATE NOT NULL,
    
    -- 投注号码（JSON array of arrays，如 [[1,2,3,4,5,6], [7,8,...]]）
    bet_numbers TEXT NOT NULL,
    
    -- 投注金额
    total_amount REAL NOT NULL,
    bet_count INTEGER NOT NULL,  -- 注数
    
    -- 关联的预测记录 ID（可多个）
    linked_prediction_ids TEXT,  -- JSON array
    
    -- 核验结果
    actual_win_amount REAL,
    verified BOOLEAN DEFAULT 0,
    
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_investment_lottery_period ON investment_plans(lottery_type, target_period DESC);
CREATE INDEX IF NOT EXISTS idx_investment_verified ON investment_plans(verified, lottery_type);

-- ============================================================
-- 表6: optimization_log - 优化操作日志（迁移自 optimization_log.py）
-- ============================================================
CREATE TABLE IF NOT EXISTS optimization_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lottery_type TEXT NOT NULL CHECK(lottery_type IN ('ssq', 'dlt', 'all')),
    timestamp TIMESTAMP NOT NULL,
    
    -- 操作类型
    action_type TEXT NOT NULL,  -- 'update_weights', 'calibrate_model', 'retrain_model', 'feature_analysis'
    
    -- 操作参数（JSON）
    action_params TEXT,
    
    -- 执行状态
    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'partial')),
    
    -- 执行模式
    execution_mode TEXT NOT NULL CHECK(execution_mode IN ('auto', 'manual')),
    
    -- 执行结果摘要
    result_summary TEXT,
    
    -- 效果对比（JSON，如 {"before_hit_rate": 0.055, "after_hit_rate": 0.062}）
    effect_metrics TEXT,
    
    -- 错误信息（失败时）
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_optimization_log_lottery_time ON optimization_log(lottery_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_optimization_log_action ON optimization_log(action_type, status);

-- ============================================================
-- 表7: analysis_cache - 分析结果缓存
-- ============================================================
CREATE TABLE IF NOT EXISTS analysis_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lottery_type TEXT NOT NULL CHECK(lottery_type IN ('ssq', 'dlt')),
    analysis_type TEXT NOT NULL,  -- 'frequency', 'hot_cold', 'missing', 'pattern', 'trend', 'advanced_stats'
    
    -- 缓存键（如 'ssq_frequency_last_100'）
    cache_key TEXT NOT NULL UNIQUE,
    
    -- 分析结果（JSON）
    result_data TEXT NOT NULL,
    
    -- 数据时间范围
    data_from_period INTEGER,
    data_to_period INTEGER,
    
    -- 缓存过期时间
    expires_at TIMESTAMP NOT NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analysis_cache_key ON analysis_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_analysis_cache_expires ON analysis_cache(expires_at);

-- ============================================================
-- 触发器：自动更新 updated_at
-- ============================================================
CREATE TRIGGER IF NOT EXISTS update_analysis_cache_timestamp 
AFTER UPDATE ON analysis_cache
BEGIN
    UPDATE analysis_cache SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================================
-- 视图：便捷查询
-- ============================================================

-- 视图1: 最新权重快照（每个彩种的最新记录）
-- SQLite 不支持 DISTINCT ON，改用子查询
CREATE VIEW IF NOT EXISTS v_latest_model_weights AS
SELECT mwh.*
FROM model_weights_history mwh
INNER JOIN (
    SELECT lottery_type, MAX(updated_at) as max_updated
    FROM model_weights_history
    GROUP BY lottery_type
) latest ON mwh.lottery_type = latest.lottery_type AND mwh.updated_at = latest.max_updated;

-- 视图2: 已核验预测记录（含命中率）
CREATE VIEW IF NOT EXISTS v_verified_predictions AS
SELECT 
    p.*,
    CAST(p.red_hits AS REAL) / CASE 
        WHEN p.lottery_type = 'ssq' THEN 6.0
        WHEN p.lottery_type = 'dlt' THEN 5.0
    END AS red_hit_rate,
    CAST(p.blue_hits AS REAL) / CASE 
        WHEN p.lottery_type = 'ssq' THEN 1.0
        WHEN p.lottery_type = 'dlt' THEN 2.0
    END AS blue_hit_rate
FROM predictions p
WHERE p.verified_at IS NOT NULL;

-- 视图3: 模型表现排行（近 30 天）
CREATE VIEW IF NOT EXISTS v_model_performance_30d AS
SELECT 
    lottery_type,
    model_type,
    COUNT(*) AS prediction_count,
    AVG(red_hits) AS avg_red_hits,
    AVG(blue_hits) AS avg_blue_hits,
    SUM(CASE WHEN prize_amount > 0 THEN 1 ELSE 0 END) AS win_count,
    SUM(prize_amount) AS total_prize,
    MAX(verified_at) AS last_verified
FROM predictions
WHERE verified_at >= datetime('now', '-30 days')
GROUP BY lottery_type, model_type
ORDER BY lottery_type, avg_red_hits DESC, avg_blue_hits DESC;

-- ============================================================
-- 数据完整性约束补充说明
-- ============================================================

-- 约束1: predictions.red_numbers 必须是合法的 JSON 数组
-- 约束2: predictions.red_probabilities 如果非空，数组长度必须与号码一致
-- 约束3: model_weights_history.weights 必须是合法的 JSON 对象
-- 这些约束在 Python 层强制执行（通过 json.loads 验证）

-- ============================================================
-- 初始化数据（可选）
-- ============================================================

-- 插入一个标记记录，表示数据库已初始化
CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO system_metadata (key, value) VALUES ('schema_version', '3.0');
INSERT OR IGNORE INTO system_metadata (key, value) VALUES ('migrated_from_json', 'false');
INSERT OR IGNORE INTO system_metadata (key, value) VALUES ('last_backup', '');

-- ============================================================
-- 性能优化建议
-- ============================================================

-- 1. 定期 VACUUM（重建数据库，释放空间）
--    建议：每周执行一次 VACUUM;

-- 2. 定期 ANALYZE（更新统计信息，优化查询计划）
--    建议：每次大量插入后执行 ANALYZE;

-- 3. 备份策略
--    建议：每天 03:00 自动备份到 backup/busuan_YYYYMMDD.db
--    保留最近 30 天备份

-- ============================================================
-- Schema 设计说明
-- ============================================================

-- 设计原则：
-- 1. 规范化（3NF）：减少数据冗余
-- 2. 历史追溯：权重、校准、优化日志都保留历史版本
-- 3. 软删除：不直接 DELETE，使用 deleted_at 字段（未来扩展）
-- 4. 索引优化：针对常见查询模式（按彩种、按时间、按核验状态）
-- 5. 类型安全：使用 CHECK 约束确保枚举值合法

-- 迁移兼容：
-- - JSON 文件中的所有字段都保留
-- - 新增字段向后兼容（nullable 或有默认值）
-- - 提供迁移脚本一键导入现有 524 条记录

-- 查询性能：
-- - 主键自增 INTEGER PRIMARY KEY（SQLite 优化）
-- - 复合索引覆盖 WHERE + ORDER BY 常见查询
-- - 视图预计算常用聚合（减少 JOIN 开销）
