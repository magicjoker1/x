class EarlyStopping:
    """
    优化后的早停机制：
    1. 保存验证损失最低时的模型权重
    2. 更合理的改善判断逻辑
    3. 支持设置保存路径
    """

    def __init__(self, patience=15, min_delta=1e-5, save_path=None):
        """
        :param patience: 容忍验证损失不改善的epoch数量
        :param min_delta: 被视为"改善"的最小损失下降值
        :param save_path: 最优模型权重的保存路径
        """
        self.patience = patience
        self.min_delta = min_delta  # 损失至少要下降这么多才算改善
        self.save_path = save_path  # 新增：保存最优模型

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model=None):
        # 第一次记录最佳损失
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_model(model)  # 保存初始最佳模型
            return

        # 计算当前损失与最佳损失的差异
        loss_improvement = self.best_loss - val_loss

        # 情况1：损失改善超过阈值（变好）
        if loss_improvement > self.min_delta:
            self.best_loss = val_loss  # 更新最佳损失
            self.counter = 0  # 重置计数器
            self._save_model(model)  # 保存新的最优模型
            print(f"验证损失改善 {loss_improvement:.6f}，更新最佳模型")

        # 情况2：损失无改善或上升（未变好）
        else:
            self.counter += 1
            print(f"早停计数器: {self.counter}/{self.patience} (当前损失: {val_loss:.6f})")

            # 达到容忍次数，触发早停
            if self.counter >= self.patience:
                print(f"早停触发！最佳验证损失: {self.best_loss:.6f}")
                self.early_stop = True

    def _save_model(self, model):
        """保存最优模型权重（如果提供了模型和保存路径）"""
        if self.save_path and model is not None:
            torch.save(model.state_dict(), self.save_path)
            print(f"最优模型已保存至: {self.save_path}")
