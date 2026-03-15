# =====================================================================
# MISCADA ASML Classification Coursework - report.R
# =====================================================================

# 1. 加载必要的 R 包
# 如果没有安装，请先运行: install.packages(c("tidyverse", "caret", "randomForest", "pROC"))
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)

# 固定随机种子，确保报告结果可完全复现 (作业要求)
set.seed(2026) 

# 2. 读取数据
# 假设源文件 bank_personal_loan.csv 与脚本在同一目录下
loan_data <- read.csv("bank_personal_loan.csv")

# 3. 数据预处理 (Data Preprocessing) - 已修复没有 ID 列的问题
clean_data <- loan_data %>%
  select(-ZIP.Code) %>%  # 现在我们只移除毫无预测价值的邮政编码列
  mutate(
    Personal.Loan = factor(Personal.Loan, levels = c(0, 1), labels = c("No", "Yes")),
    Education = factor(Education, levels = c(1, 2, 3), labels = c("Undergrad", "Graduate", "Advanced")),
    Securities.Account = factor(Securities.Account),
    CD.Account = factor(CD.Account),
    Online = factor(Online),
    CreditCard = factor(CreditCard)
  )

# 检查一下前几行，确保数据正确转换了
head(clean_data)

# 4. 简单数据可视化 (EDA - Part 2 要求)
# 图 1: 收入与是否贷款的关系 (箱线图)
p1 <- ggplot(clean_data, aes(x = Personal.Loan, y = Income, fill = Personal.Loan)) +
  geom_boxplot() +
  labs(title = "Income vs. Personal Loan Acceptance", x = "Accepted Loan?", y = "Income ($000)") +
  theme_minimal()
print(p1)

# 图 2: 教育水平与是否贷款的关系 (条形图)
p2 <- ggplot(clean_data, aes(x = Education, fill = Personal.Loan)) +
  geom_bar(position = "fill") +
  labs(title = "Loan Acceptance Proportion by Education Level", y = "Proportion", x = "Education") +
  theme_minimal()
print(p2)

# 5. 划分训练集和测试集 (Train/Test Split)
# 使用 80% 作为训练集，20% 作为测试集
trainIndex <- createDataPartition(clean_data$Personal.Loan, p = 0.8, list = FALSE)
train_data <- clean_data[trainIndex, ]
test_data  <- clean_data[-trainIndex, ]

# 6. 模型拟合 (Model Fitting - 包含交叉验证)
# 设置 5折交叉验证
fitControl <- trainControl(method = "cv", 
                           number = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)

# Baseline 模型: 逻辑回归 (Logistic Regression)
cat("Training Logistic Regression Model...\n")
log_model <- train(Personal.Loan ~ ., 
                   data = train_data, 
                   method = "glm", 
                   family = "binomial", 
                   trControl = fitControl, 
                   metric = "ROC")

# 进阶模型: 随机森林 (Random Forest)
cat("Training Random Forest Model...\n")
rf_model <- train(Personal.Loan ~ ., 
                  data = train_data, 
                  method = "rf", 
                  trControl = fitControl, 
                  metric = "ROC")

# 7. 性能报告 (Performance Report)
# 比较两个模型的训练集交叉验证结果
model_results <- resamples(list(Logistic = log_model, RandomForest = rf_model))
summary(model_results)

# 在测试集上进行预测 (使用表现更好的随机森林为例)
rf_pred_class <- predict(rf_model, newdata = test_data)
rf_pred_prob <- predict(rf_model, newdata = test_data, type = "prob")

# 混淆矩阵
conf_mat <- confusionMatrix(rf_pred_class, test_data$Personal.Loan, positive = "Yes")
print(conf_mat)

# 绘制 ROC 曲线
roc_obj <- roc(test_data$Personal.Loan, rf_pred_prob$Yes)
plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc(roc_obj), 3), ")"), col = "blue", lwd = 2)