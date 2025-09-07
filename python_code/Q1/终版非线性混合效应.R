# 加载必要的包
library(readxl)
library(nlme)
library(ggplot2)
library(lme4)
library(MuMIn)
library(mgcv)  # 用于GAMM模型
library(merTools)
library(sjPlot)
library(gridExtra)
library(itsadug)  # 用于GAM模型可视化

# 读取数据
data_path <- "D:/math_modeling/25_C/mathmodeling_2024/python_code/Q1/C(1)预处理.xlsx"
data <- read_excel(data_path)

# 查看数据结构
str(data)
head(data)

# 重命名列名以便使用
colnames(data) <- c("Subject", "BMI", "Y_concentration", "Gestational_days")

# 数据预处理
data$Subject <- as.factor(data$Subject)
data$BMI <- as.numeric(data$BMI)
data$Y_concentration <- as.numeric(data$Y_concentration)
data$Gestational_days <- as.numeric(data$Gestational_days)

# 探索性数据分析
exploratory_plot <- ggplot(data, aes(x = Gestational_days, y = Y_concentration, color = Subject)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal() +
  labs(title = "Y染色体浓度随孕周变化", x = "孕周(天)", y = "Y染色体浓度")

print(exploratory_plot)
ggsave("exploratory_plot.png", width = 10, height = 6, dpi = 300)

# 1. 线性混合模型 (LMM)
cat("拟合线性混合模型...\n")
lmm_model <- lmer(Y_concentration ~ BMI + Gestational_days + (1|Subject), data = data)
cat("线性混合模型拟合完成\n")
print(summary(lmm_model))

# 2. 指数模型
cat("拟合指数模型...\n")
exp_model_formula <- function(bmi, days, a, b, c) {
  a * exp(b * bmi + c * days)
}

tryCatch({
  exp_nlme <- nlme(Y_concentration ~ exp_model_formula(BMI, Gestational_days, a, b, c),
                   data = data,
                   fixed = a + b + c ~ 1,
                   random = a ~ 1 | Subject,
                   start = list(a = 0.05, b = 0.001, c = 0.001),
                   control = nlmeControl(maxIter = 2000, pnlsTol = 1e-4, msTol = 1e-4))
  cat("指数模型拟合成功\n")
  print(summary(exp_nlme))
}, error = function(e) {
  message("指数模型拟合失败: ", e$message)
  exp_nlme <- NULL
})

# 3. Logistic生长模型
cat("拟合Logistic模型...\n")
logistic_model_formula <- function(bmi, days, a, b, c, d) {
  a / (1 + exp(-(b * bmi + c * days - d)))
}

tryCatch({
  logistic_nlme <- nlme(Y_concentration ~ logistic_model_formula(BMI, Gestational_days, a, b, c, d),
                        data = data,
                        fixed = a + b + c + d ~ 1,
                        random = a ~ 1 | Subject,
                        start = list(a = 0.1, b = 0.01, c = 0.01, d = 50),
                        control = nlmeControl(maxIter = 2000, pnlsTol = 1e-4, msTol = 1e-4))
  cat("Logistic模型拟合成功\n")
  print(summary(logistic_nlme))
}, error = function(e) {
  message("Logistic模型拟合失败: ", e$message)
  logistic_nlme <- NULL
})

# 4. Gompertz生长模型
cat("拟合Gompertz模型...\n")
gompertz_model_formula <- function(bmi, days, a, b, c, d) {
  a * exp(-exp(-(b * bmi + c * days - d)))
}

tryCatch({
  gompertz_nlme <- nlme(Y_concentration ~ gompertz_model_formula(BMI, Gestational_days, a, b, c, d),
                        data = data,
                        fixed = a + b + c + d ~ 1,
                        random = a ~ 1 | Subject,
                        start = list(a = 0.1, b = 0.01, c = 0.01, d = 50),
                        control = nlmeControl(maxIter = 2000, pnlsTol = 1e-4, msTol = 1e-4))
  cat("Gompertz模型拟合成功\n")
  print(summary(gompertz_nlme))
}, error = function(e) {
  message("Gompertz模型拟合失败: ", e$message)
  gompertz_nlme <- NULL
})

# 5. 广义加性混合模型 (GAMM)
cat("拟合广义加性混合模型...\n")
tryCatch({
  # 使用bam函数拟合GAMM，适用于大型数据集
  gamm_model <- bam(Y_concentration ~ s(BMI) + s(Gestational_days) + s(Subject, bs = "re"),
                    data = data,
                    method = "REML")
  cat("GAMM模型拟合成功\n")
  print(summary(gamm_model))
  
  # 绘制平滑项图
  png("gamm_smooth_terms.png", width = 10, height = 6, units = "in", res = 300)
  par(mfrow = c(1, 2))
  plot(gamm_model, select = 1, main = "BMI平滑项")
  plot(gamm_model, select = 2, main = "孕周平滑项")
  dev.off()
}, error = function(e) {
  message("GAMM模型拟合失败: ", e$message)
  gamm_model <- NULL
})

# 模型比较
cat("开始模型比较...\n")
model_comparison <- data.frame(
  Model = "线性混合模型",
  AIC = AIC(lmm_model),
  BIC = BIC(lmm_model),
  LogLik = as.numeric(logLik(lmm_model))
)

# 添加非线性模型（如果成功拟合）
if(exists("exp_nlme") && !is.null(exp_nlme)) {
  model_comparison <- rbind(model_comparison, 
                            data.frame(Model = "指数模型", 
                                       AIC = AIC(exp_nlme),
                                       BIC = BIC(exp_nlme),
                                       LogLik = logLik(exp_nlme)))
}

if(exists("logistic_nlme") && !is.null(logistic_nlme)) {
  model_comparison <- rbind(model_comparison, 
                            data.frame(Model = "Logistic模型", 
                                       AIC = AIC(logistic_nlme),
                                       BIC = BIC(logistic_nlme),
                                       LogLik = logLik(logistic_nlme)))
}

if(exists("gompertz_nlme") && !is.null(gompertz_nlme)) {
  model_comparison <- rbind(model_comparison, 
                            data.frame(Model = "Gompertz模型", 
                                       AIC = AIC(gompertz_nlme),
                                       BIC = BIC(gompertz_nlme),
                                       LogLik = logLik(gompertz_nlme)))
}

if(exists("gamm_model") && !is.null(gamm_model)) {
  model_comparison <- rbind(model_comparison, 
                            data.frame(Model = "GAMM模型", 
                                       AIC = AIC(gamm_model),
                                       BIC = BIC(gamm_model),
                                       LogLik = logLik(gamm_model)))
}

# 显示模型比较结果
print("模型比较结果:")
print(model_comparison)

# 可视化拟合结果
# 创建预测函数
predict_nlme <- function(model, newdata) {
  if(inherits(model, "nlme")) {
    return(predict(model, newdata = newdata, level = 0))
  } else if(inherits(model, "lmerMod")) {
    return(predict(model, newdata = newdata, re.form = NA))
  } else if(inherits(model, "bam")) {
    return(predict(model, newdata = newdata, exclude = "s(Subject)"))
  } else {
    return(rep(NA, nrow(newdata)))
  }
}

# 生成预测数据 - 修复：使用平均BMI值
pred_data <- data.frame(
  BMI = rep(mean(data$BMI), 100),  # 使用平均BMI值
  Gestational_days = seq(min(data$Gestational_days), max(data$Gestational_days), length.out = 100),
  Subject = rep(levels(data$Subject)[1], 100)  # 使用第一个受试者
)

# 为每个模型生成预测
pred_data$lmm <- predict_nlme(lmm_model, pred_data)

if(exists("exp_nlme") && !is.null(exp_nlme)) {
  pred_data$exp <- predict_nlme(exp_nlme, pred_data)
}

if(exists("logistic_nlme") && !is.null(logistic_nlme)) {
  pred_data$logistic <- predict_nlme(logistic_nlme, pred_data)
}

if(exists("gompertz_nlme") && !is.null(gompertz_nlme)) {
  pred_data$gompertz <- predict_nlme(gompertz_nlme, pred_data)
}

if(exists("gamm_model") && !is.null(gamm_model)) {
  pred_data$gamm <- predict_nlme(gamm_model, pred_data)
}

# 创建每个模型的拟合图
plots <- list()

# 线性混合模型
plots[[1]] <- ggplot() +
  geom_point(data = data, aes(x = Gestational_days, y = Y_concentration, color = Subject), alpha = 0.6) +
  geom_line(data = pred_data, aes(x = Gestational_days, y = lmm, color = "总体拟合"), size = 1) +
  theme_minimal() +
  labs(title = "线性混合模型拟合", x = "孕周(天)", y = "Y染色体浓度")

# 指数模型
if(exists("exp_nlme") && !is.null(exp_nlme)) {
  plots[[2]] <- ggplot() +
    geom_point(data = data, aes(x = Gestational_days, y = Y_concentration, color = Subject), alpha = 0.6) +
    geom_line(data = pred_data, aes(x = Gestational_days, y = exp, color = "总体拟合"), size = 1) +
    theme_minimal() +
    labs(title = "指数模型拟合", x = "孕周(天)", y = "Y染色体浓度")
}

# Logistic模型
if(exists("logistic_nlme") && !is.null(logistic_nlme)) {
  plots[[3]] <- ggplot() +
    geom_point(data = data, aes(x = Gestational_days, y = Y_concentration, color = Subject), alpha = 0.6) +
    geom_line(data = pred_data, aes(x = Gestational_days, y = logistic, color = "总体拟合"), size = 1) +
    theme_minimal() +
    labs(title = "Logistic模型拟合", x = "孕周(天)", y = "Y染色体浓度")
}

# Gompertz模型
if(exists("gompertz_nlme") && !is.null(gompertz_nlme)) {
  plots[[4]] <- ggplot() +
    geom_point(data = data, aes(x = Gestational_days, y = Y_concentration, color = Subject), alpha = 0.6) +
    geom_line(data = pred_data, aes(x = Gestational_days, y = gompertz, color = "总体拟合"), size = 1) +
    theme_minimal() +
    labs(title = "Gompertz模型拟合", x = "孕周(天)", y = "Y染色体浓度")
}

# GAMM模型
if(exists("gamm_model") && !is.null(gamm_model)) {
  plots[[5]] <- ggplot() +
    geom_point(data = data, aes(x = Gestational_days, y = Y_concentration, color = Subject), alpha = 0.6) +
    geom_line(data = pred_data, aes(x = Gestational_days, y = gamm, color = "总体拟合"), size = 1) +
    theme_minimal() +
    labs(title = "GAMM模型拟合", x = "孕周(天)", y = "Y染色体浓度")
}

# 保存所有图表
for(i in seq_along(plots)) {
  ggsave(paste0("model_fit_", i, ".png"), plots[[i]], width = 10, height = 6, dpi = 300)
}

# 创建模型比较图
model_comp_plot <- ggplot(model_comparison, aes(x = reorder(Model, AIC), y = AIC, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "模型比较 - AIC值", x = "模型", y = "AIC值") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set2")

ggsave("model_comparison_aic.png", model_comp_plot, width = 10, height = 6, dpi = 300)

# 残差分析
residual_plots <- list()

# 线性混合模型残差
residual_plots[[1]] <- ggplot(data, aes(x = fitted(lmm_model), y = resid(lmm_model))) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_minimal() +
  labs(title = "线性混合模型残差图", x = "拟合值", y = "残差")

# 指数模型残差
if(exists("exp_nlme") && !is.null(exp_nlme)) {
  residual_plots[[2]] <- ggplot(data, aes(x = fitted(exp_nlme), y = resid(exp_nlme))) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    theme_minimal() +
    labs(title = "指数模型残差图", x = "拟合值", y = "残差")
}

# GAMM模型残差
if(exists("gamm_model") && !is.null(gamm_model)) {
  residual_plots[[3]] <- ggplot(data, aes(x = fitted(gamm_model), y = resid(gamm_model))) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    theme_minimal() +
    labs(title = "GAMM模型残差图", x = "拟合值", y = "残差")
}

# 保存残差图
for(i in seq_along(residual_plots)) {
  ggsave(paste0("residual_plot_", i, ".png"), residual_plots[[i]], width = 10, height = 6, dpi = 300)
}

# 输出模型摘要
sink("model_summaries.txt")
cat("线性混合模型摘要:\n")
print(summary(lmm_model))

if(exists("exp_nlme") && !is.null(exp_nlme)) {
  cat("\n指数模型摘要:\n")
  print(summary(exp_nlme))
}

if(exists("logistic_nlme") && !is.null(logistic_nlme)) {
  cat("\nLogistic模型摘要:\n")
  print(summary(logistic_nlme))
}

if(exists("gompertz_nlme") && !is.null(gompertz_nlme)) {
  cat("\nGompertz模型摘要:\n")
  print(summary(gompertz_nlme))
}

if(exists("gamm_model") && !is.null(gamm_model)) {
  cat("\nGAMM模型摘要:\n")
  print(summary(gamm_model))
}

cat("\n模型比较:\n")
print(model_comparison)
sink()

# 显示最佳模型
if(nrow(model_comparison) > 0) {
  best_model_idx <- which.min(model_comparison$AIC)
  cat("根据AIC准则，最佳模型是:", model_comparison$Model[best_model_idx], "\n")
  
  # 绘制最佳模型拟合图
  if(model_comparison$Model[best_model_idx] == "线性混合模型") {
    best_plot <- plots[[1]]
  } else if(model_comparison$Model[best_model_idx] == "指数模型") {
    best_plot <- plots[[2]]
  } else if(model_comparison$Model[best_model_idx] == "Logistic模型") {
    best_plot <- plots[[3]]
  } else if(model_comparison$Model[best_model_idx] == "Gompertz模型") {
    best_plot <- plots[[4]]
  } else if(model_comparison$Model[best_model_idx] == "GAMM模型") {
    best_plot <- plots[[5]]
  }
  
  ggsave("best_model_fit.png", best_plot, width = 10, height = 6, dpi = 300)
}

# 保存工作空间
save.image("mixed_model_analysis.RData")

cat("分析完成！所有结果已保存到当前工作目录。\n")