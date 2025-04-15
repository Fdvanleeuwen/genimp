# Sim of gain
# Ideas: Generator for each missing data pattern, use complete cases in the D
# loss is Wasserstein-1 distance 
# Multiple Imputation via Generative Adversarial Network for High-dimensional Blockwise Missing Value Problems

# Other loss functions:
#  Frechet Inception Distance (FID) 
# generalized energy distance 

source("func_gain_torch.R")
library(tidyverse)
# Step 1: generate data

# Set seed
set.seed(123)
df = gendat(N = 1000, P = 5, rho = 0.5, R2 = 0.5)
df_test = gendat(N = 1000, P = 5, rho = 0.5, R2 = 0.5)

# Step 2: Induce missings
P = 5
propmiss = 0.5
#pattern <- 1-diag(P+1)
pattern <- (1 - upper.tri(diag(P+1)))[2:P,]
amp <- mice::ampute(df, prop = propmiss, patterns = pattern, mech = "MCAR")$amp

ggmice::plot_pattern(amp)

# Step 3: gain

n_iter = 10
imps = gain_train(data = amp, hidden_dim = 10, lr = 0.001,
                  n_iter = n_iter, hint_rate = 0.5, m = 5)

df_plot = data.frame(index = c(1:n_iter, 1:n_iter), value = c(imps$losses$G_loss,imps$losses$D_loss),
                     loss = c(rep("G", n_iter), rep("D", n_iter))) 

imps$losses$G_loss[1000]
imps$losses$G_loss[10000]

imps$losses$D_loss[1000]
imps$losses$D_loss[10000]

df_plot %>% 
  ggplot(aes(x = index, y = value)) + 
  geom_point() +
  facet_wrap(~loss, scales = "free_y") +
  theme_minimal()

# Step 4: evaluate the imputations
df_imp = as.array(imps$preds[[1]])
df_imp = data.frame(Y = df_imp[,1], X = df_imp[,-1])

out_mice = mice(amp)
out_mice_c = complete(out_mice, "all")

cor(df[-1, -1])
cor(df_imp[-1, -1])
cor(out_mice_c$`1`[-1, -1])

plot(df[,4], df[,5])
plot(df_imp[,4], df_imp[,5])
plot(out_mice_c$`1`[,4], out_mice_c$`1`[,5])

out_test = eval_gain(data = df_imp, data_test = df_test, data_real = df, df_mask = imps$mask, P = P)


# some sanity checks
df_use = (imps$mask * df) + ((1 - imps$mask) * df_imp)
m_use = lm(Y ~., data =df_use)
m_use2 = lm(Y ~., data =df)
m_use3 = lm(Y ~., data =df_imp)
preds = predict(m_use3, newdata = df_test, interval = "prediction")
mse = mean((df_test$Y > preds[,1])^2)
cov_PI = mean(df_test$Y > preds[,2] & df_test$Y < preds[,3])

test = hint_matrix(hint_rate = 0.5, no = 5, dim = 6)
df_mask[1:5, ] * test

torch_tensor(df_mask[1:5, ] ) * torch_tensor(test)

df_mask = amp %>%
  replace(!is.na(.), 1) %>% 
  replace(is.na(.), 0) %>% 
  as.matrix()

df_mask[1:5, ]
amp[1:5, ]

D_loss(df_mask,imps$d_out)
imps$d_out
as.array(imps$d_out)

#H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
#H_mb = M_mb * H_mb_temp

# add test noise
mse_use = imps$mse

df_imp_new = df_imp + matrix(rnorm(nrow(df_imp) *ncol(df_imp), 0, 0.15 *sqrt(mse_use)),
                             nrow= nrow(df_imp), ncol = ncol(df_imp))

cor(df_imp_new)

# maybe look at the MSE for rows that are similair?

# Step 5: Simulate
out <- expand.grid(
  sim = 1:100,
  Ntrain = c(100, 200, 1000),
  propmis = c(0.35, 0.70),
  hidden_dim = c(10, 20),
  patmiss = c("diag","upper.tri")
)

library(parallel)
library(pbapply)

funcs <- ls()
cl <- parallel::makeCluster(parallel::detectCores() - 4)
parallel::clusterExport(cl, varlist = funcs)

clusterEvalQ(cl, {
  library(dplyr)
  library(torch)
  library(tibble)
})

out <- cbind(out, pbsapply(1:nrow(out), \(i) {
  out <- wrap(N = out[i,"Ntrain"], N_test = 1000, P = 5, rho = 0.5,
              R2 = 0.5, propmiss = out[i,"propmis"], hidden_dim = out[i,"hidden_dim"],
              lr = 0.001, n_iter = 10000, hint_rate = 0.05, m = 1, patmiss = out[i,"patmiss"])
}, cl = cl, simplify = FALSE) |>
  do.call(what = rbind))

save(out, file = "Results/run_21_03_2025.RData")
load("Results/run_14_03_2025.RData")

## Results

out %>% 
  filter(Ntrain == 100, propmis == 0.35, hidden_dim == 10, patmiss == "diag") %>% 
  group_by(rowname) %>% 
  summarise(coverage = mean(cover),
            bias = mean(bias))


out %>% 
  filter(hidden_dim == 10) %>% 
  group_by(Ntrain, propmis, hidden_dim, patmiss) %>% 
  summarise(mse_imp = mean(mse_imp),
            mse = mean(mse),
            cov_PI = cov_PI) %>% 
  ggplot(aes(x = as.factor(Ntrain), y = mse)) +
  geom_point() +
  facet_wrap(propmis~patmiss) +
  theme_minimal()

out %>% 
  filter(hidden_dim == 10) %>% 
  group_by(sim, Ntrain, propmis, hidden_dim, patmiss) %>% 
  summarise(mse_imp = mean(mse_imp),
            mse = mean(mse),
            cov_PI = mean(cov_PI)) %>% 
  ggplot(aes(y = cov_PI, group = as.factor(Ntrain ))) +
  geom_boxplot() +
  geom_hline(yintercept = 0.95) +
  facet_wrap(propmis~patmiss) +
  theme_minimal()
