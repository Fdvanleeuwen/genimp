## Try to implement GAIn in torch
library(torch)
library(mice)
library(MASS)


df = MASS::mvrnorm(n = 100, mu = c(0,0), Sigma = matrix(c(1, 0.8, 0.8, 1), ncol = 2, nrow = 2))

# add some missing values
df_miss = mice::ampute(df)

# Data matrix
df_data = df_miss$amp %>% 
  replace(is.na(.), 0)
df_data_tens = torch_tensor(df_data)

# Random matrix
df_rand = df_miss$amp %>% 
  replace(!is.na(.), 0) %>% 
  replace(is.na(.), rnorm(n = sum(is.na(df_miss$amp))))
df_rand_tens = torch_tensor(df_rand)

df_combine_tens = df_data_tens + df_rand_tens
dim(df_combine_tens) # this is the imput dim, output of the generatior is dim/2

# Mask matrix
df_mask = df_miss$amp %>%
  replace(!is.na(.), 1) %>% 
  replace(is.na(.), 0)
df_mask_tens = torch_tensor(df_mask)

# this is the input for the generator
gen_inp = torch_cat(list(df_combine_tens, df_mask_tens), dim = 2)

# This function generates a "hint matrix," typically used in data imputation algorithms.
# The matrix is binary, indicating whether a value is included in the hint.
# Parameters:
# - hint_rate: Probability of including a value in the hint.
# - no: Number of rows in the generated matrix.
# - dim: Number of columns in the generated matrix.
hint_matrix <- function(hint_rate, no, dim) {
  # Create a probability matrix of random numbers between 0 and 1.
  prob_matrix <- matrix(runif(no * dim), no, dim)
  # Generate a binary hint matrix based on the hint rate.
  hint <- ifelse((prob_matrix < hint_rate), 1, 0)
  return(hint)
}

hint_mat = torch_tensor(hint_matrix(hint_rate = 0.05, no = nrow(df_mask), dim = ncol(df_mask)))

# Build the generator 
gen <- nn_module(
  "Generator",
  initialize = function(input_dim, hiddem_dim) {
    self$fc1 <- nn_linear(input_dim, hiddem_dim)
    self$fc2 <- nn_linear(hiddem_dim, hiddem_dim)
    self$fc3 <- nn_linear(hiddem_dim, input_dim/2)
    
    # Custom initialization
    #nn_init_xavier_uniform_(self$fc1$weight)
    #nn_init_xavier_uniform_(self$fc2$weight)
    #nn_init_xavier_uniform_(self$fc3$weight)
    
    #nn_init_constant_(self$fc1$bias, 0)
    #nn_init_constant_(self$fc2$bias, 0)
    #nn_init_constant_(self$fc3$bias, 0)
  },
  
  forward = function(x){
    x |> 
      self$fc1() |>
      nnf_relu() |>
      self$fc2() |>
      nnf_relu() |>
      self$fc3()  # Sigmoid activation for stardardised data
  }
)

discrim <- nn_module(
  "Discriminator",
  initialize = function(input_dim, hiddem_dim) {
    self$fc1 <- nn_linear(input_dim, hiddem_dim)
    self$fc2 <- nn_linear(hiddem_dim, hiddem_dim)
    self$fc3 <- nn_linear(hiddem_dim, 1)
  },
  
  forward = function(x){
    x |> 
      self$fc1() |>
      nnf_relu() |>
      self$fc2() |>
      nnf_relu() |>
      self$fc3() |>
      nnf_sigmoid()
  }
)

## loss functions
D_loss <- function(mask, D_pred){
  -torch_mean(mask * torch_log(D_pred + 1e-8) + (1- mask) * torch_log(1 - D_pred + 1e-8))
}

G_loss <- function(mask, D_pred, G_input, G_pred, alpha){
  
  # Loss term for fooling the discriminator
  G_loss_temp <- -torch_mean((1 - mask) * torch_log(D_pred + 1e-8))
  
  # Mean squared error (MSE) for observed data
  MSE_loss <- torch_mean((mask * G_input - mask * G_pred)**2) / torch_mean(mask)
  
  # Total generator loss
  G_loss_temp + alpha * MSE_loss
}

# check the generator
# First, create an instance of the generator
#input_dim <- dim(gen_inp)[2]  
#hidden_dim <- 4  

# Initialize the generator
#generator <- gen(input_dim, hidden_dim)

# Make a forward pass
#output <- generator(gen_inp)

# Now 'output' contains the generated data
# The shape should be [batch_size, input_dim/2]
#print(output$shape)

# check the discriminator
# First, create an instance of the discriminator
#input_dim_d <- output$shape[2]

# Initialize the discriminator
#discriminator <- discrim(input_dim_d, hidden_dim)

#output_D <- discriminator(output)
#output_D$shape

# now the loss functions
#G_loss(mask = df_mask_tens, D_pred = output_D, G_input = df_combine_tens, G_pred = output, alpha = 1)
#D_loss(mask = df_mask_tens, D_pred = output_D)


# This seems to work, now we need to train the model

# Initialize the gen/discrim
input_dim <- dim(gen_inp)[2]  
hidden_dim <- 10  
input_dim_d <- input_dim

generator <- gen(input_dim, hidden_dim)
discriminator <- discrim(input_dim_d, hidden_dim)

# We need to define the optimizer
g_optimizer <- optim_adam(generator$parameters, lr = 0.001, betas = c(0.5, 0.999))
d_optimizer <- optim_adam(discriminator$parameters, lr = 0.001, betas = c(0.5, 0.999))
  
n_iter = 1000
D_loss_all = c()
G_loss_all = c()

# training the model

for (i in 1:n_iter){
  # DISCRIMINATOR STEP
  d_optimizer$zero_grad()
  
  # Generate fake data
  output_G <- generator(gen_inp)
  
  # add hint matrix
  
  D_input = torch_cat(list(output_G$detach(), hint_mat), dim = 2)
  
  # Discriminator predictions on fake data
  fake_pred <- discriminator(D_input)  # Detach to avoid backprop through generator
  
  # Discriminator loss
  d_loss_val <- D_loss(mask = df_mask_tens, D_pred = fake_pred)
  
  # Backpropagation and optimization for discriminator
  d_loss_val$backward()
  d_optimizer$step()
  
  # GENERATOR STEP
  g_optimizer$zero_grad()
  
  # Generate fake data again (important!)
  output_G <- generator(gen_inp)
  
  # Discriminator predictions on new fake data
  D_input2 = torch_cat(list(output_G, hint_mat), dim = 2) # No detach here!
  output_D <- discriminator(D_input2)  
  
  # Generator loss
  g_loss_val <- G_loss(mask = df_mask_tens, D_pred = output_D, 
                       G_input = df_combine_tens, G_pred = output_G, alpha = 1)
  
  # Back prop and optimization for generator
  g_loss_val$backward()
  g_optimizer$step()
  
  # Save losses
  D_loss_all = c(D_loss_all, as.numeric(d_loss_val$item()))
  G_loss_all = c(G_loss_all, as.numeric(g_loss_val$item()))
}

out_df = data.frame(niter = 1:n_iter, D_loss = D_loss_all, G_loss = G_loss_all)

out_df %>% 
  ggplot(aes(x = niter)) +
  geom_line(aes(y = D_loss, color = "D loss")) +
  geom_line(aes(y = G_loss, color = "G loss")) +
  theme_minimal()

# RMSE
sum(((1 - df_mask) * df - (1 - df_mask) * output_G)^2) / sum(1 - df_mask)

# plot the missing vs real data
real = df * (1 - df_mask)
imputed = as_array((1 - df_mask) * output_G)

helper_missing = data.frame(df_mask) %>% 
  mutate(miss_ind = case_when(X1 == 1 & X2 == 1 ~ "None",
                              X1 == 1 & X2 == 0 ~ "X1",
                              X1 == 0 & X2 == 1 ~ "X2",
                              X1 == 0 & X2 == 0 ~ "Both"))

df_plot = data.frame(real_x1 = df[,1],
                     real_x2 =df[,2],
                     imputed_x1 = as_array(output_G)[,1],
                     imputed_x2 = as_array(output_G)[,2]) %>% 
  mutate(miss_ind = helper_missing$miss_ind)

p1 = df_plot %>% 
  ggplot(aes(x = real_x1, y = real_x2)) +
  geom_point(aes(color = miss_ind)) +
  theme_minimal() + 
  labs(x = "X1", y = "X2", title = "Real and masked data") 

p2 = df_plot %>% 
  ggplot(aes(x = imputed_x1, y = imputed_x2)) +
  geom_point(aes(color = miss_ind)) +  
  theme_minimal() + 
  labs(x = "X1", y = "X2", title = "Real and Imputed data")

library(patchwork)

p1 + p2 +plot_layout(guides = "collect") & theme(legend.position = 'bottom')

# TO DO
# mini batch
# Normalization
# initialization
# dropout
# add real data to discriminator loss?



