functions {

  vector constrain_a(vector a_unconstrained, vector bound_type, vector lb, vector ub) {

  /*

  bound_type:
    0 = no bounds
    1 = lower bound
    2 = upper bound
    3 = both upper and lower

  */


  int L = num_elements(a_unconstrained);
  vector[L] a_constrained;

  for(i in 1:L) {

    if(bound_type[i] == 0) {
      a_constrained[i] = a_unconstrained[i];
    } else if(bound_type[i] == 1) {
      a_constrained[i] = fmax(lb[i], a_unconstrained[i]);
    } else if(bound_type[i] == 2) {
      a_constrained[i] = fmin(ub[i],  a_unconstrained[i]);
    } else {
      a_constrained[i] = fmin(ub[i], fmax(lb[i], a_unconstrained[i]));
    }

  }

  return(a_constrained);
  }

  matrix conditional_dist_joint_normal(vector mean_state, vector mean_obs, vector obs, matrix var_state, matrix var_obs, matrix covar_state_obs) {

    int m = dims(mean_state)[1];
    matrix[m, m+1] ret;
    vector[m] mn;
    matrix[m, m] v;

    mn = mean_state + covar_state_obs * inverse(var_obs) * (obs - mean_obs);
    v = var_state - covar_state_obs * inverse(var_obs) * covar_state_obs';

    ret[,1] = mn;
    ret[,2:m+1] = v;

    return(ret);
  }

  matrix update_state(
    vector y,
    vector x,
    matrix Px,
    matrix Zt_vec,
    matrix H
  ) {

    int p = dims(y)[1];
    int m = dims(x)[1];
    vector[p] mean_obs;
    matrix[m, p] covar_state_obs;
    matrix[p, p] var_obs;
    matrix[m, m+1] update;

    mean_obs = Zt_vec * x;
    covar_state_obs = Px * Zt_vec';
    var_obs = Zt_vec * Px * Zt_vec' + H;

    update = conditional_dist_joint_normal(
      x,
      mean_obs,
      y,
      Px,
      var_obs,
      covar_state_obs
    );

    return(update);

  }

  matrix update_x(vector y,
    vector x,
    vector A,
    matrix Px,
    matrix PA,
    matrix Zt_vec,
    matrix lambda_vec,
    matrix H
  ) {

    int p = dims(y)[1];
    int m = dims(x)[1];
    vector[p] mean_obs;
    matrix[m, p] covar_state_obs;
    matrix[p, p] var_obs;
    matrix[m, m+1] update;

    mean_obs = lambda_vec * A + Zt_vec * x;
    covar_state_obs = Px * Zt_vec';
    var_obs = lambda_vec * PA * lambda_vec' + Zt_vec * Px * Zt_vec' + H;

    update = conditional_dist_joint_normal(
      x,
      mean_obs,
      y,
      Px,
      var_obs,
      covar_state_obs
    );

    return(update);

  }

  matrix update_A(
    vector y,
    vector x,
    vector A,
    matrix Px,
    matrix PA,
    matrix Zt_vec,
    matrix lambda_vec,
    matrix H) {

    int p = num_elements(y);
    int m = num_elements(x);
    vector[p] mean_obs;
    matrix[m, p] covar_state_obs;
    matrix[p, p] var_obs;
    matrix[m, m+1] update;

    mean_obs = lambda_vec * A + Zt_vec * x;
    covar_state_obs = PA* lambda_vec';
    var_obs = lambda_vec * PA * lambda_vec' + Zt_vec * Px * Zt_vec' + H;

    update = conditional_dist_joint_normal(
      A,
      mean_obs,
      y,
      PA,
      var_obs,
      covar_state_obs
    );

    return(update);

  }

  matrix ssm(
    matrix Q, // m x m
    matrix H, // scalar
    matrix Z, // n x m
    matrix T, // transition matrix // given
    matrix y,
    int n_in_sample) {

      int n = dims(y)[1];
      int p = dims(y)[2];
      int m = dims(Z)[2];
      int n_cols_return = 2 + p + 2*m;
      int n_rows_return = n + m;
      matrix[n_rows_return, n_cols_return] ret = rep_matrix(0, n_rows_return, n_cols_return); // this is the big matrix we'll return
      matrix[n, m] states = rep_matrix(0, n, m);
      matrix[n, m] adstocks = rep_matrix(0, n, m);
      matrix[n, p] predictions = rep_matrix(0, n, p);
      matrix[n, 1] loglik = rep_matrix(0, n, 1);
      vector[m] x;
      matrix[m, m] Px;
      row_vector [m] Zt;
      vector[p] mean_obs;
      matrix[p, p] var_obs;
      matrix[m, m+1] upd_x;
      matrix[n, p] var_obs_v = rep_matrix(0, n, p);
      vector[m] x1     = rep_vector(0, m);
      matrix[m, m] Px1 = rep_matrix(0, m, m);
      real reference_point = y[1,1] / Z[1,1];

      if(reference_point <= 0) {
        reference_point = mean(y[, 1] ./ Z[, 1]);
      }

      if(reference_point <= 0) {
        reference_point = y[1,1] / 5;
      }


      x1[1] = reference_point / 10;
      x1[2] = y[1,1];
      x1[3] = 0;
      x1[4:m] = rep_vector(0, m-3);
      Px1[1,1] = reference_point / 5;
      Px1[2,2] = y[1,1]/4;
      Px1[3,3] = y[1,1]/10;
      for(i in 4:m) Px1[i,i] = y[1,1]/4;

      x = x1;
      Px = Px1;


      for(t in 1:n) {

        Zt = Z[t, ];

        if(t > 1) {

          // have to ensure that we don't update x before applying the adstock rates
          x = T * x;
          Px = T * Px * T' + Q;

        }

        predictions[t, 1] = max([Zt * x, 0]);



        if(t <= n_in_sample) {
          // likelihood

          mean_obs[1] = Zt * x;
          var_obs = Zt * Px * Zt' + H;
          loglik[t, 1] = normal_lpdf(y[t, 1] | mean_obs[1], sqrt(var_obs[1, 1]));
          var_obs_v[t, 1] = var_obs[1,1];


          upd_x  = update_state(to_vector(y[t,]), x, Px, to_matrix(Zt), H);

          x  = upd_x[, 1];
          Px = upd_x[, 2:m+1];
        }


        // record states
        states[t, ] = x';


      }

    ret[1:n, 1:1]  = loglik;
    ret[1:n, 2:(p+1)] = predictions;
    ret[1:n, (p+2):(m+p+1)] = states;
    ret[(n+1):(n+m), (p+2):(m+p+1)] = Px;
    ret[1:n, (2*m+p+2):(2*m+p+2)] = var_obs_v;
    return(ret);
  }



  matrix custom_adstock_filter_states(
    vector x1, // column vector
    matrix Px1, // m x m
    vector A1, // column vector
    matrix PA1, // m x m
    matrix Q, // m x m
    matrix R, // m x m
    matrix H, // scalar -- going to set this equal to 0
    matrix Z, // n x m
    matrix T, // transition matrix // given
    matrix lambda_vec, // 1 x m
    matrix y,
    vector bound_type,
    vector lb,
    vector ub,
    int n_in_sample) {

      int n = dims(y)[1];
      int p = dims(y)[2];
      int m = dims(Z)[2];
      int n_cols_return = 2 + p + 3*m;
      int n_rows_return = n + m;
      matrix[n_rows_return, n_cols_return] ret = rep_matrix(0, n_rows_return, n_cols_return); // this is the big matrix we'll return
      matrix[n, m] states = rep_matrix(0, n, m);
      matrix[n, m] adstocks = rep_matrix(0, n, m);
      matrix[n, p] predictions = rep_matrix(0, n, p);
      matrix[n, 1] loglik = rep_matrix(0, n, 1);
      vector[m] x;
      vector[m] A;
      matrix[n, m] Px_diag;
      matrix[m, m] Px;
      matrix[m, m] PA;
      matrix[m, m] lambda_m = diag_matrix(to_vector(lambda_vec));
      row_vector [m] Zt;
      row_vector [m] Ztprev;
      matrix[m, m] Zt_m;
      matrix[m, m] Ztprev_m;
      vector[p] mean_obs;
      matrix[p, p] var_obs;
      matrix[m, m+1] upd_x;
      matrix[m, m+1] upd_A;
      matrix[n, p] var_obs_v = rep_matrix(0, n, p);
      matrix[n, m] cumulative_spend = rep_matrix(0, n, m);
      matrix[m, m] Q_adj = Q;

      x = x1;
      Px = Px1;
      A = A1;
      PA = PA1;

      for(chan in 1:m) {
        for(t in 1:n) {
          cumulative_spend[t, chan] = sum(Z[1:t, chan]);
        }
      }


      for(t in 1:n) {

        Zt = Z[t, ];

        if(t > 1) {

          Ztprev = Z[t-1, ];

          Zt_m = diag_matrix(to_vector(Zt));
          Ztprev_m = diag_matrix(to_vector(Ztprev));

          // have to ensure that we don't update x before applying the adstock rates
          A = lambda_m * A + Ztprev_m * x;
          // x = T * x;
          x = x;



          // have to update PA before Px before
          // PA depends on Px{t-1}
          PA = lambda_m * PA * lambda_m' + Ztprev_m * Px * Ztprev_m' + R;

          // adjust Q s.t. if a channel hasn't started yet, we don't add uncertainty
          Q_adj = Q; // reset to eliminate previous period's adjustments
          for(chan in 1:m) {
            if(cumulative_spend[t, chan] == 0) {
              Q_adj[chan, chan] = 0;
            }
          }

          // Px = T * Px * T' + Q_adj;
          Px = Px + Q_adj;

        }

        predictions[t, ] = to_row_vector(lambda_vec * A + Zt * x);



        if(t <= n_in_sample) {
          // likelihood

          mean_obs = lambda_vec * A + Zt * x;
          var_obs = lambda_vec * PA * lambda_vec' + Zt * Px * Zt' + H;
          loglik[t, 1] = normal_lpdf(y[t, 1] | mean_obs[1], sqrt(var_obs[1, 1]));
          var_obs_v[t, 1] = var_obs[1,1];


          upd_x  = update_x(to_vector(y[t,]), x, A, Px, PA, to_matrix(Zt), lambda_vec, H);
          x  = upd_x[, 1];
          Px = upd_x[, 2:m+1];
          upd_A  = update_A(to_vector(y[t, ]), x, A, Px, PA, to_matrix(Zt), lambda_vec, H);
          A  = upd_A[, 1];
          PA = upd_A[, 2:m+1];

          x  = constrain_a(x, bound_type, lb, ub);
        }


        // record states
        states[t, ] = x';
        adstocks[t, ] = A';
        Px_diag[t, ] = diagonal(Px)';

      }

    ret[1:n, 1:1]  = loglik;
    ret[1:n, 2:(p+1)] = predictions;
    ret[1:n, (p+2):(m+p+1)] = states;
    ret[1:n, (m+p+2):(2*m+p+1)] = adstocks;
    ret[(n+1):(n+m), (p+2):(m+p+1)] = Px;
    ret[(n+1):(n+m), (m+p+2):(2*m+p+1)] = PA;
    ret[1:n, (2*m+p+2):(2*m+p+2)] = var_obs_v;
    ret[1:n, (2*m+p+3):(2*m+p+2+m)] = Px_diag;



    return(ret);
  }
}

data {
  int<lower=0> N; // number of data points
  int<lower=0> P; // number of time series of data
  int<lower=0> K; // number of trends
  int<lower=0> nZ; // number of unique z elements
  int<lower=0> row_indx[nZ];
  int<lower=0> col_indx[nZ];
  int<lower=0> nVariances;
  int<lower=0> varIndx[P];
  int<lower=0> nZero;
  int<lower=0> row_indx_z[nZero];
  int<lower=0> col_indx_z[nZero];
  int<lower=0> n_pos; // number of non-missing observations
  int<lower=0> row_indx_pos[n_pos]; // row indices of non-missing obs
  int<lower=0> col_indx_pos[n_pos]; // col indices of non-missing obs
  real y[n_pos]; // vectorized matrix of observations
  int<lower=0> n_na; // number of missing observations
  int<lower=0> row_indx_na[n_na]; // row indices of missing obs
  int<lower=0> col_indx_na[n_na]; // col indices of missing obs
  real<lower=1> nu_fixed; // df on student-t
  int estimate_nu; // Estimate degrees of freedom?
  int use_normal; // flag, for large values of nu > 100, use normal instead
  int est_cor; // whether to estimate correlation in obs error (=1) or not (=0)
  int est_phi; // whether to estimate autocorrelation in trends (=1) or not (= 0)
  int est_theta; // whether to estimate moving-average in trends (=1) or not (= 0
  int<lower=0> num_obs_covar; // number of unique observation covariates, dimension of matrix
  int<lower=0> n_obs_covar; // number of unique covariates included
  int obs_covar_index[num_obs_covar,3] ;// indexed by time, trend, covariate #, covariate value. +1 because of indexing issues
  real obs_covar_value[num_obs_covar];
  int<lower=0> num_pro_covar; // number of unique process covariates, dimension of matrix
  int<lower=0> n_pro_covar; // number of unique process covariates included
  int pro_covar_index[num_pro_covar,3] ;// indexed by time, trend, covariate #, covariate value. +1 because of indexing issues
  real pro_covar_value[num_pro_covar];

  // ssm part
  matrix[N, 1] depvar;
  matrix[N, P] predictors;
  int n_holdout;

  // channel expectations
  vector[P+K] x1_exo;
  vector[P+K] Px1_vector;
  vector[P+K] sigma_channels_ub;
  //// state constraints
  vector[P+K] state_bound_type;
  vector[P+K] state_lb;
  vector[P+K] state_ub;
  //// adstock constraints
  int<lower=0, upper=1>  rate_index[P+K];
  vector[P+K] rate_lb;
  vector[P+K] rate_ub;

  // constraints
  //// parameter constraints;
  real sigma_adstock_ub;

  real measurement_exo_ub;
}
transformed data {
  int n_pcor; // dimension for cov matrix
  int n_loglik; // dimension for loglik calculation
  vector[K] zeros;
  int smooth = 1;
  int n_cols_return_exo    = 2 + 1 + 3*(P+K);
  int n_rows_return_exo    = N + (P+K);
  int n_in_sample          = N - n_holdout;
  int n_adstocked_channels = sum(rate_index);
  matrix[P+K, P+K] Tr_exo  = diag_matrix(rep_vector(1, P+K));
  matrix[P+K, P+K] Px1_exo = diag_matrix(Px1_vector);
  vector[P+K] A1           = rep_vector(0, P+K);
  matrix[P+K, P+K] PA1     = rep_matrix(0, P+K, P+K);

  for(i in 1:(P+K)) {
    if(rate_index[i] == 1) {
      A1[i]     = 0; // Z_exo[1, i] / 10;
      if(i <= P) {
        PA1[i, i] = predictors[1, i] / 10;
      }
    } else {
      A1[i]     = 0;
      PA1[i, i] = 0;
    }
  }

  for(k in 1:K) {
    zeros[k] = 0; // used in MVT / MVN below
  }

  if(est_cor == 0) {
     n_loglik = P * N;
  } else {
    n_loglik = N; // TODO: likely needs to be fixed
  }

  if(est_cor == 0) {
    n_pcor = P;
    if(nVariances < 2) {
      n_pcor = 2;
    }
  } else {
    n_pcor = P;
  }
}
parameters {
  matrix[K,N-1] devs; // random deviations of trends
  vector[K] x0; // initial state
  vector<lower=0>[K] psi; // expansion parameters
  vector[nZ] z; // estimated loadings in vec form
  vector[K] zpos; // constrained positive values
  matrix[n_obs_covar, P] b_obs; // coefficients on observation model
  matrix[n_pro_covar, K] b_pro; // coefficients on process model
  real<lower=0> sigma[nVariances];
  real<lower=2> nu[estimate_nu]; // df on student-t
  real ymiss[n_na];
  real<lower=-1,upper=1> phi[est_phi*K];
  real<lower=-1,upper=1> theta[est_theta*K];
  cholesky_factor_corr[n_pcor] Lcorr;
  // our part
  vector<lower=0, upper=1>[n_adstocked_channels] rates_raw;
  real<lower=0, upper=measurement_exo_ub> measurement_exo;
  vector<lower=0, upper=1>[P+K] sigma_channels;
  vector<lower=0, upper=sigma_adstock_ub>[n_adstocked_channels]    sigma_adstock;
}
transformed parameters {
  matrix[P,N] pred; //vector[P] pred[N];
  matrix[P,K] Z;
  //vector[N] yall[P]; // combined vectors of missing and non-missing values
  matrix[P,N] yall;
  vector[P] sigma_vec;
  vector[K] phi_vec; // for AR(1) part
  vector[K] theta_vec; // for MA(1) part
  matrix[K,N] x; //vector[N] x[P]; // random walk-trends
  vector[K] indicator; // indicates whether diagonal is neg or pos
  vector[K] psi_root; // derived sqrt(expansion parameter psi)
  matrix[n_rows_return_exo,  n_cols_return_exo]  ssm_results_exo;
  vector[P+K] rates;
  matrix[1, 1] H_exo = rep_matrix(1, 1, 1);
  matrix[P+K, P+K] Q_exo  = rep_matrix(1, P+K, P+K);
  matrix[P+K, P+K] R_exo  = rep_matrix(0, P+K, P+K);
  matrix[N, 1] y_pred;
  matrix[N, P+K] concat_predictors;



  // phi is the ar(1) parameter, fixed or estimated
  if(est_phi == 1) {
    for(k in 1:K) {phi_vec[k] = phi[k];}
  } else {
    for(k in 1:K) {phi_vec[k] = 1;}
  }

  // theta is the ma(1) parameter, fixed or estimated
  if(est_theta == 1) {
    for(k in 1:K) {theta_vec[k] = theta[k];}
  } else {
    for(k in 1:K) {theta_vec[k] = 0;}
  }

  for(p in 1:P) {
    sigma_vec[p] = sigma[varIndx[p]]; // convert estimated sigmas to vec form
  }

  // Fill yall with non-missing values
  for(i in 1:n_pos) {
    yall[row_indx_pos[i], col_indx_pos[i]] = y[i];
  }
  // Include missing observations
  if(n_na > 0) {
    for(i in 1:n_na) {
      yall[row_indx_na[i], col_indx_na[i]] = ymiss[i];
    }
  }

  for(i in 1:nZ) {
    Z[row_indx[i],col_indx[i]] = z[i]; // convert z to from vec to matrix
  }
  // fill in zero elements in upper diagonal
  if(nZero > 2) {
    for(i in 1:(nZero-2)) {
      Z[row_indx_z[i],col_indx_z[i]] = 0;
    }
  }

  for(k in 1:K) {
    Z[k,k] = zpos[k];// add constraint for Z diagonal
  }

  // this block is for the expansion prior
  for(k in 1:K) {
    if(zpos[k] < 0) {
      indicator[k] = -1;
    } else {
      indicator[k] = 1;
    }
    psi_root[k] = sqrt(psi[k]);
    for(p in 1:P) {
      Z[p,k] = Z[p,k] * indicator[k] * (1/psi_root[k]);
    }
  }

  // initial state for each trend
  for(k in 1:K) {
    x[k,1] = x0[k];
    // trend is modeled as random walk, with optional
    // AR(1) component = phi, and optional MA(1) component
    // theta. Theta is included in the model block below.
    for(t in 2:N) {
      x[k,t] = phi_vec[k]*x[k,t-1] + devs[k,t-1];
    }
  }
  // this block also for the expansion prior, used to convert trends
  for(k in 1:K) {
    //  x[k,1:N] = x[k,1:N] * indicator[k] * psi_root[k];
    for(t in 1:N) {
      x[k,t] = x[k,t] * indicator[k] * psi_root[k];
    }
  }

  // adjust predictions if process covariates exist
  if(num_pro_covar > 0) {
    for(i in 1:num_pro_covar) {
      // indexed by time, trend, covariate #, covariate value
      x[pro_covar_index[i,2],pro_covar_index[i,1]] = x[pro_covar_index[i,2],pro_covar_index[i,1]] + b_pro[pro_covar_index[i,3], pro_covar_index[i,2]] * pro_covar_value[i];
    }
  }

  // N is sample size, P = time series, K = number trends
  // [PxN] = [PxK] * [KxN]
  pred = Z * x;

  // adjust predictions if observation covariates exist
  if(num_obs_covar > 0) {
    for(i in 1:num_obs_covar) {
      // indexed by time, trend, covariate #, covariate value
      pred[obs_covar_index[i,2],obs_covar_index[i,1]] = pred[obs_covar_index[i,2],obs_covar_index[i,1]] + b_obs[obs_covar_index[i,3], obs_covar_index[i,2]] * obs_covar_value[i];
    }
  }

  H_exo  = diag_matrix([measurement_exo]');
  Q_exo  = diag_matrix(sigma_channels .* sigma_channels_ub);

  { // spreading the raw rates to the rates vector
    int counter = 1;
    for(i in 1:(P+K)) {
      if(rate_index[i] == 1) {
        rates[i]   = rates_raw[counter] * (rate_ub[i] - rate_lb[i]) + rate_lb[i];
        R_exo[i,i] = sigma_adstock[counter];
        counter += 1;
      } else {
        rates[i] = 0;
        R_exo[i,i] = 0;
      }
    }
  }

  concat_predictors = append_col(x', predictors);

  ssm_results_exo = custom_adstock_filter_states(
    x1_exo,
    Px1_exo,
    A1,
    PA1,
    Q_exo,
    R_exo,
    H_exo,
    concat_predictors,
    Tr_exo,
    to_matrix(rates'),
    depvar,
    state_bound_type, state_lb, state_ub,
    n_in_sample
  );

  y_pred = ssm_results_exo[1:N, 2:(1+1)];
}
model {

  // initial state for each trend
  x0 ~ normal(0, 1); // initial state estimate at t=1
  psi ~ gamma(2, 1); // expansion parameter for par-expanded priors

  // This is deviations - either normal or Student t, and
  // if Student-t, df parameter nu can be estimated or fixed
  for(k in 1:K) {
    if(use_normal == 0) {
      for(t in 1:1) {
        if (estimate_nu == 1) {
          devs[k,t] ~ student_t(nu[1], 0, 1); // random walk
        } else {
          devs[k,t] ~ student_t(nu_fixed, 0, 1); // random walk
        }
      }
      for(t in 2:(N-1)) {
        // if MA is not included, theta_vec = 0
        if (estimate_nu == 1) {
          devs[k,t] ~ student_t(nu[1], theta_vec[k]*devs[k,t-1], 1); // random walk
        } else {
          devs[k,t] ~ student_t(nu_fixed, theta_vec[k]*devs[k,t-1], 1); // random walk
        }
      }
    } else {
      devs[k,1] ~ normal(0, 1);
      for(t in 2:(N-1)) {
        // if MA is not included, theta_vec = 0
        devs[k,t] ~ normal(theta_vec[k]*devs[k,t-1], 1);
      }
    }

  }

  // prior for df parameter for t-distribution
  if (estimate_nu == 1) {
    nu[1] ~ gamma(2, 0.1);
  }
  // prior on AR(1) component if included
  if(est_phi == 1) {
    phi ~ uniform(0,1); // K elements
  }
  // prior on MA(1) component if included
  if(est_theta == 1) {
    theta ~ uniform(0,1); // K elements
  }

  // prior on loadings
  z ~ normal(0, 1);
  zpos ~ normal(0, 1);// diagonal

  // observation variance
  sigma ~ student_t(3, 0, 2);
  if(est_cor == 1) {
    Lcorr ~ lkj_corr_cholesky(1);
  }

  // likelihood for independent
  if(est_cor == 0) {
    for(i in 1:P){
      target += normal_lpdf(yall[i] | pred[i], sigma_vec[i]);
    }
  } else {
    // need to loop over time slices / columns - each ~ MVN
    for(i in 1:N) {
      target += multi_normal_cholesky_lpdf(col(yall,i) | col(pred,i), diag_pre_multiply(sigma_vec, Lcorr));
    }
  }

  sigma_channels ~ uniform(0, 1);
  sigma_adstock  ~ uniform(0, sigma_adstock_ub);

  for(i in 1:n_adstocked_channels) rates_raw[i] ~ normal(.5, .2)T[0, 1];

  measurement_exo  ~ uniform(0, measurement_exo_ub);

  target += sum(ssm_results_exo[1:n_in_sample, 1:1]);
}
generated quantities {
  vector[n_loglik] log_lik;
  matrix[n_pcor, n_pcor] Omega;
  matrix[n_pcor, n_pcor] Sigma;
  int<lower=0> j;

  j = 0;

  if(est_cor == 1) {
    Omega = multiply_lower_tri_self_transpose(Lcorr);
    Sigma = quad_form_diag(Omega, sigma_vec);
  }

  // calculate pointwise log_lik for loo package:
  if(est_cor == 0) {
    j = 0;
    for(n in 1:N) {
      for(p in 1:P) {
        j = j + 1;
        log_lik[j] = normal_lpdf(yall[p,n] | pred[p,n], sigma_vec[p]);
      }
    }
  } else {
    // TODO: this needs to be fixed:
    for(i in 1:N) {
      log_lik[i] = multi_normal_cholesky_lpdf(col(yall,i) | col(pred,i), diag_pre_multiply(sigma_vec, Lcorr));
    }
  }

}
