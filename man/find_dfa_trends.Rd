% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/find_dfa_trends.R
\name{find_dfa_trends}
\alias{find_dfa_trends}
\title{Find the best number of trends according to LOOIC}
\usage{
find_dfa_trends(y = y, kmin = 1, kmax = 5, iter = 2000, thin = 1,
  compare_normal = FALSE, convergence_threshold = 1.05,
  variance = c("equal", "unequal"), ...)
}
\arguments{
\item{y}{A matrix of data to fit. Columns represent time element.}

\item{kmin}{Minimum number of trends, defaults to 1.}

\item{kmax}{Maximum number of trends, defaults to 5.}

\item{iter}{Iterations when sampling from each Stan model, defaults to 2000.}

\item{thin}{Thinning rate when sampling from each Stan model, defaults to 1.}

\item{compare_normal}{If \code{TRUE}, does model selection comparison of Normal vs.
Student-t errors}

\item{convergence_threshold}{The maximum allowed value of Rhat to determine
convergence of parameters}

\item{variance}{Vector of variance arguments for searching over large groups
of models. Can be either or both of ("equal","unequal")}

\item{...}{Other arguments to pass to \code{fit_dfa()}}
}
\description{
Fit a DFA with different number of trends and return the leave one out (LOO)
value as calculated by the \link[loo:loo-package]{loo} package.
}
\examples{
\donttest{
set.seed(42)
s <- sim_dfa(num_trends = 2, num_years = 20, num_ts = 3)
# only 1 chain and 180 iterations used so example runs quickly:
m <- find_dfa_trends(
  y = s$y_sim, iter = 180,
  kmin = 1, kmax = 2, chains = 1, compare_normal = FALSE,
  variance = "equal", convergence_threshold = 1.1,
  control = list(adapt_delta = 0.95, max_treedepth = 20))
m$summary
m$best_model
}
}
