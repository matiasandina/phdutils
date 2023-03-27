# airPLS functions just in
devtools::source_url("https://raw.githubusercontent.com/zmzhang/airPLS_R/master/R/airPLS.R")

clean_cut_labels <- function(cut_labels){
  mat <- stringr::str_split(string = stringr::str_sub(cut_labels, start = 2, end = -2), 
                            pattern = ",",
                            simplify = T,
                            n = 2)
  mat <- apply(mat, 2, as.numeric)
  colnames(mat) <- c("low", "high")
  return(mat)
}
