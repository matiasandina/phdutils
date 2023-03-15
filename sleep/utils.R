# This function helps import sleep labels saved into .mat by Accusleep
# the labels are buried inside the $labels as an nx1 matrix, we want to output a vector
import_mat_labels <- function(filepath){
  return(as.vector(R.matlab::readMat(filepath)$labels))
}

convert_accusleep_labels <- function(col){
  return(
    case_when(col == 1 ~ "REM",
              col == 2 ~ "Wake",
              col == 3 ~ "NREM")
    )
  }

make_time_column <- function(sf, ...){
  return(seq(from = 0, by = 1/sf, ... ))
}

seconds_to_hours <- function(x){x/3600}


plot_eeg_array <- function(ap, ml){
  df <- dplyr::arrange(tidyr::expand_grid(ap, ml), desc(ap))
  df$label <- paste("channel",1:nrow(df)) 
  ggplot(df, aes(x=ml, y=ap)) +
    geom_hline(yintercept = 0, lty = 2) +
    geom_vline(xintercept = 0, lty = 2) +
    geom_point(pch=19, size = 5) +
    geom_text(aes(label=label), size = 5, nudge_y = 0.6) +
    scale_x_continuous(limits = c(-3.5, 3.5), )+
    scale_y_continuous(limits = c(-5, 5))
}
