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

smooth_mode <- function(x, width=3){
  zoo::rollapply(data = x, 
                 align = "center",
                 width = width,
                 FUN = collapse::fmode, 
                 # partial = TRUE keeps the ends to be same length
                 partial=TRUE)
}


# Plotting ----------------------------------------------------------------
# This is not a multi-taper spectrogram but it's OK
spectro <- function(data, sf, nfft=1024, window=256, overlap=128, t0=0, plot_spec = T, normalize = F, return_data = F, ...){
  
  # create spectrogram
  spec = signal::specgram(x = data,
                          n = nfft,
                          Fs = sf,
                          window = window,
                          overlap = overlap
  )
  
  # discard phase info
  S = as.data.frame(abs(spec$S))
  # normalize
  if(normalize){
    S = S/max(S)  
  }
  
  # name S
  names(S) <- spec$t
  # add freq
  S$f = spec$f
  
  # pivot longer
  S <- S %>% pivot_longer(-f, names_to = "time", values_to = "power") %>% 
    mutate(time = as.numeric(time))
  
  # config time axis
  if (is.numeric(t0)) {
    S <- mutate(S, time = time + t0)
  } else if (is.POSIXct(t0)){
    S <- mutate(S, time = as.POSIXct(time, origin = t0))
  } else{
    stop("t0 must be either `numeric` or `POSIXct`")
  }
  
  out_plot <- 
    ggplot(S, aes(time, f, fill = power)) + 
    geom_tile(show.legend = F) +
    scale_fill_viridis_c(...) +
    labs(y = "Freq (Hz)") 
  return(out_plot)
}

plot_trace <- function(df, x, y, trange, ...){
  p1 <- ggplot(df %>% 
                 filter(data.table::between({{x}},
                                            trange[1],
                                            trange[2])
                 ),
               aes(x={{x}}, y={{y}})) + 
    geom_line(...) + 
    xlab("")
  return(p1)
}

get_ethogram <- function(data, x, behaviour, sampling_period = NULL){
  if (is.null(sampling_period)){
    cli::cli_alert_warning("`sampling_period` not provided.")
    sampling_period <- min(diff(dplyr::pull(data, {{x}})))
    cli::cli_inform("Sampling period estimated to {sampling_period} using min difference between observations")
  }
  data <- dplyr::select(data, x = {{x}}, behaviour = {{behaviour}}) 
  etho <- data %>% 
    dplyr::mutate(run_id = vctrs::vec_identify_runs(behaviour)) %>% 
    dplyr::group_by(run_id) %>% 
    dplyr::summarise(behaviour = base::unique(behaviour), 
                     xend = dplyr::last(x) + sampling_period, 
                     x = dplyr::first(x), 
                     duration = xend - x,
                     .groups = "keep") %>% 
    dplyr::select(run_id, x, xend, behaviour, duration)
  return(etho)
}

filter_between_join_behavior <- function(photometry_data, sleep_data, .x) {
  photometry_data %>%  
    filter(data.table::between(aligned_time_sec, 
                               lower = max(0, .x$x - t_delta), 
                               # don't add anything here
                               upper = min(max_t, .x$xend))) %>%
    mutate(rel_time = aligned_time_sec - dplyr::first(aligned_time_sec) - t_delta) %>%
    left_join(select(sleep_data, aligned_time_sec, sleep), 
              # we need a rolling join here because the two time columns will not be identical (numerical precision)
              # photometry time >= behavior time is key to avoid off by-one errors
              by = join_by(closest(aligned_time_sec >= aligned_time_sec)))
}
